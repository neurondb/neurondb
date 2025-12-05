/*-------------------------------------------------------------------------
 *
 * llm_image_utils.c
 *    Image processing utilities for LLM vision operations
 *
 * Provides image format detection, validation, preprocessing, and
 * metadata extraction for vision-capable LLM models.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/llm/llm_image_utils.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "lib/stringinfo.h"
#include "utils/jsonb.h"
#include "neurondb_llm.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Image format detection constants */
#define IMAGE_MAGIC_PNG_1    0x89
#define IMAGE_MAGIC_PNG_2    0x50
#define IMAGE_MAGIC_PNG_3    0x4E
#define IMAGE_MAGIC_PNG_4    0x47
#define IMAGE_MAGIC_JPEG_1   0xFF
#define IMAGE_MAGIC_JPEG_2   0xD8
#define IMAGE_MAGIC_GIF_1    0x47
#define IMAGE_MAGIC_GIF_2    0x49
#define IMAGE_MAGIC_WEBP_1   0x52
#define IMAGE_MAGIC_WEBP_2   0x49
#define IMAGE_MAGIC_WEBP_3   0x46
#define IMAGE_MAGIC_WEBP_4   0x46

/* Maximum image size (100MB) */
#define MAX_IMAGE_SIZE (100 * 1024 * 1024)

/* Minimum image size (100 bytes) */
#define MIN_IMAGE_SIZE 100

/* Maximum dimensions for vision models */
#define MAX_IMAGE_DIMENSION 8192

/* ImageFormat and ImageMetadata are now defined in neurondb_llm.h */

/*
 * ndb_detect_image_format
 *    Detect image format from magic bytes
 */
static ImageFormat
ndb_detect_image_format(const unsigned char *data, size_t size)
{
	if (size < 8)
		return IMAGE_FORMAT_UNKNOWN;

	/* PNG: 89 50 4E 47 0D 0A 1A 0A */
	if (size >= 8 &&
		data[0] == IMAGE_MAGIC_PNG_1 &&
		data[1] == IMAGE_MAGIC_PNG_2 &&
		data[2] == IMAGE_MAGIC_PNG_3 &&
		data[3] == IMAGE_MAGIC_PNG_4 &&
		data[4] == 0x0D && data[5] == 0x0A &&
		data[6] == 0x1A && data[7] == 0x0A)
		return IMAGE_FORMAT_PNG;

	/* JPEG: FF D8 FF */
	if (size >= 3 &&
		data[0] == IMAGE_MAGIC_JPEG_1 &&
		data[1] == IMAGE_MAGIC_JPEG_2 &&
		data[2] == IMAGE_MAGIC_JPEG_1)
		return IMAGE_FORMAT_JPEG;

	/* GIF: 47 49 46 38 (GIF8) */
	if (size >= 4 &&
		data[0] == IMAGE_MAGIC_GIF_1 &&
		data[1] == IMAGE_MAGIC_GIF_2 &&
		data[2] == 0x46 &&
		(data[3] == 0x38 || data[3] == 0x39))
		return IMAGE_FORMAT_GIF;

	/* WebP: RIFF...WEBP */
	if (size >= 12 &&
		data[0] == IMAGE_MAGIC_WEBP_1 &&
		data[1] == IMAGE_MAGIC_WEBP_2 &&
		data[2] == IMAGE_MAGIC_WEBP_3 &&
		data[3] == IMAGE_MAGIC_WEBP_4)
	{
		/* Check for WEBP signature at offset 8 */
		if (size >= 12 &&
			data[8] == 'W' && data[9] == 'E' &&
			data[10] == 'B' && data[11] == 'P')
			return IMAGE_FORMAT_WEBP;
	}

	/* BMP: BM */
	if (size >= 2 && data[0] == 'B' && data[1] == 'M')
		return IMAGE_FORMAT_BMP;

	return IMAGE_FORMAT_UNKNOWN;
}

/*
 * ndb_get_image_mime_type
 *    Get MIME type for image format
 */
static const char *
ndb_get_image_mime_type(ImageFormat format)
{
	switch (format)
	{
		case IMAGE_FORMAT_JPEG:
			return "image/jpeg";
		case IMAGE_FORMAT_PNG:
			return "image/png";
		case IMAGE_FORMAT_GIF:
			return "image/gif";
		case IMAGE_FORMAT_WEBP:
			return "image/webp";
		case IMAGE_FORMAT_BMP:
			return "image/bmp";
		default:
			return "image/jpeg";	/* Default fallback */
	}
}

/*
 * ndb_parse_png_dimensions
 *    Parse PNG dimensions from IHDR chunk
 */
static bool
ndb_parse_png_dimensions(const unsigned char *data,
						 size_t size,
						 int *width,
						 int *height)
{
	/* PNG IHDR chunk starts at offset 16 */
	if (size < 24)
		return false;

	/* IHDR chunk: width (4 bytes) at offset 16, height (4 bytes) at offset 20 */
	*width = (data[16] << 24) | (data[17] << 16) | (data[18] << 8) | data[19];
	*height = (data[20] << 24) | (data[21] << 16) | (data[22] << 8) | data[23];

	return (*width > 0 && *width <= MAX_IMAGE_DIMENSION &&
			*height > 0 && *height <= MAX_IMAGE_DIMENSION);
}

/*
 * ndb_parse_jpeg_dimensions
 *    Parse JPEG dimensions from SOF markers (simplified)
 *    Note: This is a simplified parser. Full JPEG parsing would require
 *    handling multiple SOF markers and progressive JPEGs.
 */
static bool
ndb_parse_jpeg_dimensions(const unsigned char *data,
						  size_t size,
						  int *width,
						  int *height)
{
	const unsigned char *p = data;
	const unsigned char *end = data + size;

	/* Skip JPEG header (FF D8) */
	if (size < 4 || p[0] != 0xFF || p[1] != 0xD8)
		return false;

	p += 2;

	/* Search for SOF markers (Start of Frame) */
	while (p < end - 8)
	{
		if (p[0] == 0xFF)
		{
			unsigned char marker = p[1];

			/* SOF markers: 0xC0-0xC3, 0xC5-0xC7, 0xC9-0xCB, 0xCD-0xCF */
			if ((marker >= 0xC0 && marker <= 0xC3) ||
				(marker >= 0xC5 && marker <= 0xC7) ||
				(marker >= 0xC9 && marker <= 0xCB) ||
				(marker >= 0xCD && marker <= 0xCF))
			{
				/* Height (2 bytes) at offset 5, Width (2 bytes) at offset 7 */
				if (p + 9 <= end)
				{
					*height = (p[5] << 8) | p[6];
					*width = (p[7] << 8) | p[8];
					if (*width > 0 && *width <= MAX_IMAGE_DIMENSION &&
						*height > 0 && *height <= MAX_IMAGE_DIMENSION)
						return true;
				}
			}
			p++;
		}
		else
		{
			p++;
		}
	}

	return false;
}

/*
 * ndb_validate_image
 *    Comprehensive image validation
 */
ImageMetadata *
ndb_validate_image(const unsigned char *data, size_t size, MemoryContext mctx)
{
	ImageMetadata *meta;
	MemoryContext oldctx;

	if (mctx == NULL)
		mctx = CurrentMemoryContext;

	oldctx = MemoryContextSwitchTo(mctx);
	meta = (ImageMetadata *) palloc0(sizeof(ImageMetadata));
	MemoryContextSwitchTo(oldctx);

	meta->size = size;
	meta->is_valid = false;

	/* Check minimum size */
	if (size < MIN_IMAGE_SIZE)
	{
		meta->error_msg = pstrdup("Image too small (minimum 100 bytes)");
		return meta;
	}

	/* Check maximum size */
	if (size > MAX_IMAGE_SIZE)
	{
		meta->error_msg = pstrdup("Image too large (maximum 100MB)");
		return meta;
	}

	/* Detect format */
	meta->format = ndb_detect_image_format(data, size);
	if (meta->format == IMAGE_FORMAT_UNKNOWN)
	{
		meta->error_msg = pstrdup("Unknown or unsupported image format");
		return meta;
	}

	/* Get MIME type */
	meta->mime_type = pstrdup(ndb_get_image_mime_type(meta->format));

	/* Parse dimensions based on format */
	switch (meta->format)
	{
		case IMAGE_FORMAT_PNG:
			if (!ndb_parse_png_dimensions(data, size, &meta->width, &meta->height))
			{
				meta->error_msg = pstrdup("Failed to parse PNG dimensions");
				return meta;
			}
			meta->channels = 4; /* RGBA for PNG */
			break;

		case IMAGE_FORMAT_JPEG:
			if (!ndb_parse_jpeg_dimensions(data, size, &meta->width, &meta->height))
			{
				meta->error_msg = pstrdup("Failed to parse JPEG dimensions");
				return meta;
			}
			meta->channels = 3; /* RGB for JPEG */
			break;

		case IMAGE_FORMAT_GIF:
		case IMAGE_FORMAT_WEBP:
		case IMAGE_FORMAT_BMP:

			/*
			 * Dimensions not parsed for these formats (would require full
			 * parser)
			 */
			meta->width = 0;
			meta->height = 0;
			meta->channels = 0;
			break;

		default:
			meta->error_msg = pstrdup("Unsupported image format for dimension parsing");
			return meta;
	}

	/* Validate dimensions if parsed */
	if (meta->width > 0 && meta->height > 0)
	{
		if (meta->width > MAX_IMAGE_DIMENSION || meta->height > MAX_IMAGE_DIMENSION)
		{
			meta->error_msg = pstrdup("Image dimensions exceed maximum (8192x8192)");
			return meta;
		}
	}

	meta->is_valid = true;
	return meta;
}

/*
 * ndb_get_image_format_name
 *    Get human-readable format name
 */
const char *
ndb_get_image_format_name(ImageFormat format)
{
	switch (format)
	{
		case IMAGE_FORMAT_JPEG:
			return "JPEG";
		case IMAGE_FORMAT_PNG:
			return "PNG";
		case IMAGE_FORMAT_GIF:
			return "GIF";
		case IMAGE_FORMAT_WEBP:
			return "WebP";
		case IMAGE_FORMAT_BMP:
			return "BMP";
		default:
			return "Unknown";
	}
}

/*
 * ndb_image_metadata_to_json
 *    Convert image metadata to JSON string
 */
char *
ndb_image_metadata_to_json(const ImageMetadata *meta)
{
	StringInfoData json;
	char	   *format_name;

	if (meta == NULL)
		return pstrdup("{}");

	initStringInfo(&json);
	format_name = pstrdup(ndb_get_image_format_name(meta->format));

	appendStringInfo(&json, "{");
	appendStringInfo(&json, "\"format\":\"%s\",", format_name);
	appendStringInfo(&json, "\"mime_type\":\"%s\",", meta->mime_type ? meta->mime_type : "");
	appendStringInfo(&json, "\"size\":%zu,", meta->size);
	appendStringInfo(&json, "\"width\":%d,", meta->width);
	appendStringInfo(&json, "\"height\":%d,", meta->height);
	appendStringInfo(&json, "\"channels\":%d,", meta->channels);
	appendStringInfo(&json, "\"is_valid\":%s", meta->is_valid ? "true" : "false");
	if (meta->error_msg)
	{
		appendStringInfo(&json, ",\"error\":\"%s\"", meta->error_msg);
	}
	appendStringInfo(&json, "}");

	NDB_FREE(format_name);
	return json.data;
}
