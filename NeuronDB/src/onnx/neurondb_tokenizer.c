/*-------------------------------------------------------------------------
 *
 * neurondb_tokenizer.c
 *    Highly detailed tokenizer implementation for BERT-style HuggingFace models.
 *
 * This module provides a BERT-style tokenizer for preprocessing text prior
 * to ONNX model inference. It includes lowercasing, special token handling,
 * whitespace and rudimentary WordPiece tokenization, and vocabulary lookup.
 *
 * Supported Features:
 *    - Lowercasing (English-language normalization)
 *    - Handles [CLS], [SEP], [PAD], [UNK], [MASK]
 *    - Simple hash-based vocabulary table for fast lookups
 *    - Tokenization: whitespace split, with facility for future WordPiece extension
 *    - Attention mask generation
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 * SPDX-License-Identifier: PostgreSQL
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "neurondb_onnx.h"

#include <ctype.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* External declarations */
extern char *neurondb_onnx_model_path;

/* ---- SPECIAL TOKEN IDs (BERT standard) ---- */
#define TOKEN_PAD_ID 0			/* [PAD] */
#define TOKEN_UNK_ID 100		/* [UNK] */
#define TOKEN_CLS_ID 101		/* [CLS] */
#define TOKEN_SEP_ID 102		/* [SEP] */
#define TOKEN_MASK_ID 103		/* [MASK] */

/* ---- Sequence length limits ---- */
#define MAX_SEQ_LENGTH 512

/* ---- Vocabulary table implementation ---- */
typedef struct VocabEntry
{
	char	   *token;			/* Owned string in TopMemoryContext */
	int32		token_id;		/* Vocabulary index */
	struct VocabEntry *next;	/* For chaining on hash collision */
}			VocabEntry;

/* ---- Tokenizer cache entry ---- */
typedef struct TokenizerCacheEntry
{
	char	   *model_name;		/* Model identifier */
	VocabEntry **vocab_table;	/* Vocabulary hash table */
	bool		vocab_loaded;	/* Whether vocabulary is loaded */
	int32		vocab_size;		/* Vocabulary size */
	char	   *vocab_path;		/* Path to vocabulary file */
	time_t		last_used;		/* Last access time for LRU */
	struct TokenizerCacheEntry *next;	/* Linked list pointer */
}			TokenizerCacheEntry;

#define VOCAB_HASH_SIZE 65536
#define MAX_MODEL_NAME 256
#define MAX_TOKENIZER_CACHE 10

/* Global tokenizer cache */
static TokenizerCacheEntry * g_tokenizer_cache_head = NULL;
static int	g_tokenizer_cache_count = 0;

/* Default vocabulary (used when no model-specific vocab is loaded) */
static VocabEntry * g_default_vocab_table[VOCAB_HASH_SIZE] =
{
	0
};
static bool g_default_vocab_loaded = false;

/**
 * hash_token
 *    Lightweight hash for token-string to vocabulary index.
 *    Not cryptographically secure, but stable and simple.
 */
static uint32
hash_token(const char *token)
{
	uint32		hash = 0;

	for (const char *p = token; *p != '\0'; ++p)
		hash = hash * 31 + (unsigned char) *p;
	return hash % VOCAB_HASH_SIZE;
}

/**
 * vocab_add_token
 *    Safely adds a token and its associated token_id to a vocabulary table.
 */
static void
vocab_add_token(VocabEntry * *vocab_table, const char *token, int32 token_id)
{
	uint32		hash = hash_token(token);
	VocabEntry *entry = (VocabEntry *) MemoryContextAlloc(
														  TopMemoryContext, sizeof(VocabEntry));

	/* Copy the token string into TopMemoryContext as well */
	entry->token = MemoryContextStrdup(TopMemoryContext, token);
	entry->token_id = token_id;
	entry->next = vocab_table[hash];
	vocab_table[hash] = entry;
}

/**
 * vocab_lookup_token
 *    Returns the integer token_id for a provided string token.
 *    If not found, RETURN TOKEN_UNK_ID.
 */
static int32
vocab_lookup_token(VocabEntry * *vocab_table, const char *token)
{
	uint32		hash = hash_token(token);

	for (VocabEntry * entry = vocab_table[hash]; entry != NULL;
		 entry = entry->next)
	{
		if (strcmp(entry->token, token) == 0)
			return entry->token_id;
	}
	return TOKEN_UNK_ID;
}

/**
 * vocab_lookup_id
 *    Returns the token string for a provided token_id.
 *    Returns NULL if not found.
 *
 * NOTE: This uses a linear search through all entries.
 *       For better performance, consider using a reverse mapping (token_id -> token string).
 */
static const char *
vocab_lookup_id(VocabEntry * *vocab_table, int32 token_id)
{
	int32		i;

	/*
	 * Linear search through all entries (could be optimized with reverse
	 * mapping)
	 */
	for (i = 0; i < VOCAB_HASH_SIZE; i++)
	{
		for (VocabEntry * entry = vocab_table[i]; entry != NULL;
			 entry = entry->next)
		{
			if (entry->token_id == token_id)
				return entry->token;
		}
	}
	return NULL;
}

/**
 * load_vocab_file
 *    Loads a vocab.txt file, as exported by HuggingFace or other BERT-compatible models.
 *    Each line is a token. Line number = token_id.
 */
static int32
load_vocab_file(VocabEntry * *vocab_table, const char *vocab_path)
{
	FILE	   *fp;
	char		line[512];
	int32		token_id = 0;

	fp = fopen(vocab_path, "r");
	if (fp == NULL)
	{
		int			save_errno = errno;

		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open vocabulary file: %s "
						"(%s)",
						vocab_path,
						strerror(save_errno))));
	}

	while (fgets(line, sizeof(line), fp) != NULL)
	{
		size_t		len = strlen(line);

		/* Trim trailing newline and possible carriage return */
		if (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
			line[--len] = '\0';
		if (len > 0 && (line[len - 1] == '\r'))
			line[--len] = '\0';

		/* Skip empty lines in vocab.txt */
		if (line[0] == '\0')
			continue;

		vocab_add_token(vocab_table, line, token_id++);
	}

	if (fclose(fp) != 0)
	{
		int			saved_errno = errno;

		ereport(WARNING,
				(errcode_for_file_access(),
				 errmsg("could not close vocabulary file \"%s\" (%s)",
						vocab_path, strerror(saved_errno))));
	}

	elog(LOG,
		 "loaded vocabulary with %d tokens from \"%s\"",
		 token_id,
		 vocab_path);
	return token_id;
}

/**
 * find_tokenizer_cache_entry
 *    Finds a tokenizer cache entry by model name.
 */
static TokenizerCacheEntry *
find_tokenizer_cache_entry(const char *model_name)
{
	TokenizerCacheEntry *entry;

	for (entry = g_tokenizer_cache_head; entry != NULL; entry = entry->next)
	{
		if (strcmp(entry->model_name, model_name) == 0)
		{
			entry->last_used = time(NULL);
			return entry;
		}
	}
	return NULL;
}

/**
 * evict_lru_tokenizer
 *    Evicts the least recently used tokenizer from the cache.
 */
static void
evict_lru_tokenizer(void)
{
	TokenizerCacheEntry *entry,
			   *prev,
			   *lru_entry,
			   *lru_prev;
	time_t		oldest;

	if (!g_tokenizer_cache_head)
		return;

	/* Find the LRU entry */
	lru_entry = g_tokenizer_cache_head;
	lru_prev = NULL;
	oldest = lru_entry->last_used;
	prev = NULL;
	for (entry = g_tokenizer_cache_head; entry;
		 prev = entry, entry = entry->next)
	{
		if (entry->last_used < oldest)
		{
			oldest = entry->last_used;
			lru_entry = entry;
			lru_prev = prev;
		}
	}

	/* Remove from linked list */
	if (lru_prev)
		lru_prev->next = lru_entry->next;
	else
		g_tokenizer_cache_head = lru_entry->next;

	/* Free vocabulary table (only if it's not the default vocabulary) */
	if (lru_entry->vocab_table
		&& lru_entry->vocab_table != g_default_vocab_table)
	{
		int32		i;

		for (i = 0; i < VOCAB_HASH_SIZE; i++)
		{
			VocabEntry *vocab_entry,
					   *next;

			for (vocab_entry = lru_entry->vocab_table[i];
				 vocab_entry;
				 vocab_entry = next)
			{
				next = vocab_entry->next;
				if (vocab_entry->token)
					NDB_FREE(vocab_entry->token);
				NDB_FREE(vocab_entry);
			}
		}
		NDB_FREE(lru_entry->vocab_table);
	}

	/* Free cache entry */
	if (lru_entry->model_name)
		NDB_FREE(lru_entry->model_name);
	if (lru_entry->vocab_path)
		NDB_FREE(lru_entry->vocab_path);
	NDB_FREE(lru_entry);

	g_tokenizer_cache_count--;
}

/**
 * get_or_load_tokenizer
 *    Gets or loads a tokenizer for a model.
 */
static TokenizerCacheEntry *
get_or_load_tokenizer(const char *model_name)
{
	TokenizerCacheEntry *entry;
	char		vocab_path[MAXPGPATH];
	int32		vocab_size;

	/* Check cache first */
	entry = find_tokenizer_cache_entry(model_name);
	if (entry)
		return entry;

	/* Evict LRU if cache is full */
	if (g_tokenizer_cache_count >= MAX_TOKENIZER_CACHE)
		evict_lru_tokenizer();

	/* Create new cache entry */
	entry = (TokenizerCacheEntry *) MemoryContextAllocZero(
														   TopMemoryContext, sizeof(TokenizerCacheEntry));
	entry->model_name = MemoryContextStrdup(TopMemoryContext, model_name);
	entry->vocab_table = (VocabEntry * *) MemoryContextAllocZero(
																 TopMemoryContext, VOCAB_HASH_SIZE * sizeof(VocabEntry *));
	entry->last_used = time(NULL);

	/* Try to load vocabulary from model directory */
	if (neurondb_onnx_model_path)
	{
		snprintf(vocab_path,
				 MAXPGPATH,
				 "%s/%s/vocab.txt",
				 neurondb_onnx_model_path,
				 model_name);

		/* Check if vocab file exists */
		if (access(vocab_path, R_OK) == 0)
		{
			entry->vocab_path = MemoryContextStrdup(
													TopMemoryContext, vocab_path);
			vocab_size =
				load_vocab_file(entry->vocab_table, vocab_path);
			entry->vocab_size = vocab_size;
			entry->vocab_loaded = true;
		}
		else
		{
			/* Use default vocabulary if model-specific vocab not found */
			entry->vocab_table = g_default_vocab_table;
			entry->vocab_loaded = g_default_vocab_loaded;
			entry->vocab_size = 0;
		}
	}
	else
	{
		/* Use default vocabulary if model path not set */
		entry->vocab_table = g_default_vocab_table;
		entry->vocab_loaded = g_default_vocab_loaded;
		entry->vocab_size = 0;
	}

	/* Add to cache */
	entry->next = g_tokenizer_cache_head;
	g_tokenizer_cache_head = entry;
	g_tokenizer_cache_count++;

	return entry;
}

/**
 * _to_lower_case
 *    Converts a string (in-place) to lower case, only if the char is in A-Z.
 *    This is in-place: modifies the input string.
 */
static void
_to_lower_case(char *src)
{
	for (char *p = src; *p; ++p)
	{
		if (*p >= 'A' && *p <= 'Z')
			*p = *p + ('a' - 'A');
	}
}

/**
 * tokenize_text
 *    Tokenizes incoming UTF-8 string based on whitespace.
 *    Converts to lower case (by default) for basic BERT models.
 *    Returns a palloc'd char** array of palloc'd null-terminated strings.
 *    The number of tokens is stored in (*num_tokens).
 *
 * NOTE: This does not include advanced tokenization like punctuation splitting, WordPiece, CJK, etc.
 *       See HuggingFace Tokenizers for production tokenization.
 */
static char **
tokenize_text(const char *text, int *num_tokens)
{
	char	   *text_copy;
	char	   *token;
	char	  **tokens;			/* Array of pointers to token strings */
	int			count = 0;
	int			capacity = 128;
	char	   *saveptr = NULL;

	Assert(text != NULL);
	Assert(num_tokens != NULL);

	/* Copy the source text since strtok_r will mutate it */
	text_copy = pstrdup(text);

	/* Lowercase transformation for BERT-style preprocessing */
	_to_lower_case(text_copy);

	/* Preallocate token pointer array */
	tokens = (char **) palloc(capacity * sizeof(char *));
	count = 0;

	/* Tokenize by whitespace: space, tab, newline, and carriage return */
	token = strtok_r(text_copy, " \t\n\r", &saveptr);
	while (token != NULL)
	{
		/* Expand array if necessary */
		if (count >= capacity)
		{
			capacity *= 2;
			tokens = (char **) repalloc(
										tokens, capacity * sizeof(char *));
		}
		tokens[count++] = pstrdup(token);
		token = strtok_r(NULL, " \t\n\r", &saveptr);
	}

	/* Free the temporary, mutable copy */
	NDB_FREE(text_copy);

	*num_tokens = count;
	return tokens;
}

/**
 * neurondb_tokenize
 *    Complete procedure for converting input text to an ID sequence suitable for model input.
 *    - Tokenizes and normalizes input
 *    - Adds [CLS] at the start, [SEP] at the end
 *    - Maps each token to a vocab index or to [UNK] if unknown
 *    - Pads the result to 'max_length' with [PAD]
 *    - Returns a palloc'd int32 array of full length
 *    - Fills *output_length with the produced sequence length (always 'max_length')
 *
 * Arguments:
 *    - text:        UTF-8 null-terminated C string to tokenize
 *    - max_length:  Max sequence length, typically 128, 256, or 512
 *    - output_length: pointer to int32 for returning the length written (== max_length)
 */
int32 *
neurondb_tokenize(const char *text, int32 max_length, int32 * output_length)
{
	return neurondb_tokenize_with_model(
										text, max_length, output_length, NULL);
}

/**
 * neurondb_tokenize_with_model
 *    Tokenize text with a specific model's tokenizer.
 *    Uses cached tokenizer if available, otherwise loads from model directory.
 */
int32 *
neurondb_tokenize_with_model(const char *text,
							 int32 max_length,
							 int32 * output_length,
							 const char *model_name)
{
	char	  **tokens = NULL;	/* Array of palloc'd char* */
	int32	   *token_ids = NULL;	/* Output array */
	int			num_tokens = 0;
	int			output_idx = 0;
	int			i;
	TokenizerCacheEntry *tokenizer = NULL;
	VocabEntry **vocab_table = NULL;
	bool		vocab_loaded = false;

	Assert(text != NULL);
	Assert(output_length != NULL);
	Assert(max_length > 1);		/* Provide at least space for [CLS], [SEP] */

	/* Get or load tokenizer for model */
	if (model_name != NULL && strlen(model_name) > 0)
	{
		tokenizer = get_or_load_tokenizer(model_name);
		if (tokenizer)
		{
			vocab_table = tokenizer->vocab_table;
			vocab_loaded = tokenizer->vocab_loaded;
		}
	}

	/* Fall back to default vocabulary if no model-specific tokenizer */
	if (!vocab_table)
	{
		vocab_table = g_default_vocab_table;
		vocab_loaded = g_default_vocab_loaded;
	}

	/*
	 * Tokenize text. tokens is a palloc'd array of palloc'd strings.
	 * num_tokens: number of tokens in the result.
	 */
	tokens = tokenize_text(text, &num_tokens);

	/*
	 * Allocate output array, zeroed (so all trailing positions are [PAD]). We
	 * always write max_length elements, zero-initialized.
	 */
	token_ids = (int32 *) palloc0(max_length * sizeof(int32));

	/* Add [CLS] at beginning */
	output_idx = 0;
	token_ids[output_idx++] = TOKEN_CLS_ID;

	/*
	 * Add as many content tokens as fit, but reserve one slot for [SEP]. If
	 * tokens don't fit, silently truncate.
	 */
	for (i = 0; i < num_tokens && output_idx < max_length - 1; ++i)
	{
		int32		token_id;

		if (vocab_loaded)
		{
			token_id = vocab_lookup_token(vocab_table, tokens[i]);
		}
		else
		{
			token_id = TOKEN_UNK_ID;
		}
		token_ids[output_idx++] = token_id;
		NDB_FREE(tokens[i]); /* Free each token string */
	}
	/* Free the tokens array itself */
	NDB_FREE(tokens);

	/* Always add [SEP] if there is room */
	if (output_idx < max_length)
		token_ids[output_idx++] = TOKEN_SEP_ID;

	/* All remaining positions are [PAD] (== 0), already set by palloc0 */

	*output_length = max_length;
	return token_ids;
}

/**
 * neurondb_detokenize
 *    Convert token IDs back to text string.
 *    Uses model-specific tokenizer if provided, otherwise uses default vocabulary.
 */
char *
neurondb_detokenize(const int32 * token_ids,
					int32 length,
					const char *model_name)
{
	StringInfoData buf;
	int32		i;
	TokenizerCacheEntry *tokenizer = NULL;
	VocabEntry **vocab_table = NULL;
	bool		vocab_loaded = false;
	const char *token_str;

	Assert(token_ids != NULL);
	Assert(length > 0);

	/* Get or load tokenizer for model */
	if (model_name != NULL && strlen(model_name) > 0)
	{
		tokenizer = get_or_load_tokenizer(model_name);
		if (tokenizer)
		{
			vocab_table = tokenizer->vocab_table;
			vocab_loaded = tokenizer->vocab_loaded;
		}
	}

	/* Fall back to default vocabulary if no model-specific tokenizer */
	if (!vocab_table)
	{
		vocab_table = g_default_vocab_table;
		vocab_loaded = g_default_vocab_loaded;
	}

	initStringInfo(&buf);

	/* Convert token IDs to text */
	for (i = 0; i < length; i++)
	{
		int32		token_id = token_ids[i];

		/* Skip special tokens */
		if (token_id == TOKEN_PAD_ID || token_id == TOKEN_CLS_ID
			|| token_id == TOKEN_SEP_ID
			|| token_id == TOKEN_MASK_ID)
		{
			continue;
		}

		/* Lookup token string */
		if (vocab_loaded)
		{
			token_str = vocab_lookup_id(vocab_table, token_id);
			if (token_str)
			{
				if (buf.len > 0)
					appendStringInfoChar(&buf, ' ');
				appendStringInfoString(&buf, token_str);
			}
			else
			{
				/* Unknown token - use [UNK] */
				if (buf.len > 0)
					appendStringInfoChar(&buf, ' ');
				appendStringInfoString(&buf, "[UNK]");
			}
		}
		else
		{
			/* No vocabulary loaded - just use token ID */
			if (buf.len > 0)
				appendStringInfoChar(&buf, ' ');
			appendStringInfo(&buf, "%d", token_id);
		}
	}

	return buf.data;
}

/**
 * neurondb_create_attention_mask
 *    Create attention mask (int32[]), value 1 for any real token (not [PAD]), 0 for [PAD] token.
 *    Sequence length matches input length.
 *    Used during inference to tell model which positions are valid tokens.
 */
int32 *
neurondb_create_attention_mask(int32 * token_ids, int32 length)
{
	int32	   *mask;
	int			i;

	Assert(token_ids != NULL);
	Assert(length > 0);

	mask = (int32 *) palloc(length * sizeof(int32));
	for (i = 0; i < length; ++i)
	{
		/* If not a [PAD] ID, mask value = 1, else 0. */
		mask[i] = (token_ids[i] != TOKEN_PAD_ID) ? 1 : 0;
	}

	return mask;
}

/* Forward declaration for SQL function */
PG_FUNCTION_INFO_V1(neurondb_hf_tokenize);

/**
 * neurondb_hf_tokenize
 *    SQL-callable function to tokenize text.
 *    Returns an array of token IDs.
 *
 * Versions:
 *    1. neurondb_hf_tokenize(model_name, text, max_length)
 *    2. neurondb_hf_tokenize(text, max_length)
 *    3. neurondb_hf_tokenize(model_name, text)
 *    4. neurondb_hf_tokenize(text)
 */
Datum
neurondb_hf_tokenize(PG_FUNCTION_ARGS)
{
	text	   *model_name_text = NULL;
	text	   *input_text = NULL;
	int32		max_length = 512;
	char	   *model_name = NULL;
	char	   *text_str = NULL;
	int32	   *token_ids = NULL;
	int32		output_length = 0;
	Datum	   *dvalues = NULL;
	bool	   *dnulls = NULL;
	ArrayType  *result;
	int			i;
	Oid			arg1_type = InvalidOid;

	/* Determine which overloaded version based on argument types and count */
	if (PG_NARGS() == 3)
	{
		/* Version 1: neurondb_hf_tokenize(model_name, text, max_length) */
		model_name_text = PG_ARGISNULL(0) ? NULL : PG_GETARG_TEXT_P(0);
		input_text = PG_GETARG_TEXT_P(1);
		max_length = PG_ARGISNULL(2) ? 512 : PG_GETARG_INT32(2);
	}
	else if (PG_NARGS() == 2)
	{
		arg1_type = get_fn_expr_argtype(fcinfo->flinfo, 1);

		if (arg1_type == INT4OID)
		{
			/* Version 2: neurondb_hf_tokenize(text, max_length) */
			input_text = PG_GETARG_TEXT_P(0);
			max_length = PG_GETARG_INT32(1);
			model_name_text = NULL;
		}
		else
		{
			/* Version 3: neurondb_hf_tokenize(model_name, text) */
			model_name_text =
				PG_ARGISNULL(0) ? NULL : PG_GETARG_TEXT_P(0);
			input_text = PG_GETARG_TEXT_P(1);
			max_length = 512;
		}
	}
	else if (PG_NARGS() == 1)
	{
		/* Version 4: neurondb_hf_tokenize(text) */
		input_text = PG_GETARG_TEXT_P(0);
		max_length = 512;
		model_name_text = NULL;
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb_hf_tokenize: invalid number "
						"of arguments")));
	}

	/* Check if input_text is NULL */
	if (PG_NARGS() == 3)
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
	}
	else if (PG_NARGS() == 2)
	{
		if (arg1_type == INT4OID)
		{
			if (PG_ARGISNULL(0))
				PG_RETURN_NULL();
		}
		else
		{
			if (PG_ARGISNULL(1))
				PG_RETURN_NULL();
		}
	}
	else if (PG_NARGS() == 1)
	{
		if (PG_ARGISNULL(0))
			PG_RETURN_NULL();
	}

	/* Convert input text to C string */
	text_str = text_to_cstring(input_text);
	if (model_name_text)
		model_name = text_to_cstring(model_name_text);

	/* Validate max_length */
	if (max_length < 2 || max_length > MAX_SEQ_LENGTH)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("max_length must be between 2 and %d",
						MAX_SEQ_LENGTH)));

	/* Tokenize text */
	token_ids = neurondb_tokenize_with_model(
											 text_str, max_length, &output_length, model_name);

	/* Allocate arrays for result */
	dvalues = (Datum *) palloc(output_length * sizeof(Datum));
	dnulls = (bool *) palloc(output_length * sizeof(bool));

	/* Fill arrays */
	for (i = 0; i < output_length; i++)
	{
		dvalues[i] = Int32GetDatum(token_ids[i]);
		dnulls[i] = false;
	}

	/* Create array result */
	result = construct_array(
							 dvalues, output_length, INT4OID, sizeof(int32), true, 'i');

	/* Cleanup */
	NDB_FREE(token_ids);
	NDB_FREE(dvalues);
	NDB_FREE(dnulls);
	NDB_FREE(text_str);
	if (model_name)
		NDB_FREE(model_name);

	PG_RETURN_ARRAYTYPE_P(result);
}

/* Forward declaration for SQL detokenize function */
PG_FUNCTION_INFO_V1(neurondb_hf_detokenize);

/**
 * neurondb_hf_detokenize
 *    SQL-callable function to detokenize token IDs back to text.
 *
 * Versions:
 *    1. neurondb_hf_detokenize(model_name, token_ids)
 *    2. neurondb_hf_detokenize(token_ids)
 */
Datum
neurondb_hf_detokenize(PG_FUNCTION_ARGS)
{
	text	   *model_name_text = NULL;
	ArrayType  *token_ids_array = NULL;
	char	   *model_name = NULL;
	int32	   *token_ids = NULL;
	int32		length = 0;
	char	   *text_str = NULL;
	int			i;
	Datum	   *dvalues = NULL;
	bool	   *dnulls = NULL;
	int			ndims;

	/* Handle two overloaded versions */
	if (PG_NARGS() == 2)
	{
		/* Version 1: neurondb_hf_detokenize(model_name, token_ids) */
		model_name_text = PG_ARGISNULL(0) ? NULL : PG_GETARG_TEXT_P(0);
		token_ids_array = PG_GETARG_ARRAYTYPE_P(1);
	}
	else if (PG_NARGS() == 1)
	{
		/* Version 2: neurondb_hf_detokenize(token_ids) */
		model_name_text = NULL;
		token_ids_array = PG_GETARG_ARRAYTYPE_P(0);
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb_hf_detokenize: invalid number "
						"of arguments")));
	}

	if (token_ids_array == NULL || PG_ARGISNULL(PG_NARGS() == 2 ? 1 : 0))
		PG_RETURN_NULL();

	/* Convert model name to C string */
	if (model_name_text)
		model_name = text_to_cstring(model_name_text);

	/* Extract token IDs from array */
	ndims = ARR_NDIM(token_ids_array);
	if (ndims != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("token_ids must be a one-dimensional "
						"array")));

	if (ARR_ELEMTYPE(token_ids_array) != INT4OID)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("token_ids must be an array of "
						"integers")));

	length = ARR_DIMS(token_ids_array)[0];
	if (length < 0 || length > MAX_SEQ_LENGTH)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("token_ids array length must be between "
						"0 and %d",
						MAX_SEQ_LENGTH)));

	/* Extract array values */
	deconstruct_array(token_ids_array,
					  INT4OID,
					  sizeof(int32),
					  true,
					  'i',
					  &dvalues,
					  &dnulls,
					  &length);

	/* Allocate token IDs array */
	token_ids = (int32 *) palloc(length * sizeof(int32));

	/* Copy token IDs */
	for (i = 0; i < length; i++)
	{
		if (dnulls[i])
			token_ids[i] = TOKEN_PAD_ID;
		else
			token_ids[i] = DatumGetInt32(dvalues[i]);
	}

	/* Detokenize */
	text_str = neurondb_detokenize(token_ids, length, model_name);

	/* Cleanup */
	NDB_FREE(token_ids);
	NDB_FREE(dvalues);
	NDB_FREE(dnulls);
	if (model_name)
		NDB_FREE(model_name);

	PG_RETURN_TEXT_P(cstring_to_text(text_str));
}
