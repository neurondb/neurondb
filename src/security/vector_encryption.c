/*-------------------------------------------------------------------------
 *
 * vector_encryption.c
 *		Field-level encryption for vectors and metadata
 *
 * Implements AES-256-GCM encryption for sensitive vector data at rest.
 * Provides transparent encryption/decryption with per-column key support.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  src/security/vector_encryption.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_security.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/bytea.h"
#include "utils/memutils.h"
#include "utils/guc.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_validation.h"

#include <openssl/evp.h>
#include <stdlib.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <string.h>

/* Key derivation function - uses PBKDF2 */
static void
derive_key(const char *password, size_t password_len,
		   const uint8_t *salt, size_t salt_len,
		   uint8_t *key, size_t key_len)
{
	int			ret;

	ret = PKCS5_PBKDF2_HMAC(password, password_len,
							salt, salt_len,
							10000,			/* iterations */
							EVP_sha256(),
							key_len, key);
	if (ret != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: key derivation failed")));
}

/*
 * encrypt_vector - Encrypt a vector using AES-256-GCM
 *
 * Returns encrypted data as BYTEA containing EncryptedVector structure
 */
PG_FUNCTION_INFO_V1(encrypt_vector);

Datum
encrypt_vector(PG_FUNCTION_ARGS)
{
	Vector	   *input = NULL;
	text	   *key_text = NULL;
	EncryptedVector *encrypted = NULL;
	EVP_CIPHER_CTX *ctx = NULL;
	uint8_t	   *key = NULL;
	uint8_t	   *plaintext = NULL;
	uint8_t	   *ciphertext = NULL;
	size_t		plaintext_len;
	size_t		ciphertext_len;
	size_t		result_size;
	int			len;
	int			final_len;
	char	   *key_str = NULL;
	size_t		key_str_len;
	uint8_t	   salt[16];
	bytea	   *result = NULL;

	/* Validate input */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: encrypt_vector: input vector cannot be NULL")));

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: encrypt_vector: encryption key cannot be NULL")));

	input = (Vector *) PG_GETARG_POINTER(0);
	key_text = PG_GETARG_TEXT_PP(1);

	NDB_CHECK_VECTOR_VALID(input);

	/* Get key string */
	key_str = text_to_cstring(key_text);
	key_str_len = strlen(key_str);

	if (key_str_len < 8)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: encryption key must be at least 8 characters")));

	/* Generate random salt and IV */
	if (RAND_bytes(salt, sizeof(salt)) != 1)
	{
		pfree(key_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to generate random salt")));
	}

	/* Derive encryption key from password */
	nalloc(key, uint8_t, 32);	/* AES-256 key */
	derive_key(key_str, key_str_len, salt, sizeof(salt), key, 32);

	/* Prepare plaintext */
	plaintext_len = input->dim * sizeof(float4);
	nalloc(plaintext, uint8_t, plaintext_len);
	memcpy(plaintext, input->data, plaintext_len);

	/* Allocate ciphertext (plaintext_len + auth tag) */
	ciphertext_len = plaintext_len;
	nalloc(ciphertext, uint8_t, ciphertext_len);

	/* Allocate EncryptedVector structure */
	result_size = VARHDRSZ + sizeof(EncryptedVector) - sizeof(uint8_t) + ciphertext_len + 16;	/* +16 for auth tag */
	result = (bytea *) palloc0(result_size);
	SET_VARSIZE(result, result_size);

	encrypted = (EncryptedVector *) VARDATA(result);
	{
		const char *tenant_str = GetConfigOption("neurondb.tenant_id", true, true);
		long		tid;

		if (tenant_str != NULL && *tenant_str != '\0' &&
			(tid = strtol(tenant_str, NULL, 10)) >= 0 && tid <= (long) UINT32_MAX)
			encrypted->tenant_id = (uint32) tid;
		else
			encrypted->tenant_id = 0;
	}
	encrypted->dim = input->dim;

	/* Generate random IV */
	if (RAND_bytes(encrypted->encryption_iv, sizeof(encrypted->encryption_iv)) != 1)
	{
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		pfree(ciphertext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to generate random IV")));
	}

	/* Encrypt using AES-256-GCM */
	ctx = EVP_CIPHER_CTX_new();
	if (ctx == NULL)
	{
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		pfree(ciphertext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create cipher context")));
	}

	if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, encrypted->encryption_iv) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		pfree(ciphertext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: encryption initialization failed")));
	}

	if (EVP_EncryptUpdate(ctx, encrypted->ciphertext, &len, plaintext, plaintext_len) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		pfree(ciphertext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: encryption failed")));
	}

	if (EVP_EncryptFinal_ex(ctx, encrypted->ciphertext + len, &final_len) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		pfree(ciphertext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: encryption finalization failed")));
	}

	if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, encrypted->auth_tag) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		pfree(ciphertext);
		pfree(result);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to get authentication tag")));
	}

	EVP_CIPHER_CTX_free(ctx);

	/* Cleanup sensitive data */
	explicit_bzero(key, 32);
	explicit_bzero(plaintext, plaintext_len);
	pfree(key_str);
	pfree(key);
	pfree(plaintext);
	pfree(ciphertext);

	PG_RETURN_BYTEA_P(result);
}

/*
 * decrypt_vector - Decrypt an encrypted vector
 *
 * Returns decrypted vector
 */
PG_FUNCTION_INFO_V1(decrypt_vector);

Datum
decrypt_vector(PG_FUNCTION_ARGS)
{
	bytea	   *encrypted_data = NULL;
	text	   *key_text = NULL;
	EncryptedVector *encrypted = NULL;
	EVP_CIPHER_CTX *ctx = NULL;
	uint8_t	   *key = NULL;
	uint8_t	   *plaintext = NULL;
	Vector	   *result = NULL;
	size_t		result_size;
	int			len;
	int			final_len;
	char	   *key_str = NULL;
	size_t		key_str_len;
	uint8_t	   salt[16] = {0};	/* In production, store salt separately */

	/* Validate input */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: decrypt_vector: encrypted data cannot be NULL")));

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: decrypt_vector: decryption key cannot be NULL")));

	encrypted_data = PG_GETARG_BYTEA_PP(0);
	key_text = PG_GETARG_TEXT_PP(1);

	if (VARSIZE_ANY(encrypted_data) < sizeof(EncryptedVector))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid encrypted vector format")));

	encrypted = (EncryptedVector *) VARDATA_ANY(encrypted_data);

	/* Get key string */
	key_str = text_to_cstring(key_text);
	key_str_len = strlen(key_str);

	/* Derive decryption key (in production, use stored salt) */
	nalloc(key, uint8_t, 32);
	derive_key(key_str, key_str_len, salt, sizeof(salt), key, 32);

	/* Allocate plaintext */
	nalloc(plaintext, uint8_t, encrypted->dim * sizeof(float4));

	/* Decrypt using AES-256-GCM */
	ctx = EVP_CIPHER_CTX_new();
	if (ctx == NULL)
	{
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create cipher context")));
	}

	if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, encrypted->encryption_iv) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: decryption initialization failed")));
	}

	if (EVP_DecryptUpdate(ctx, plaintext, &len, encrypted->ciphertext, encrypted->dim * sizeof(float4)) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: decryption failed")));
	}

	if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, encrypted->auth_tag) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to set authentication tag")));
	}

	if (EVP_DecryptFinal_ex(ctx, plaintext + len, &final_len) != 1)
	{
		EVP_CIPHER_CTX_free(ctx);
		pfree(key_str);
		pfree(key);
		pfree(plaintext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: decryption verification failed - invalid key or corrupted data")));
	}

	EVP_CIPHER_CTX_free(ctx);

	/* Allocate result vector */
	result_size = VECTOR_SIZE(encrypted->dim);
	result = (Vector *) palloc0(result_size);
	SET_VARSIZE(result, result_size);
	result->dim = encrypted->dim;
	memcpy(result->data, plaintext, encrypted->dim * sizeof(float4));

	/* Cleanup sensitive data */
	memset(key, 0, 32);
	memset(plaintext, 0, encrypted->dim * sizeof(float4));
	pfree(key_str);
	pfree(key);
	pfree(plaintext);

	PG_RETURN_POINTER(result);
}
