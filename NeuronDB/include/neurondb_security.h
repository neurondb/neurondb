/*-------------------------------------------------------------------------
 *
 * neurondb_security.h
 *		Security and privacy definitions
 *
 * Defines structures for vector encryption, differential privacy,
 * RLS integration, and result signing for audit trails.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_security.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_SECURITY_H
#define NEURONDB_SECURITY_H

#include "postgres.h"

/*
 * Encrypted vector storage
 * AES-GCM encryption at rest with SIMD-safe decrypt
 */
typedef struct EncryptedVector
{
	int32 vl_len_;
	uint32 tenant_id;
	uint8 encryption_iv[12]; /* GCM initialization vector */
	uint8 auth_tag[16]; /* GCM authentication tag */
	uint16 dim; /* Original dimension */
	uint16 unused;
	/* Followed by encrypted data */
	uint8 ciphertext[FLEXIBLE_ARRAY_MEMBER];
} EncryptedVector;

/*
 * Differential privacy parameters
 * Per-tenant epsilon tracking
 */
typedef struct DPEmbedding
{
	int32 tenant_id;
	float8 epsilon; /* Privacy budget */
	float8 delta; /* Privacy parameter */
	float8 sensitivity; /* L2 sensitivity */
	int32 queries_used; /* Query budget consumed */
	int32 queries_total; /* Total budget */
} DPEmbedding;

/*
 * RLS vector constraints
 * Row-level security for vectors
 */
typedef struct VectorRLS
{
	int32 tenant_id;
	int32 label_id; /* Classification label */
	uint32 access_mask; /* Bitfield of allowed operations */
	bool enforce_in_index; /* Check during index scan */
} VectorRLS;

/*
 * Signed query result
 * HMAC for audit trails
 */
typedef struct SignedResult
{
	uint64 query_id;
	TimestampTz timestamp;
	int32 result_count;
	uint8 hmac[32]; /* SHA-256 HMAC */
	/* Followed by result data */
} SignedResult;

/* Access mask bits */
#define VEC_RLS_READ 0x01
#define VEC_RLS_WRITE 0x02
#define VEC_RLS_DELETE 0x04
#define VEC_RLS_SIMILARITY 0x08

#endif /* NEURONDB_SECURITY_H */
