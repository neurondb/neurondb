-- ============================================================================
-- Vector Comparison Operators
-- ============================================================================
-- Add missing comparison operators for vector type: <, >, <=, >=
-- These operators use lexicographic comparison (element-by-element)

-- Create the functions first (they should exist in C code but ensure they're accessible)
CREATE OR REPLACE FUNCTION vector_lt(vector, vector)
RETURNS bool
LANGUAGE internal
IMMUTABLE STRICT
AS 'vector_lt';

CREATE OR REPLACE FUNCTION vector_le(vector, vector)
RETURNS bool
LANGUAGE internal
IMMUTABLE STRICT
AS 'vector_le';

CREATE OR REPLACE FUNCTION vector_gt(vector, vector)
RETURNS bool
LANGUAGE internal
IMMUTABLE STRICT
AS 'vector_gt';

CREATE OR REPLACE FUNCTION vector_ge(vector, vector)
RETURNS bool
LANGUAGE internal
IMMUTABLE STRICT
AS 'vector_ge';

CREATE OPERATOR < (
	LEFTARG = vector,
	RIGHTARG = vector,
	FUNCTION = vector_lt
);

CREATE OPERATOR <= (
	LEFTARG = vector,
	RIGHTARG = vector,
	FUNCTION = vector_le
);

CREATE OPERATOR > (
	LEFTARG = vector,
	RIGHTARG = vector,
	FUNCTION = vector_gt
);

CREATE OPERATOR >= (
	LEFTARG = vector,
	RIGHTARG = vector,
	FUNCTION = vector_ge
);

-- Comments removed to avoid parsing issues during operator creation

