-- Upgrade script from NeuronDB 2.0 to 2.1.0
-- This file is used by PostgreSQL ALTER EXTENSION ... UPDATE TO

-- Advanced geometric transformations
CREATE FUNCTION vector_project(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_project'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_project IS 'Project vector onto another vector';

CREATE FUNCTION vector_reject(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_reject'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_reject IS 'Reject component (orthogonal projection)';

CREATE FUNCTION vector_reflect(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_reflect'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_reflect IS 'Reflect vector across plane defined by normal vector';

CREATE FUNCTION vector_rotate(vector, double precision) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_rotate'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_rotate IS 'Rotate 2D vector by angle (in radians), or rotate first two dimensions for higher-D vectors';

-- Advanced mathematical operations
CREATE FUNCTION vector_exp(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_exp'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_exp IS 'Element-wise exponential';

CREATE FUNCTION vector_log(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_log'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_log IS 'Element-wise natural logarithm';

CREATE FUNCTION vector_log10(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_log10'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_log10 IS 'Element-wise base-10 logarithm';

CREATE FUNCTION vector_sin(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_sin'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_sin IS 'Element-wise sine';

CREATE FUNCTION vector_cos(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_cos'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_cos IS 'Element-wise cosine';

CREATE FUNCTION vector_tan(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_tan'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_tan IS 'Element-wise tangent';

CREATE FUNCTION vector_asin(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_asin'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_asin IS 'Element-wise arcsine';

CREATE FUNCTION vector_acos(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_acos'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_acos IS 'Element-wise arccosine';

CREATE FUNCTION vector_atan(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_atan'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_atan IS 'Element-wise arctangent';

CREATE FUNCTION vector_sinh(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_sinh'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_sinh IS 'Element-wise hyperbolic sine';

CREATE FUNCTION vector_cosh(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_cosh'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_cosh IS 'Element-wise hyperbolic cosine';

CREATE FUNCTION vector_tanh(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_tanh'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_tanh IS 'Element-wise hyperbolic tangent';

CREATE FUNCTION vector_erf(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_erf'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_erf IS 'Element-wise error function';

CREATE FUNCTION vector_erfc(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'vector_erfc'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_erfc IS 'Element-wise complementary error function';

-- Advanced statistical functions
CREATE FUNCTION vector_skewness(vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_skewness'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_skewness IS 'Skewness of vector elements';

CREATE FUNCTION vector_kurtosis(vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_kurtosis'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_kurtosis IS 'Kurtosis (excess) of vector elements';

CREATE FUNCTION vector_entropy(vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_entropy'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION vector_entropy IS 'Shannon entropy of vector elements (as probability distribution)';

-- Correlation-based distance metrics
CREATE FUNCTION vector_pearson_correlation(vector, vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_pearson_correlation'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_pearson_correlation IS 'Pearson correlation coefficient between two vectors';

CREATE FUNCTION vector_weighted_distance(vector, vector, vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_weighted_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_weighted_distance IS 'Weighted L2 distance with per-dimension weights';

-- Information-theoretic distance metrics
CREATE FUNCTION vector_kl_divergence(vector, vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_kl_divergence'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_kl_divergence IS 'Kullback-Leibler divergence (requires probability distributions)';

CREATE FUNCTION vector_js_divergence(vector, vector) RETURNS double precision
    AS 'MODULE_PATHNAME', 'vector_js_divergence'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
COMMENT ON FUNCTION vector_js_divergence IS 'Jensen-Shannon divergence between two probability distributions';

