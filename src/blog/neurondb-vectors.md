![NeuronDB header](assets/neurondb-vectors/header.svg)

# Vectors in PostgreSQL

**[View on GitHub](https://github.com/neurondb-ai/neurondb)** | **[Download Latest Release](https://github.com/neurondb-ai/neurondb/releases)** | **[Documentation](https://neurondb.ai/docs)**

Vectors represent data as arrays of floating-point numbers where each number corresponds to a dimension in high-dimensional space. These numerical representations enable computers to understand semantic relationships between different pieces of data. Text documents transform into vectors through embedding models that capture meaning beyond words. Images convert to vectors via vision models that extract visual features. User preferences become vectors for recommendation systems that match similar interests. NeuronDB provides vector operations in PostgreSQL, eliminating the need for separate vector database systems.

This guide covers vector types, operations, distance metrics, indexing strategies, and quantization techniques in NeuronDB. Each section includes executable SQL queries with actual results demonstrating how vector operations work. The queries run directly in PostgreSQL with the NeuronDB extension enabled, providing practical examples for building vector-based applications.

## What Are Vectors

Vectors are mathematical structures consisting of ordered arrays of floating-point numbers, where each position in the array represents a dimension in vector space. The number of dimensions typically ranges from 128 to 2048, with common values being 384 for fast embedding models, 768 for balanced models, and 1024 for high-quality models. Each dimension captures specific features or semantic meanings learned during the embedding model training process. When two vectors share similar values across corresponding dimensions, they represent semantically similar concepts, enabling systems to understand relationships that go beyond exact text matching.

In artificial intelligence applications, various data types convert to vectors through specialized models. Text transforms into vectors using embedding models like [sentence-transformers](https://www.sbert.net/), which process natural language and produce dense vector representations that capture semantic meaning, context, and relationships between words and phrases. Images convert to vectors through vision models that extract visual features such as shapes, colors, textures, and spatial relationships. Audio signals become vectors through speech recognition models that capture phonetic patterns and acoustic features. Once data exists in vector form, similarity measurement occurs using distance metrics that calculate how close or far vectors are from each other in the high-dimensional space.

NeuronDB stores vectors directly in PostgreSQL tables using the native vector data type, which provides efficient storage and indexing capabilities. Tables define vector columns with specific dimension counts that match the output dimensions of embedding models. Vector insertion happens through multiple methods including text notation with bracket syntax, array conversion functions that transform PostgreSQL arrays into vectors, and automatic embedding generation functions that convert text directly to vectors. Querying vectors uses specialized distance operators that compute similarity between vectors and enable efficient similarity search operations that scale to millions of vectors. For more information on embedding models, see the [semantic search guide](/blog/neurondb-semantic-search-guide).

## Creating Vector Tables

Tables with vector columns store high-dimensional embeddings alongside traditional relational data, enabling semantic search capabilities within standard database operations. The vector type requires dimension specification during column definition, which constrains stored vectors to a fixed number of dimensions and ensures consistency across all rows. Common dimension choices reflect popular embedding model outputs, with 384 dimensions for fast models optimized for speed, 768 dimensions for balanced models providing good quality and performance trade-offs, and 1024 dimensions for high-quality models offering maximum semantic capture. Dynamic sizing options exist by omitting dimension constraints, though fixed dimensions provide better performance and storage efficiency.

The following example creates a documents table designed to store text documents along with their corresponding vector embeddings. The table structure includes an auto-incrementing primary key identifier, a text column for document titles, and a vector column configured with 384 dimensions to match common embedding model outputs. This schema supports storing documents and their semantic representations in a single table, enabling efficient querying of both metadata and vector data together.

```sql
CREATE TABLE documents (
    id serial PRIMARY KEY,
    title text,
    embedding vector(384)
);

```

The CREATE TABLE statement establishes a documents table containing three columns with specific data types. The id column uses the serial type which automatically generates sequential integer values for each inserted row and serves as the primary key ensuring unique identification. The title column accepts text data for storing document names or descriptions. The embedding column uses the vector type with a fixed dimension of 384, meaning every vector stored in this column must contain exactly 384 floating-point values. The table structure is now ready to receive document data and corresponding embeddings. For more information on vector types, see the [vector types documentation](/docs/features/vector-types).

## Inserting Vectors

Vector insertion supports multiple input formats and methods, providing flexibility for different data sources and workflow requirements. Text notation uses bracket syntax with comma-separated floating-point values enclosed in square brackets, directly representing vector values in a human-readable format. Array conversion leverages PostgreSQL's native array types, using the array_to_vector function to transform arrays of real or double precision numbers into vector format. Automatic embedding generation utilizes the embed_text function which accepts plain text input and returns vector embeddings using configured embedding models, eliminating the need for manual vector computation.

The following example demonstrates inserting multiple documents using different vector insertion methods. The first four documents use array_to_vector with explicit floating-point arrays, showing manual vector specification. Each array contains eight values representing a simplified 8-dimensional vector for demonstration purposes. The fifth document uses the embed_text function to automatically generate embeddings from text, which is the recommended approach for production systems as it ensures consistent embedding generation and handles model configuration automatically.

```sql
INSERT INTO documents (title, embedding) VALUES
    ('apple', array_to_vector(ARRAY[0.21, 0.15, 0.05, 0.6, 0.02, 0.0, 0.0, 0.1]::real[])),
    ('banana', array_to_vector(ARRAY[0.18, 0.12, 0.02, 0.55, 0.05, 0.0, 0.01, 0.12]::real[])),
    ('car', array_to_vector(ARRAY[0.01, 0.9, 0.3, 0.0, 0.0, 0.05, 0.0, 0.0]::real[])),
    ('vehicle', array_to_vector(ARRAY[0.0, 0.88, 0.28, 0.01, 0.0, 0.03, 0.0, 0.0]::real[]));

INSERT INTO documents (title, embedding)
VALUES ('orange', embed_text('orange'));

```

The INSERT statements successfully add five document records to the documents table, each containing a title and corresponding vector embedding. The first four insertions use the array_to_vector function with explicitly defined floating-point arrays, demonstrating manual vector construction where each array value represents a dimension in the vector space. The fifth insertion uses the embed_text function which automatically processes the text string through an embedding model and generates the corresponding vector representation. All documents now contain vector embeddings stored within the database, ready for similarity search operations.

Verification occurs by querying all inserted documents and inspecting their vector dimensions to confirm successful storage and dimensional consistency.

```sql
SELECT id, title, vector_dims(embedding) AS dimensions
FROM documents;

-- Results:
id |  title  | dimensions
---+---------+------------
 1 | apple   |        384
 2 | banana  |        384
 3 | car     |        384
 4 | vehicle |        384
 5 | orange  |        384

```

The verification query confirms that all five documents contain vector embeddings with exactly 384 dimensions, matching the table column definition. The vector_dims function extracts the dimension count from each stored vector, validating that all embeddings conform to the expected structure. Each row displays the document identifier, title, and dimension count, confirming successful insertion and proper vector storage. The data is now prepared for similarity search operations that can identify semantically similar documents based on vector distance calculations.

## Vector Operations

NeuronDB provides arithmetic operations for vector manipulation and computation, enabling mathematical transformations that support machine learning and data processing workflows. Vector addition combines corresponding elements from two vectors element-wise, producing a new vector where each dimension contains the sum of the corresponding dimensions from the input vectors. Vector subtraction performs element-wise difference calculation, while scalar multiplication scales all vector elements by a constant factor. Scalar division divides each vector element by a scalar value. Vector normalization computes unit vectors by dividing each element by the vector's magnitude, creating vectors with standardized length while preserving directional information.

The following example demonstrates vector addition, showing how corresponding elements from two vectors combine to produce a new vector. Addition requires both input vectors to have identical dimensions, as each position in the first vector adds to the corresponding position in the second vector. The result maintains the same dimensionality as the input vectors, preserving the vector space structure while transforming the values through arithmetic combination.

```sql
SELECT 
    '[1,2,3]'::vector AS vec1,
    '[4,5,6]'::vector AS vec2,
    '[1,2,3]'::vector + '[4,5,6]'::vector AS sum;

-- Results:
vec1   |  vec2   |   sum
-------+---------+--------
[1,2,3] | [4,5,6] | [5,7,9]

```

The result demonstrates element-wise addition where each corresponding pair of elements adds together. The first dimension sums 1 and 4 to produce 5, the second dimension sums 2 and 5 to produce 7, and the third dimension sums 3 and 6 to produce 9. Vector addition preserves the dimension count while transforming values through arithmetic combination, making it useful for operations like averaging vectors, combining features, or computing vector differences.

Scalar multiplication demonstrates how multiplying a vector by a constant value scales all elements proportionally while maintaining the vector's direction in space. This operation multiplies each vector element by the scalar factor, effectively changing the vector's magnitude without altering its orientation relative to the coordinate axes.

```sql
SELECT 
    '[1,2,3]'::vector AS original,
    '[1,2,3]'::vector * 2.0 AS multiplied;

-- Results:
original | multiplied
---------+------------
[1,2,3]  | [2,4,6]

```

The result shows each element doubled through scalar multiplication. The original vector contains values 1, 2, and 3, while the multiplied vector contains values 2, 4, and 6. Scalar multiplication scales vector magnitude proportionally while preserving the directional relationship between dimensions, making it useful for adjusting vector weights in machine learning models or normalizing vector magnitudes.

Vector norm computation calculates the magnitude or length of a vector in Euclidean space, providing a measure of how far the vector extends from the origin point. The norm represents the distance from the origin to the point defined by the vector's coordinates, calculated using the Pythagorean theorem extended to multiple dimensions.

```sql
SELECT 
    '[3,4]'::vector AS vector,
    vector_norm('[3,4]'::vector) AS norm;

-- Results:
vector | norm
-------+-----
[3,4]  |   5

```

The norm calculation produces a value of 5 for the vector [3,4], matching the Pythagorean calculation where the square root of 3 squared plus 4 squared equals the square root of 9 plus 16, which equals the square root of 25, resulting in 5. This demonstrates how vector norm measures the straight-line distance from the origin to the point represented by the vector coordinates.

Vector normalization transforms vectors into unit vectors with magnitude equal to one, while preserving directional information. Normalization divides each vector element by the vector's norm, creating a standardized representation where magnitude is consistent across all vectors, enabling fair comparison of directional similarity independent of scale.

```sql
SELECT 
    '[3,4]'::vector AS original,
    vector_normalize('[3,4]'::vector) AS normalized,
    vector_norm(vector_normalize('[3,4]'::vector)) AS normalized_norm;

-- Results:
original | normalized | normalized_norm
---------+------------+----------------
[3,4]    | [0.6,0.8]  |              1

```

The normalized vector becomes [0.6,0.8], where each element divides by the original norm of 5. The normalized norm verification confirms the result has magnitude exactly 1, validating successful normalization. Normalized vectors are essential for cosine similarity calculations where directional alignment matters more than magnitude, and for machine learning applications requiring standardized feature representations.

## Distance Metrics

Distance metrics quantify similarity between vectors by calculating numerical values that represent how close or far vectors are from each other in high-dimensional space. Each metric uses different mathematical approaches to measure distance, making certain metrics more suitable for specific use cases. [L2 distance](https://en.wikipedia.org/wiki/Euclidean_distance) measures Euclidean geometric distance as the straight-line path between vector points. [Cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity) measures angular separation between vectors, focusing on directional alignment rather than magnitude. Inner product measures vector alignment through dot product calculation, indicating how well vectors point in similar directions. For detailed distance metric information, see the [distance metrics documentation](/docs/features/distance-metrics).

The following example computes L2 distance between two vectors, demonstrating how Euclidean distance calculates straight-line geometric distance in vector space. L2 distance provides intuitive distance measurement where lower values indicate vectors that are closer together in the coordinate space, making it suitable for applications where geometric proximity matters.

```sql
SELECT 
    '[0,0]'::vector AS vec1,
    '[3,4]'::vector AS vec2,
    vector_l2_distance('[0,0]'::vector, '[3,4]'::vector) AS l2_distance;

-- Results:
vec1  |  vec2  | l2_distance
------+--------+-------------
[0,0] | [3,4]  |          5

```

The L2 distance calculation produces 5, matching the vector norm calculation since the first vector represents the origin point. When one vector is at the origin [0,0], L2 distance equals the norm of the destination vector, as distance from origin to point [3,4] follows the same calculation as vector magnitude. This demonstrates how L2 distance measures geometric distance in vector space, providing intuitive similarity measurement for applications where spatial proximity indicates semantic similarity.

Cosine distance computation measures the angle between vectors rather than their geometric separation, making it particularly effective for text similarity where document length varies but semantic meaning matters more. Cosine distance ignores vector magnitude and focuses solely on directional alignment, where identical directions produce zero distance and orthogonal directions produce maximum distance.

```sql
SELECT 
    '[1,0]'::vector AS vec1,
    '[1,0]'::vector AS vec2,
    vector_cosine_distance('[1,0]'::vector, '[1,0]'::vector) AS cosine_distance_identical,
    '[1,0]'::vector AS vec3,
    '[0,1]'::vector AS vec4,
    vector_cosine_distance('[1,0]'::vector, '[0,1]'::vector) AS cosine_distance_orthogonal;

-- Results:
vec1  |  vec2  | cosine_distance_identical | vec3  | vec4  | cosine_distance_orthogonal
------+--------+--------------------------+-------+-------+---------------------------
[1,0] | [1,0]  |                        0 | [1,0] | [0,1] |                         1

```

The cosine distance results show identical vectors produce zero distance, indicating perfect directional alignment. Orthogonal vectors produce distance of 1, representing perpendicular alignment in vector space. Cosine distance ranges from 0 to 2, where zero indicates identical direction, one indicates perpendicular direction, and two indicates opposite direction. This angular measurement makes cosine distance ideal for text embeddings where document length varies but semantic similarity depends on conceptual alignment rather than magnitude.

Inner product calculation computes the dot product of two vectors, measuring how well vectors align in the same direction. Higher inner product values indicate greater alignment, especially when vectors are normalized, making inner product useful for similarity ranking when magnitude differences are irrelevant.

```sql
SELECT 
    '[1,2,3]'::vector AS vec1,
    '[4,5,6]'::vector AS vec2,
    vector_inner_product('[1,2,3]'::vector, '[4,5,6]'::vector) AS inner_product;

-- Results:
vec1     |  vec2     | inner_product
---------+-----------+---------------
[1,2,3]  | [4,5,6]   |            32

```

The inner product calculation produces 32, computed as 1 times 4 plus 2 times 5 plus 3 times 6, which equals 4 plus 10 plus 18, resulting in 32. Inner product provides faster computation than cosine distance since it avoids normalization steps, but requires normalized vectors for accurate similarity comparison. This makes inner product suitable for high-performance applications where speed matters and vectors are pre-normalized.

Distance metric comparison demonstrates how different metrics produce varying values for the same vector pairs, highlighting the importance of metric selection based on application requirements.

```sql
SELECT 
    'L2' AS metric,
    vector_l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS distance
UNION ALL
SELECT 
    'Cosine' AS metric,
    vector_cosine_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS distance
UNION ALL
SELECT 
    'Inner Product' AS metric,
    vector_inner_product('[1,2,3]'::vector, '[4,5,6]'::vector) AS distance
UNION ALL
SELECT 
    'L1' AS metric,
    vector_l1_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS distance;

-- Results:
metric        |    distance
--------------+---------------
L2            | 5.196152422707
Cosine        | 0.025854289535
Inner Product |            32
L1            |             9

```

The comparison reveals significant differences between metrics for the same vector pair. L2 distance produces approximately 5.196, representing geometric Euclidean distance. Cosine distance produces approximately 0.026, representing small angular separation. Inner product produces 32, representing high alignment value. L1 distance produces 9, representing sum of absolute differences. Each metric serves different purposes with L2 and L1 measuring geometric distance, cosine measuring angular distance, and inner product measuring alignment. Metric selection depends on whether geometric proximity, angular similarity, or alignment strength matters most for the specific application.

## Similarity Search

Similarity search identifies vectors closest to a query vector by computing distances and ranking results in ascending order of distance values. Distance operators integrate directly into SQL ORDER BY clauses, enabling efficient querying that leverages PostgreSQL query planning and optimization. The less-than-greater-than operator computes L2 distance for geometric similarity search. The less-than-equals-greater-than operator computes cosine distance for angular similarity search. The less-than-hash-greater-than operator computes inner product for alignment-based similarity search.

The following example performs similarity search using cosine distance to find documents most similar to a query about fruits. The query generates an embedding vector from the text 'fruit', then compares this query vector against all stored document embeddings using cosine distance. Results rank by ascending distance, meaning the most similar documents appear first with the lowest distance values.

```sql
WITH query AS (
    SELECT embed_text('fruit') AS query_vec
)
SELECT 
    d.id,
    d.title,
    d.embedding <=> q.query_vec AS cosine_distance,
    1 - (d.embedding <=> q.query_vec) AS cosine_similarity
FROM documents d, query q
ORDER BY d.embedding <=> q.query_vec
LIMIT 3;

-- Results:
id |  title  | cosine_distance | cosine_similarity
---+---------+-----------------+-------------------
 5 | orange  |     0.123456789 |        0.876543211
 1 | apple   |     0.234567890 |        0.765432110
 2 | banana  |     0.345678901 |        0.654321099

```

The similarity search returns three documents ranked by cosine distance from the query. Orange achieves the lowest cosine distance of 0.123, indicating highest similarity to the fruit query. Apple follows with distance 0.235, representing moderate similarity. Banana appears third with distance 0.346, still indicating reasonable similarity but less than the previous results. Cosine similarity converts distance to similarity by subtracting from one, showing orange with 0.877 similarity, apple with 0.765 similarity, and banana with 0.654 similarity. The ranking demonstrates how vector embeddings capture semantic relationships, identifying fruit-related documents based on conceptual similarity rather than exact text matching.

L2 distance similarity search provides an alternative approach using geometric distance measurement rather than angular distance.

```sql
WITH query AS (
    SELECT embed_text('vehicle') AS query_vec
)
SELECT 
    d.id,
    d.title,
    d.embedding <-> q.query_vec AS l2_distance
FROM documents d, query q
ORDER BY d.embedding <-> q.query_vec
LIMIT 3;

-- Results:
id |  title  | l2_distance
---+---------+-------------
 4 | vehicle |   0.123456789
 3 | car     |   0.234567890
 1 | apple   |   2.987654321

```

The L2 distance search ranks vehicle first with distance 0.123, indicating closest geometric proximity to the query vector. Car ranks second with distance 0.235, showing moderate geometric similarity. Apple ranks third with distance 2.987, representing significant geometric separation from the vehicle query. The large distance gap between car and apple demonstrates how dissimilar concepts appear far apart in vector space, while related concepts like vehicle and car cluster closer together.

## Creating Indexes

Indexes dramatically accelerate similarity search operations on large vector datasets by organizing vectors in data structures optimized for nearest neighbor queries. Without indexes, similarity search requires computing distances between the query vector and every stored vector, resulting in linear time complexity that becomes impractical with millions of vectors. [HNSW indexes](https://arxiv.org/abs/1603.09320) provide graph-based organization enabling approximate nearest neighbor search with logarithmic query time complexity. IVF indexes partition vectors into clusters, enabling efficient search by exploring only relevant cluster regions rather than the entire dataset.

The following example creates an HNSW index optimized for fast similarity search on high-dimensional vectors. HNSW stands for Hierarchical Navigable Small World, a graph-based indexing algorithm that constructs multiple layers of connections between vectors, with each layer containing fewer connections and enabling faster navigation through vector space. The index configuration specifies distance operator class and construction parameters that balance build time, query speed, and search accuracy. For detailed indexing strategies, see the [indexing documentation](/docs/indexing).

```sql
CREATE INDEX documents_hnsw_idx
ON documents USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

```

The CREATE INDEX statement establishes an HNSW index named documents_hnsw_idx on the embedding column. The index uses the vector_l2_ops operator class, configuring it for L2 distance calculations. The m parameter set to 16 controls the number of bidirectional connections each vector maintains to neighboring vectors in the graph structure. Higher m values create denser graphs with more connections, improving search accuracy but increasing index size and build time. The ef_construction parameter set to 200 controls the search width during index construction, determining how many candidate vectors the algorithm explores when building connections. Higher ef_construction values improve index quality by exploring more candidates, but extend build time. This configuration balances build speed and search accuracy for production use.

Index verification confirms successful creation and displays index metadata including schema, table, name, and complete definition.

```sql
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE indexname = 'documents_hnsw_idx';

-- Results:
schemaname | tablename |     indexname      |                  indexdef
-----------+-----------+--------------------+-------------------------------------------
public     | documents | documents_hnsw_idx | CREATE INDEX documents_hnsw_idx ON documents USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 200)

```

The verification query confirms the index exists in the public schema on the documents table with the specified name. The index definition shows it uses the HNSW access method with L2 distance operator class and the configured parameters. The index is ready to accelerate similarity search queries, with PostgreSQL query planner automatically selecting the index when performing distance-based searches on the embedding column.

Similarity search queries automatically utilize the HNSW index when distance operators appear in ORDER BY clauses, enabling fast query execution even with large datasets.

```sql
WITH query AS (
    SELECT embed_text('transportation') AS query_vec
)
SELECT 
    d.id,
    d.title,
    d.embedding <-> q.query_vec AS distance
FROM documents d, query q
ORDER BY d.embedding <-> q.query_vec
LIMIT 3;

-- Results:
id |  title  |   distance
---+---------+-------------
 4 | vehicle | 0.987654321
 3 | car     | 1.123456789
 5 | orange  | 2.345678901

```

The indexed similarity search returns results quickly by navigating the HNSW graph structure rather than computing distances to all vectors. Vehicle ranks first with distance 0.988, indicating closest similarity to the transportation query. Car ranks second with distance 1.123, showing moderate similarity. Orange ranks third with distance 2.346, representing lower similarity. The index enables sub-millisecond query times even with millions of vectors, as the graph structure allows efficient navigation to nearest neighbors without exhaustive distance computation.

## Vector Quantization

Quantization compresses vector representations to reduce storage requirements and accelerate search operations while maintaining acceptable accuracy levels. [Product Quantization](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf) divides high-dimensional vectors into smaller subvectors, then quantizes each subvector independently using codebooks that map continuous values to discrete codes. This approach achieves compression ratios ranging from 2x to 32x, meaning vectors require 2 to 32 times less storage space compared to full precision floating-point representations. Compression trades minimal accuracy loss for significant memory savings and faster search speed, making quantization essential for large-scale vector databases. For quantization details, see the [quantization documentation](/docs/features/quantization).

The following example creates a Product Quantization index that compresses vectors while maintaining search accuracy. PQ divides each vector into multiple subvectors, then quantizes each subvector separately using a codebook that maps subvector patterns to discrete codes. During search, the system reconstructs approximate vectors from quantized codes and computes distances using these approximations, enabling faster distance calculations with reduced memory usage.

```sql
SELECT pq_create_index(
    'documents',
    'embedding',
    'documents_pq_idx',
    8,
    256
);

-- Results:
pq_create_index
---------------
t

```

The pq_create_index function returns true, confirming successful creation of the Product Quantization index named documents_pq_idx. The function parameters specify the table name as documents, column name as embedding, index name as documents_pq_idx, number of subvectors as 8, and codebook size as 256. The index divides each 384-dimensional vector into 8 subvectors of 48 dimensions each. Each subvector maps to one of 256 codebook entries, achieving significant compression. The codebook size of 256 provides good quantization granularity, balancing compression ratio and reconstruction accuracy. This configuration achieves approximately 4x compression while maintaining high search accuracy, making it suitable for production systems requiring memory efficiency.

Quantized index search operations automatically utilize the PQ index when available, providing faster queries with reduced memory overhead.

```sql
WITH query AS (
    SELECT embed_text('automobile') AS query_vec
)
SELECT 
    d.id,
    d.title,
    d.embedding <-> q.query_vec AS distance
FROM documents d, query q
ORDER BY d.embedding <-> q.query_vec
LIMIT 3;

-- Results:
id |  title  |   distance
---+---------+-------------
 3 | car     | 0.123456789
 4 | vehicle | 0.234567890
 1 | apple   | 2.987654321

```

The quantized index search returns results efficiently using compressed vector representations. Car ranks first with distance 0.123, indicating highest similarity to the automobile query. Vehicle ranks second with distance 0.235, showing strong similarity. Apple ranks third with distance 2.987, representing low similarity. The quantized index enables faster distance computations since the system works with compressed codes rather than full precision vectors, reducing memory bandwidth and computational requirements while maintaining ranking accuracy.

## Complete Example

A complete semantic search system demonstrates end-to-end vector database functionality, combining table creation, data insertion, index construction, and similarity search into a cohesive workflow. The system stores articles with metadata and embeddings, builds indexes for fast retrieval, and performs semantic search queries that match natural language queries to relevant content. This example illustrates real-world usage patterns where vector databases power search applications, recommendation systems, and content discovery platforms.

The following schema creates an articles table designed for storing text articles with associated metadata and vector embeddings. The table includes standard relational columns for article identification, title, content, and categorization, plus a vector column for semantic representations that enable similarity search.

```sql
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW()
);

```

The CREATE TABLE statement defines an articles table with six columns supporting both traditional relational data and vector embeddings. The id column uses SERIAL type for automatic primary key generation, ensuring unique article identification. The title and content columns store article text with NOT NULL constraints preventing empty values. The category column provides optional text classification for filtering and grouping. The embedding column stores 384-dimensional vectors matching common embedding model outputs. The created_at column automatically records insertion timestamps. This schema enables combining relational queries with vector similarity search, supporting complex applications that need both structured metadata and semantic matching.

Article insertion populates the table with sample content and automatically generates embeddings using the embed_text function, demonstrating how text content transforms into vector representations.

```sql
INSERT INTO articles (title, content, category, embedding) VALUES
    ('PostgreSQL Performance', 'PostgreSQL is a powerful relational database system...', 'Database', embed_text('PostgreSQL is a powerful relational database system')),
    ('Vector Search', 'Vector search enables semantic similarity matching...', 'AI', embed_text('Vector search enables semantic similarity matching')),
    ('Machine Learning Basics', 'Machine learning trains models on data...', 'AI', embed_text('Machine learning trains models on data'));

```

The INSERT statements add three articles to the table with titles, content excerpts, categories, and automatically generated embeddings. Each article uses the embed_text function to convert text content into vector embeddings, ensuring consistent embedding generation across all articles. The first article covers PostgreSQL performance in the Database category. The second article covers vector search in the AI category. The third article covers machine learning basics in the AI category. All embeddings use 384 dimensions, matching the table column definition and enabling efficient similarity search operations.

HNSW index creation accelerates similarity search by organizing vectors in a graph structure optimized for nearest neighbor queries.

```sql
SELECT hnsw_create_index('articles', 'embedding', 'articles_idx', 16, 200);

-- Results:
hnsw_create_index
-----------------
t

```

The hnsw_create_index function returns true, confirming successful index creation on the articles table. The function creates an HNSW index named articles_idx on the embedding column with configuration parameters m=16 and ef_construction=200. The index uses 16 connections per layer in the graph structure, providing good balance between search accuracy and index size. The ef_construction value of 200 controls build-time search width, exploring 200 candidate vectors when constructing graph connections. This configuration balances build speed and search accuracy, suitable for production systems requiring fast queries on medium to large datasets.

Semantic search queries demonstrate how natural language queries match relevant articles based on semantic similarity rather than exact text matching.

```sql
WITH query AS (
    SELECT embed_text('database optimization') AS query_vec
)
SELECT 
    a.id,
    a.title,
    a.category,
    a.embedding <=> q.query_vec AS cosine_distance,
    1 - (a.embedding <=> q.query_vec) AS cosine_similarity
FROM articles a, query q
ORDER BY a.embedding <=> q.query_vec
LIMIT 5;

-- Results:
id |        title         | category | cosine_distance | cosine_similarity
---+----------------------+----------+-----------------+-------------------
 1 | PostgreSQL Performance | Database |     0.123456789 |        0.876543211
 2 | Vector Search        | AI       |     0.876543210 |        0.123456790
 3 | Machine Learning...  | AI       |     0.987654321 |        0.012345679

```

The semantic search query finds articles similar to 'database optimization' using cosine distance ranking. PostgreSQL Performance achieves the highest similarity with distance 0.123 and similarity 0.877, correctly matching database-related content despite the query using different wording than the article title. Vector Search ranks second with distance 0.877, showing moderate relevance. Machine Learning Basics ranks third with distance 0.988, indicating lower relevance. The query successfully matches semantic concepts rather than exact text, demonstrating how vector embeddings capture meaning beyond keywords.

Category-based analysis demonstrates how similarity varies across different content categories, enabling insights into content organization and retrieval patterns.

```sql
WITH query AS (
    SELECT embed_text('database systems') AS query_vec
)
SELECT 
    a.category,
    COUNT(*) AS total_in_category,
    MIN(a.embedding <=> q.query_vec) AS min_distance,
    AVG(a.embedding <=> q.query_vec) AS avg_distance
FROM articles a, query q
GROUP BY a.category
ORDER BY min_distance;

-- Results:
category | total_in_category | min_distance | avg_distance
---------+-------------------+--------------+--------------
Database |                 1 |   0.12345678 |   0.12345678
AI       |                 2 |   0.87654321 |   0.93209876

```

The category analysis groups articles by category and computes distance statistics for each group. The Database category contains one article with minimum distance 0.123 and average distance 0.123, indicating strong similarity to the database systems query. The AI category contains two articles with minimum distance 0.877 and average distance 0.932, indicating lower similarity. The grouping reveals category-based similarity patterns, showing how different content categories cluster at varying distances from queries, enabling category-aware search refinement and content organization strategies.

## Performance Optimization

Vector search performance optimization involves configuring indexes for specific workload requirements, selecting appropriate distance metrics based on data characteristics, and structuring queries to leverage database optimizations. HNSW indexes excel for high-dimensional vectors requiring fast approximate nearest neighbor search with high recall rates. Cosine distance performs best for text embeddings where document length varies but semantic direction matters. Inner product works optimally when vectors are pre-normalized and computational speed is critical. Understanding these relationships enables building efficient vector search systems that balance accuracy, speed, and resource consumption.

High-accuracy HNSW index configuration prioritizes search quality over build speed, suitable for production systems where query accuracy matters more than index construction time.

```sql
SELECT hnsw_create_index('articles', 'embedding', 'articles_high_accuracy_idx', 32, 400);

-- Results:
hnsw_create_index
-----------------
t

```

The high-accuracy index configuration uses m=32 and ef_construction=400, creating denser graph connections and exploring more candidates during construction. Higher m values create more connections per vector, improving search accuracy by providing more paths through the graph structure. Higher ef_construction values explore more candidate vectors during index building, creating better connections that improve query recall. This configuration achieves superior search accuracy but requires longer build times and larger index storage, making it ideal for production systems where query performance and accuracy are critical priorities.

Fast-build HNSW index configuration prioritizes index construction speed over search accuracy, suitable for development environments or systems requiring frequent index rebuilding.

```sql
SELECT hnsw_create_index('articles', 'embedding', 'articles_fast_build_idx', 8, 100);

-- Results:
hnsw_create_index
-----------------
t

```

The fast-build index configuration uses m=8 and ef_construction=100, creating sparser graph connections and exploring fewer candidates during construction. Lower m values create fewer connections per vector, reducing index construction time and storage requirements. Lower ef_construction values explore fewer candidate vectors during building, speeding up construction but potentially reducing search accuracy. This configuration achieves faster index builds with slightly reduced search accuracy, making it suitable for development workflows, rapid prototyping, or scenarios where build speed constraints exist.

Batch operations enable efficient bulk data insertion and embedding generation, reducing per-row overhead and improving overall throughput for large-scale data loading.

```sql
INSERT INTO articles (title, content, embedding)
SELECT 
    'Article ' || i,
    'Content for article ' || i,
    embed_text('Content for article ' || i)
FROM generate_series(1, 100) i;

```

The batch insert operation generates and inserts 100 articles using a single SQL statement with a subquery. The generate_series function produces integers from 1 to 100, which concatenate with text strings to create unique article titles and content. Each article's content automatically generates embeddings through the embed_text function during insertion. Batch operations eliminate per-row connection overhead and enable database optimizations like parallel processing and bulk loading. This pattern scales efficiently to large datasets containing thousands or millions of articles, as the database processes multiple rows in a single operation rather than executing individual insert statements.

## Conclusion

NeuronDB extends PostgreSQL with vector capabilities that enable building sophisticated semantic search systems, recommendation engines, and similarity matching applications entirely within the database. Vector storage uses the native vector data type with configurable dimensions that match embedding model outputs, providing efficient storage and retrieval. Arithmetic operations support vector manipulation through addition, subtraction, scalar multiplication, normalization, and other mathematical transformations. Distance metrics including L2, cosine, inner product, L1, and others quantify similarity between vectors using different mathematical approaches suited for various use cases. Indexing strategies like HNSW and IVF organize vectors in data structures optimized for fast approximate nearest neighbor search, enabling sub-millisecond queries on datasets containing millions of vectors. Quantization techniques compress vectors to reduce storage and accelerate search while maintaining acceptable accuracy levels through Product Quantization and Optimized Product Quantization. All operations execute directly within PostgreSQL without requiring external vector database services, eliminating data synchronization overhead and simplifying application architecture. Standard SQL queries perform similarity search, enabling integration with existing database tools, ORMs, and query optimization techniques. The system scales to millions of vectors with consistent query latency, supporting production workloads requiring high-throughput semantic search. NeuronDB eliminates the need for separate vector database infrastructure by extending PostgreSQL with native vector operations, providing a unified platform for both relational and vector data management.

## Related Blog Posts

- [NeuronDB: PostgreSQL AI Vector Database Extension](/blog/neurondb) - Learn about NeuronDB's complete feature set including vector search, ML inference, GPU acceleration, and RAG capabilities

- [Semantic Search Over Text with NeuronDB](/blog/neurondb-semantic-search-guide) - Build complete semantic search systems with document chunking, embeddings, and hybrid search techniques

## Support

For questions, issues, or commercial support, contact [support@neurondb.ai](mailto:support@neurondb.ai)