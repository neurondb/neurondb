// NeurondbConfig moved to config.ts as DatabaseConfig
// This file now only contains tool-specific types

export interface VectorSearchParams {
	table: string;
	vector_column: string;
	query_vector: number[];
	limit?: number;
	distance_metric?: "l2" | "cosine" | "inner_product" | "l1" | "hamming" | "chebyshev" | "minkowski";
	minkowski_p?: number;
}

export interface EmbeddingParams {
	text: string;
	model?: string;
}

export interface BatchEmbeddingParams {
	texts: string[];
	model?: string;
}

export interface HNSWIndexParams {
	table: string;
	vector_column: string;
	index_name: string;
	m?: number;
	ef_construction?: number;
}

export interface HybridSearchParams {
	table: string;
	query_vector: number[];
	query_text: string;
	text_column: string;
	vector_column: string;
	vector_weight?: number;
	limit?: number;
}

export interface MLTrainingParams {
	table: string;
	feature_col: string;
	label_col: string;
	algorithm?: string;
	params?: Record<string, any>;
}

export interface MLPredictionParams {
	model_id: number;
	features: number[];
}

export interface ClusteringParams {
	table: string;
	vector_column: string;
	k: number;
	max_iter?: number;
	algorithm?: "kmeans" | "minibatch_kmeans" | "gmm" | "dbscan";
	params?: Record<string, any>;
}

