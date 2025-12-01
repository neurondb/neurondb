#!/usr/bin/env python3
"""
chunking.py
    Text chunking utilities for splitting documents into overlapping segments

Copyright (c) 2024-2025, neurondb, Inc.
"""

from typing import List, Tuple


class TextChunker:
	"""Chunk text into overlapping segments with sentence boundary awareness"""
	
	def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
		min_chunk_length: int = 50):
		"""
		Initialize chunker
		
		Args:
			chunk_size: Target size of each chunk in characters
			chunk_overlap: Number of characters to overlap between chunks
			min_chunk_length: Minimum length for a valid chunk
		"""
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
		self.min_chunk_length = min_chunk_length
	
	def chunk_text(self, text: str) -> List[Tuple[str, int]]:
		"""
		Chunk text into overlapping segments
		
		Returns:
			List of (chunk_text, chunk_index) tuples
		"""
		chunks = []
		text_len = len(text)
		start_pos = 0
		chunk_index = 0
		
		while start_pos < text_len:
			end_pos = min(start_pos + self.chunk_size, text_len)
			chunk = text[start_pos:end_pos]
			
			# Try to break at sentence boundary if not at end
			if end_pos < text_len:
				end_pos = self._find_sentence_boundary(text, start_pos, end_pos)
				chunk = text[start_pos:end_pos]
			
			# Skip very short chunks
			if len(chunk.strip()) >= self.min_chunk_length:
				chunks.append((chunk.strip(), chunk_index))
				chunk_index += 1
			
			# Move start position with overlap
			start_pos = end_pos - self.chunk_overlap
			if start_pos <= (end_pos - self.chunk_size + self.chunk_overlap):
				start_pos = end_pos
		
		return chunks
	
	def _find_sentence_boundary(self, text: str, start_pos: int, 
		end_pos: int) -> int:
		"""
		Find the best sentence boundary within the chunk
		
		Looks for sentence endings (., !, ?, \n) in the last 20% of the chunk
		"""
		chunk_length = end_pos - start_pos
		search_start = max(int(chunk_length * 0.8), chunk_length - 100)
		
		# Search backwards from end of chunk for sentence boundary
		for i in range(end_pos - 1, start_pos + search_start, -1):
			if text[i] in '.!?\n':
				return i + 1
		
		# No sentence boundary found, return original end
		return end_pos


class SQLChunker:
	"""SQL-based chunking using PostgreSQL functions"""
	
	def __init__(self, table_prefix: str):
		self.table_prefix = table_prefix
	
	def create_chunking_function(self) -> str:
		"""Generate SQL to create chunking function"""
		return f"""
		CREATE OR REPLACE FUNCTION chunk_text(
			text_content TEXT,
			chunk_size INTEGER,
			chunk_overlap INTEGER
		) RETURNS TABLE(chunk_text TEXT, chunk_index INTEGER) AS $$
		DECLARE
			text_len INTEGER;
			start_pos INTEGER := 1;
			current_index INTEGER := 0;
			current_chunk TEXT;
			chunk_end INTEGER;
		BEGIN
			text_len := LENGTH(text_content);
			
			WHILE start_pos <= text_len LOOP
				chunk_end := LEAST(start_pos + chunk_size - 1, text_len);
				current_chunk := SUBSTRING(text_content FROM start_pos 
					FOR (chunk_end - start_pos + 1));
				
				-- Try to break at sentence boundary if not at end
				IF chunk_end < text_len THEN
					DECLARE
						sentence_end INTEGER;
					BEGIN
						-- Look for sentence endings within last 20% of chunk
						sentence_end := GREATEST(
							LENGTH(current_chunk) * 8 / 10,
							LENGTH(current_chunk) - 100
						);
						
						-- Find last sentence boundary
						FOR sentence_end IN REVERSE sentence_end..1 LOOP
							IF SUBSTRING(current_chunk FROM sentence_end FOR 1) 
								IN ('.', '!', '?', '\n') THEN
								current_chunk := SUBSTRING(current_chunk 
									FROM 1 FOR sentence_end);
								chunk_end := start_pos + sentence_end - 1;
								EXIT;
							END IF;
						END LOOP;
					END;
				END IF;
				
				-- Skip very short chunks
				IF LENGTH(TRIM(current_chunk)) >= 50 THEN
					RETURN QUERY SELECT TRIM(current_chunk), current_index;
					current_index := current_index + 1;
				END IF;
				
				-- Move start position with overlap
				start_pos := chunk_end + 1 - chunk_overlap;
				IF start_pos <= (chunk_end - chunk_size + chunk_overlap) THEN
					start_pos := chunk_end + 1;
				END IF;
			END LOOP;
		END;
		$$ LANGUAGE plpgsql;
		"""
	
	def chunk_documents_sql(self, chunk_size: int, 
		chunk_overlap: int) -> str:
		"""Generate SQL to chunk all documents"""
		return f"""
		-- Delete existing chunks for re-chunking
		DELETE FROM {self.table_prefix}_chunks;
		
		-- Insert chunks
		INSERT INTO {self.table_prefix}_chunks 
			(doc_id, chunk_index, chunk_text, chunk_tokens)
		SELECT 
			d.doc_id,
			ct.chunk_index,
			ct.chunk_text,
			array_length(regexp_split_to_array(ct.chunk_text, E'\\s+'), 1) 
				AS chunk_tokens
		FROM {self.table_prefix}_documents d
		CROSS JOIN LATERAL chunk_text(d.content, {chunk_size}, {chunk_overlap}) ct
		WHERE d.content IS NOT NULL AND LENGTH(d.content) > 0;
		"""
	
	def get_chunk_statistics_sql(self) -> str:
		"""Generate SQL to get chunking statistics"""
		return f"""
		SELECT 
			COUNT(*) AS total_chunks,
			AVG(chunk_tokens)::INTEGER AS avg_tokens_per_chunk,
			MIN(chunk_tokens) AS min_tokens,
			MAX(chunk_tokens) AS max_tokens
		FROM {self.table_prefix}_chunks;
		"""

