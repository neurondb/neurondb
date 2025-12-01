#!/usr/bin/env python3
"""
doc_processor.py
    Document processing utilities for extracting and cleaning text
    from various file formats

Copyright (c) 2024-2025, neurondb, Inc.
"""

import re
from pathlib import Path
from html.parser import HTMLParser
from typing import Optional, Tuple, Dict


class HTMLTextExtractor(HTMLParser):
	"""Extract text content from HTML while skipping script/style tags"""
	
	def __init__(self):
		super().__init__()
		self.text = []
		self.skip_tags = {'script', 'style', 'meta', 'link', 'noscript'}
		self.in_skip = False
	
	def handle_starttag(self, tag, attrs):
		if tag.lower() in self.skip_tags:
			self.in_skip = True
	
	def handle_endtag(self, tag):
		if tag.lower() in self.skip_tags:
			self.in_skip = False
	
	def handle_data(self, data):
		if not self.in_skip:
			self.text.append(data.strip())
	
	def get_text(self):
		return ' '.join(self.text)


class TextCleaner:
	"""Text cleaning utilities for different file formats"""
	
	@staticmethod
	def clean_html(content: str) -> str:
		"""Remove HTML tags and clean text"""
		extractor = HTMLTextExtractor()
		extractor.feed(content)
		text = extractor.get_text()
		# Clean up whitespace
		text = re.sub(r'\s+', ' ', text)
		return text.strip()
	
	@staticmethod
	def clean_markdown(content: str) -> str:
		"""Basic markdown cleaning - remove formatting, keep text"""
		# Remove code blocks
		content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)
		content = re.sub(r'`[^`]+`', '', content)
		# Remove links but keep text
		content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
		# Remove images
		content = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', content)
		# Remove headers but keep text
		content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
		# Remove bold/italic markers
		content = re.sub(r'\*\*([^\*]+)\*\*', r'\1', content)
		content = re.sub(r'\*([^\*]+)\*', r'\1', content)
		# Clean whitespace
		content = re.sub(r'\s+', ' ', content)
		return content.strip()
	
	@staticmethod
	def clean_text(content: str) -> str:
		"""Clean plain text - normalize whitespace"""
		content = re.sub(r'\s+', ' ', content)
		return content.strip()


class TitleExtractor:
	"""Extract titles from various document formats"""
	
	@staticmethod
	def extract_title(content: str, filepath: Path) -> str:
		"""Extract title from content or use filename"""
		# Try HTML title tag
		match = re.search(r'<title[^>]*>(.*?)</title>', content, 
			re.IGNORECASE | re.DOTALL)
		if match:
			title = re.sub(r'<[^>]+>', '', match.group(1)).strip()
			if title:
				return title
		
		# Try Markdown H1
		match = re.match(r'^#\s+(.+)$', content, re.MULTILINE)
		if match:
			return match.group(1).strip()
		
		# Try RST title
		match = re.match(r'^(.+)\n=+\s*$', content, re.MULTILINE)
		if match:
			return match.group(1).strip()
		
		# Use filename
		return filepath.stem.replace('_', ' ').replace('-', ' ').title()


class FileProcessor:
	"""Process individual files and extract content"""
	
	def __init__(self, min_length: int = 50):
		self.min_length = min_length
		self.cleaner = TextCleaner()
		self.title_extractor = TitleExtractor()
	
	def process_file(self, filepath: Path) -> Optional[Tuple[str, str, Dict]]:
		"""
		Process a single file and return (title, content, metadata)
		
		Returns:
			Tuple of (title, cleaned_content, metadata) or None if file is invalid
		"""
		try:
			with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
				content = f.read()
		except Exception:
			return None
		
		if len(content.strip()) < self.min_length:
			return None
		
		# Determine file type and clean content
		ext = filepath.suffix.lower()
		if ext == '.html' or ext == '.xml':
			cleaned = self.cleaner.clean_html(content)
			file_type = 'html'
		elif ext == '.md':
			cleaned = self.cleaner.clean_markdown(content)
			file_type = 'markdown'
		else:
			# Plain text
			cleaned = self.cleaner.clean_text(content)
			file_type = 'text'
		
		if len(cleaned) < self.min_length:
			return None
		
		title = self.title_extractor.extract_title(content, filepath)
		metadata = {
			"extension": ext,
			"file_type": file_type,
			"original_length": len(content),
			"cleaned_length": len(cleaned)
		}
		
		return title, cleaned, metadata
	
	def get_supported_extensions(self) -> tuple:
		"""Get list of supported file extensions"""
		return ('.html', '.md', '.txt', '.rst', '.xml')
	
	def find_documentation_files(self, directory: Path) -> list:
		"""Find all documentation files in directory"""
		files = []
		for ext in self.get_supported_extensions():
			files.extend(directory.rglob(f'*{ext}'))
		return files

