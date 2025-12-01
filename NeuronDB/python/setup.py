"""
Setup script for NeuronDB Python SDK.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setup(
	name="neurondb",
	version="1.0.0",
	author="neurondb, Inc.",
	author_email="admin@neurondb.com",
	description="Python SDK for NeuronDB PostgreSQL extension",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/neurondb/NeurondB",
	packages=find_packages(),
	classifiers=[
		"Development Status :: 4 - Beta",
		"Intended Audience :: Developers",
		"Topic :: Database",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"License :: OSI Approved :: PostgreSQL License",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Programming Language :: Python :: 3.12",
	],
	python_requires=">=3.8",
	install_requires=[
		"psycopg2-binary>=2.9.0",
		"numpy>=1.20.0",
	],
	extras_require={
		"async": ["psycopg2-pool>=1.1"],
	},
)

