-- Upgrade NeuronDB from 3.0.0-devel to 3.1.0 (LLM Model Storage and PL/Python functions)
\echo 'Upgrading NeuronDB from 3.0.0-devel to 3.1.0 (LLM Model Storage)'
\ir neurondb_llm_models.sql
\ir neurondb_llm_functions.sql
