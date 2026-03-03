# Token Counting and LLM Robustness

## Current behavior

- NeuronAgent estimates tokens as `len(prompt)/4` (approximate for English). This is used for metrics and logging; it is not accurate for all models or languages.

## Improvements (roadmap)

- **Tiktoken or model-specific tokenizer**: Integrate tiktoken (or a Go equivalent) for OpenAI-style models so prompt and completion token counts are accurate. Use for context window checks and cost estimation.
- **Tool call overhead**: When building prompts that include tool definitions and results, add their token count to the total so context window limits are enforced correctly.
- **Context window management**: Truncate conversation history (e.g. keep last N messages or oldest + newest) when total tokens exceed the model limit. Prefer dropping middle messages; optionally summarize older context.
- **Streaming**: NeuronAgent supports streaming via `GenerateStream`; ensure clients can consume chunked responses and handle mid-stream errors.
- **Conversation history limits**: Enforce a max number of messages or max total tokens per session; evict oldest (or summarize) when exceeded.
- **Model-specific limits**: Store per-model max context (e.g. 128k for gpt-4) in config and use when truncating.
