![RAG Architectures header](assets/rag-architectures-ai-builders-should-understand/header.svg)

# RAG Architectures AI Builders Should Understand

**[View on GitHub](https://github.com/neurondb-ai/neurondb)** | **[Download Latest Release](https://github.com/neurondb-ai/neurondb/releases)** | **[Documentation](/docs/rag)**

Large language models are exceptionally good at producing fluent text. They are not, by default, good at staying current, respecting private knowledge boundaries, or proving where an answer came from. That gap is exactly where most AI products fail: the demo looks impressive, but the system is not trustworthy when a user relies on it.

Retrieval-Augmented Generation (RAG) closes the gap by designing an evidence path. Instead of letting the model “reason from memory,” you route the request through retrieval, enforce access rules, collect supporting sources, and then ask the model to answer from those sources with citations. In practice, RAG is less about prompting and more about engineering: a data pipeline, a contract, and an operational loop.

## What RAG Means in Practice

At its simplest, RAG is two steps: **retrieval** finds evidence and **generation** writes an answer constrained by that evidence.

![RAG in one picture: retrieve → generate](assets/rag-architectures-ai-builders-should-understand/rag-pipeline.svg)

When you build RAG, you are designing the request path end-to-end: the system receives a question (often with conversation state and permissions), retrieves candidate evidence (chunks, entities, tickets, tables, logs), assembles context (dedupe, rank, filter, format, budget tokens), and produces an answer that is auditable (citations, traces, and metrics). That audit trail is the difference between “it answered” and “we can explain why it answered.”

### Why RAG is an architecture (not a feature)

RAG exists because two things are true at the same time:

**Retrieval is brittle.** If you retrieve the wrong evidence, the answer will be wrong. If you retrieve a mixture of relevant and irrelevant chunks, the model often averages them into a plausible but incorrect response.

**Generation is persuasive.** Even with strong evidence, models can overgeneralize, skip qualifiers, or invent bridging statements. You need a prompting and citation contract that constrains behavior, plus guardrails that handle “no evidence” and “low confidence” cases.

Good RAG systems solve both problems jointly: retrieval that is high-recall *and* high-precision, and generation that is grounded, scoped, and testable.

## Why builders use RAG

Training data ages quickly. Your product data, policies, tickets, and runbooks change daily. RAG is how you ship AI when answers must be current, private, and verifiable. It also creates trust at scale because you can inspect what was retrieved, from where, under which access rules, and how strongly the system believed each source was relevant.

## A practical mental model: the parts that decide quality

Most RAG failures are predictable. The biggest levers are (1) how you chunk and label content (structure boundaries and metadata matter as much as size), (2) whether embeddings fit the domain and how you handle drift, (3) how you retrieve (vector, hybrid, multi-vector, or graph traversal), and (4) how you filter and rerank (precision gates). Once you have candidates, the remaining work is context assembly (dedupe, diversity, ordering, token budgeting), a generator contract (citations, refusal rules, “use only sources”), and operations (observability, evaluation, regression prevention, re-indexing, and security).

## Core RAG patterns

RAG is not one architecture. It is a family of patterns trading off **latency**, **cost**, **control**, and **correctness**. The best approach depends on what “failure” means in your product: a harmless mistake in a consumer chatbot is very different from a mistake in compliance, production troubleshooting, or internal decision-making.

![The RAG pattern landscape](assets/rag-architectures-ai-builders-should-understand/rag-pattern-landscape.svg)

### Pattern comparison (fast way to decide)

| Pattern | Use when | Core move | Extra components | Typical cost/latency |
|---|---|---|---|---|
| Basic RAG | single-shot Q&A, docs, FAQ | embed → top-k | chunking + vector index | low |
| Conversational RAG | follow-ups depend on history | rewrite → retrieve | question rewriter | low–medium |
| Filtered RAG | mistakes are expensive | retrieve → rerank/filter | reranker/filters | medium |
| Adaptive RAG | retrieval sometimes hurts | route: no-retrieve vs retrieve | router + thresholds | low–medium |
| Hypothesis-Driven RAG | queries are vague | draft → retrieve → refine | draft step + guardrails | medium |
| Agent-Driven RAG | multi-step work matters | plan → retrieve → verify | planner + tools + stop rules | high |
| Graph-Based RAG | relations drive meaning | traverse → explain | entity graph + traversal | medium–high |

### Basic RAG

Basic RAG is the default starting point: embed the question, retrieve top-k chunks, assemble context, and generate with citations. It is the fastest to ship and the easiest to debug because the pipeline is simple and the failure modes are visible.

In practice, Basic RAG breaks in three common ways. First, **chunk mismatch**: the answer spans multiple chunks, but you only retrieve one. Second, **semantic near-miss**: embeddings retrieve something “related” rather than “answering.” Third, **context dilution**: top-k includes noise, and the model blends it into the response. The most useful early metrics are straightforward: does the top-3 contain the correct source, does the answer cite at least one relevant chunk, and do you correctly refuse when evidence is missing?

### Conversational RAG

Conversational RAG is Basic RAG plus one critical step: **rewrite the user’s message into a standalone retrieval query** using chat history. Many follow-ups are not retrievable as written (for example: “what about rate limits?”, “does that apply to enterprise?”, “ok, and what should I do next?”). A rewrite turns pronouns and context into explicit terms: product names, constraints, IDs, and the specific decision being discussed.

The key is to treat rewriting as a retrieval tool, not a stylistic improvement. Keep the user’s intent and scope, but make the query searchable. The main failure mode is rewrite drift: the rewritten query subtly changes the question. The fix is operational, not theoretical: log the raw question and the rewrite, evaluate both, and keep a rollback path when the rewriter behaves badly.

### Filtered RAG

Filtered RAG assumes retrieval returns a mixture of good and bad evidence and adds a quality gate before generation. Instead of trusting the first top-k, you typically retrieve broader (for recall), rerank (for precision), apply policy and access controls, and only then generate from the remaining evidence.

![Filtered RAG: retrieve then rerank then answer](assets/rag-architectures-ai-builders-should-understand/filtered-rag-rerank.svg)

This pattern is the default choice when errors are expensive or the corpus is messy (duplicates, contradictory docs, comments mixed with canonical policies). The subtle risk is over-filtering: you can end up with empty evidence. Your generator must treat “no evidence” as a first-class outcome, either refusing, asking a clarifying question, or escalating to a human workflow depending on product needs.

### Adaptive RAG

Adaptive RAG asks a pragmatic question: **should we retrieve at all?** Some tasks are self-contained (summarizing pasted text, rewriting, formatting). Retrieval adds cost and noise. Adaptive systems route requests into “no retrieval,” “shallow retrieval,” or “deep retrieval,” often based on simple rules at first and a learned router later.

A reliable starting approach is to encode a small set of defensible heuristics: if the user pasted content, skip retrieval; if the question references private topics or internal entities, retrieve; if the system’s confidence is low, retrieve deeper and rerank. The win is cost control without sacrificing correctness on the queries that actually need external evidence.

### Hypothesis-Driven RAG

Hypothesis-driven RAG is designed for vague queries. It drafts a provisional interpretation (“this might be about X”), uses that draft to extract missing specifics, retrieves with a sharper query, and then produces a final, cited answer. This tends to work well when users do not know the right vocabulary or when the domain is messy and intent matters more than keywords.

The risk is bias: the draft can steer retrieval in the wrong direction. A practical mitigation is to retrieve using both the original user query and the draft-derived query, then merge and dedupe evidence before generation.

### Agent-Driven RAG

Agent-driven RAG treats retrieval as a plan rather than a single search. The system decomposes the task, issues multiple retrieval calls, reads results, verifies gaps, and stops when evidence is sufficient. This is powerful for multi-step analysis (compare options and recommend), troubleshooting (diagnose, propose fix, validate), and research with branching paths.

![Agent-driven RAG loop](assets/rag-architectures-ai-builders-should-understand/agent-driven-rag-loop.svg)

The biggest engineering requirement here is a **stop condition**. Without a clear budget and diminishing-returns check, agents spiral into cost. Production systems use explicit ceilings (max tool calls, max tokens, max latency) plus an evidence threshold (“new evidence is no longer changing the answer”) to end the loop predictably.

### Graph-Based RAG

Graph-based RAG is for domains where relationships carry meaning: ownership, dependencies, citations, policy applicability, and lineage questions. Instead of retrieving only by similarity, you retrieve by traversal: identify starting entities, walk edges (depends_on, owned_by, authored_by, cites), and collect a subgraph as evidence. The output is often stronger because the system can explain *why* a source is relevant, not just that it is similar.

![Graph-based RAG: traverse relations, then explain](assets/rag-architectures-ai-builders-should-understand/graph-based-rag.svg)

The operational risk is freshness and coverage. Graphs can encode bias and incompleteness. If edges are missing or stale, the reasoning chain breaks. You should monitor graph update lag and validate coverage for the entity types users ask about most.

## How to pick the right pattern

Start with Basic RAG and make it measurable. Add conversational rewriting only when follow-ups are common. Add reranking and filtering when mistakes are expensive or the corpus is noisy. Use adaptive routing when you have mixed workloads and need to control cost. Choose hypothesis-driven retrieval when queries are vague and ambiguous. Use agents when multi-step work is the product, not an implementation detail. Use graphs when relationships must be explained and similarity search alone can’t answer “why.”

![Picking the right RAG pattern](assets/rag-architectures-ai-builders-should-understand/rag-pattern-picker.svg)

## Operational reality: what makes RAG boring and reliable

Most “RAG improvements” are operational. A working demo is not a working system. The difference is disciplined data preparation, evaluation, and observability.

![RAG is architecture plus operations](assets/rag-architectures-ai-builders-should-understand/rag-ops-loop.svg)

Treat operations as part of the architecture. The checklist below is what keeps RAG systems stable as the corpus, traffic, and product expectations grow.

- **Chunking and metadata**: Tune chunk size and overlap per document type, preserve structure (headings, tables, code blocks), and retain metadata boundaries (source, version, ownership, access labels) so retrieval can be filtered and audited.
- **Embedding drift and versioning**: Drift is inevitable. Choose a re-embedding strategy (versioned embeddings, rolling updates, or periodic rebuilds) and record embedding/model versions in metadata so you can reproduce results and manage migrations.
- **Index health**: As the corpus grows, indexing latency and recall can change. Monitor both continuously and retune index parameters over time rather than treating them as “set and forget.”
- **Evaluation and regressions**: Maintain a fixed, curated evaluation set (queries, expected sources, and “should refuse” cases). Run regression evals after any change to chunking, embeddings, retrieval parameters, or reranking.
- **Quality telemetry**: Track citation coverage, empty-retrieval rate, reranker hit-rate, and refusal quality so you can detect quality drops before users do.
- **Security and access control**: Enforce access control at retrieval time (not after generation) so unauthorized context never enters the prompt.
- **End-to-end tracing**: Log the full trace from query → retrieved ids → scores → final context → answer so incidents are debuggable and improvements are measurable.

## Conclusion

RAG shifts AI from storyteller to assistant by making evidence, permissions, and evaluation part of the product. The pattern you choose sets the tradeoffs you will live with: Basic RAG optimizes for speed and debuggability, reranking and filtering buy precision when stakes are higher, adaptive routing controls cost on mixed workloads, agents unlock multi-step work at the price of orchestration, and graphs add relational explainability when similarity alone is not enough. Whatever you choose, reliability comes from operational discipline: structure-aware chunking, versioned embeddings, measurable retrieval quality, refusal correctness, and end-to-end tracing from query to sources to answer. Start with the simplest architecture that meets the requirement, make it observable and testable, and then add complexity only when it produces a clear, repeatable improvement in outcomes.

## Related Blog Posts

[RAG: Retrieval-Augmented Generation With PostgreSQL](/blog/rag-complete-guide)

Comprehensive walkthrough of the classic RAG pipeline with implementation details.

[Semantic Search Over Text with NeuronDB](/blog/neurondb-semantic-search-guide)

How retrieval quality starts with embeddings, chunking, and search design.