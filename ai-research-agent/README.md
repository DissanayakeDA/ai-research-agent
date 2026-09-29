# AI Research Assistant — an LLM agent built from scratch

A small command-line research assistant that answers questions, **decides for itself** when it needs
to search the web, and replies with an answer plus **verified sources**.

It is built with plain Python: no LangChain, LangGraph, CrewAI or AutoGen. Every step of the agent
loop is written out explicitly so you can read it, trace it and change it.

> This is a learning project, not a production application.

---

## 1. What this project is

You ask a question. The assistant:

1. sends it to an LLM along with a description of the tools it can use,
2. lets the LLM decide whether to answer directly or call `web_search`,
3. runs the search in Python when asked and feeds the results back to the LLM,
4. repeats until the LLM writes a final answer (up to 5 iterations),
5. prints the answer with sources that are checked against what the search actually returned.

## 2. Why I built it

Agent frameworks make it easy to build an "AI agent" in ten lines, but they also hide how it works.
I wanted to understand, before using a framework:

- How does an LLM "use a tool" when it can only produce text?
- Who decides when to search, and where does that decision come from?
- What does the conversation with the model look like after a tool call?
- How do you stop an agent from looping forever or inventing sources?

This project is part of my article series on modern AI engineering.

## 3. Architecture

```
                 ┌──────────────────────────┐
  you ─────────▶ │  main.py  (CLI / UI)     │ ◀── renders events: progress or debug trace
                 └────────────┬─────────────┘
                              │ question                ▲ AgentResult (answer + sources)
                              ▼                         │
                 ┌──────────────────────────────────────┴──┐
                 │  agent.py   AGENT ORCHESTRATOR (loop)   │
                 │  messages = [system, user]              │
                 │  repeat ≤ 5×:                           │
                 │    response = llm.chat(messages, tools) │
                 │    tool_calls? ─ yes ─▶ execute, append │
                 │               └ no ──▶ final answer     │
                 └───┬───────────────────┬──────────────┬──┘
          messages + │                   │ name + JSON  │ answer text
          tool schemas                   │ arguments    │ + sources seen
                     ▼                   ▼              ▼
          ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
          │ llm.py           │ │ tools/registry.py│ │ response.py          │
          │ POST /chat/      │ │ validate args,   │ │ parse [n] citations, │
          │   completions    │ │ dispatch, catch  │ │ attach real URLs,    │
          └────────┬─────────┘ │ tool errors      │ │ flag invented ones   │
                   ▼           └────────┬─────────┘ └──────────────────────┘
           OpenAI-compatible            ▼
           LLM (OpenAI, Groq,  ┌──────────────────┐
           Ollama, …)          │ web_search.py    │──▶ Tavily Search API
                               └──────────────────┘
```

| Component | File | Responsibility |
|---|---|---|
| User Interface | `main.py` | CLI, progress messages, debug trace, printing answer + sources |
| Agent Orchestrator | `app/agent.py` | The loop, system prompt, iteration limit, emitting events |
| LLM | `app/llm.py` | Raw HTTP client for any OpenAI-compatible `/chat/completions` API |
| Tool Registry | `app/tools/registry.py` | Tool schemas for the LLM; validates and dispatches tool calls |
| Web Search Tool | `app/tools/web_search.py` | `web_search` tool + Tavily provider (swappable) |
| Final Response Generator | `app/response.py` | Numbers sources, resolves `[n]` citations, flags invented ones |
| Configuration | `app/config.py` | Loads and validates environment variables |
| Data models | `app/models.py` | Pydantic shapes shared by all components |

```
ai-research-agent/
├── app/
│   ├── agent.py  config.py  llm.py  models.py  response.py
│   └── tools/  __init__.py  registry.py  web_search.py
├── tests/      conftest.py  test_agent.py  test_live.py
├── main.py  requirements.txt  pytest.ini  .env.example  .gitignore
```

## 4. How an AI agent works in this project

An **agent** here is an LLM in a loop with tools, plus Python code that runs the tools and enforces the rules.

```python
messages = [system_prompt, user_question]
for iteration in range(max_iterations):
    response = llm.chat(messages, tools)          # the model decides: text or tool call?
    messages.append(response)
    if not response.tool_calls:
        return response                           # plain text = final answer
    for call in response.tool_calls:
        result = registry.execute(call)           # WE run the tool, in Python
        messages.append(tool_message(call, result))
```

Key ideas:

- **The LLM makes the decisions; Python carries them out.** Nothing in the code checks whether the
  question contains "latest" or "2026". The model decides to search based on the system prompt, the
  tool description, and the question.
- **The `messages` list is the agent's entire memory.** The API is stateless: each call resends
  the whole conversation, including previous tool calls and results.
- **The model is told today's date.** Without it, the model can't tell that a question about
  "2026" needs fresh information.
- **Loops need a limit.** On the last allowed iteration, tools are disabled (`tool_choice: "none"`)
  and the model is told to answer with what it has, so the agent always finishes.

## 5. How tool calling works

1. **We describe the tool.** The registry turns a Pydantic model into a JSON Schema and sends it with every request:
   ```json
   {"type": "function", "function": {
     "name": "web_search",
     "description": "Search the web for up-to-date information. Use it for recent events ...",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
   ```
2. **The model asks for it.** Instead of text, it returns a structured request. The arguments are a *JSON string* the model wrote:
   ```json
   {"role": "assistant", "content": null,
    "tool_calls": [{"id": "call_abc", "type": "function",
                    "function": {"name": "web_search", "arguments": "{\"query\": \"RAG hallucinations\"}"}}]}
   ```
3. **We validate and run it.** Unknown tool? Malformed JSON? Missing `query`? Each becomes an error
   message *sent back to the model*, which can usually correct itself. The program doesn't crash.
4. **We send the result back** as a `tool` message linked by `tool_call_id`. It has to come after
   the assistant message that requested it:
   ```json
   {"role": "tool", "tool_call_id": "call_abc",
    "content": "{\"results\": [{\"source_id\": 1, \"title\": \"...\", \"url\": \"...\", \"snippet\": \"...\"}]}"}
   ```
5. **The model continues** with the results in context: it searches again or writes the final answer.

### How sources stay honest

The LLM never writes the source list. Search results are numbered by Python (`source_id`), the model
cites `[1]`, `[2]`, and `response.py` maps those numbers back to the URLs that were actually retrieved.
If the answer cites a number that doesn't exist or includes a URL no tool returned, it is shown as a
**warning**, never as a source. Anything the model adds from its own training must go in a paragraph
starting with `From general knowledge:`.

## 6. Installation

Requires Python 3.11+.

```bash
cd ai-research-agent
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 7. Configure API keys

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then edit `.env`:

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | yes | Key for your OpenAI-compatible provider |
| `SEARCH_API_KEY` | yes | [Tavily](https://tavily.com) key (free developer tier) |
| `LLM_BASE_URL` | no | Default `https://api.openai.com/v1`. Groq: `https://api.groq.com/openai/v1`, Ollama: `http://localhost:11434/v1` |
| `LLM_MODEL` | no | Default `gpt-4.1-mini`. Must support tool calling |
| `MAX_ITERATIONS` | no | Default `5` (min 2) |
| `MAX_SEARCH_RESULTS` | no | Default `5` |
| `DEBUG` | no | `true` to always show the debug trace |

`.env` is git-ignored. Never put keys in source code.

## 8. Run it

```bash
python main.py                  # interactive
python main.py --debug          # show every step of the agent loop
python main.py "What is RAG?"   # one question, then exit
```

Tests:

```bash
pytest -v                       # offline tests: fake LLM + fake search, no keys needed
RUN_LIVE_TESTS=1 pytest -v -m live               # real APIs (bash)
$env:RUN_LIVE_TESTS=1; pytest -v -m live         # real APIs (PowerShell)
```

## 9. Example interaction

The source titles and URLs below are placeholders. Your results will differ.

```
Ask a research question:
> How does RAG reduce hallucinations in LLM applications?
Deciding whether a web search is needed...
Searching the web for: 'how RAG reduces hallucinations in LLMs' ...
Found 5 results.
Generating answer...

------------------------------------------------------------
Answer:

RAG retrieves relevant documents at query time and places them in the prompt, so the model
answers from supplied evidence rather than memory alone [1]. Studies report fewer unsupported
claims when answers are grounded in retrieved text [2][4] ...

From general knowledge: RAG reduces but does not eliminate hallucinations; poor retrieval
can still lead to wrong answers.

Sources:
  [1] <title of result 1>
      https://...
  [2] <title of result 2>
      https://...
  [4] <title of result 4>
      https://...

Basis: web search (3 of 5 retrieved sources cited)
------------------------------------------------------------
```

With `--debug`, the same run shows each step:

```
[USER]            How does RAG reduce hallucinations in LLM applications?
[LLM CALL 1]      sending 2 messages | tools=['web_search'] | tool_choice='auto'
[AGENT DECISION]  Needs more information -> requested tool(s): web_search (tokens: ...)
[TOOL]            web_search({"query": "how RAG reduces hallucinations in LLMs"})
[TOOL RESULT]     {"query": ..., "results": [{"source_id": 1, "title": ..., "url": ...}, ...]}
[LLM CALL 2]      sending 4 messages | tools=['web_search'] | tool_choice='auto'
[AGENT DECISION]  Write the final answer (no tool call) (tokens: ...)
[FINAL]           Final answer received - citations checked, formatted answer below.
```

The trace shows only observable actions (requests, tool calls, results). It never asks the model to
reveal hidden reasoning.

## 10. Limitations

- **Snippets only.** The agent reads search snippets, not full pages, so answers are only as deep as the snippets.
- **No memory between questions.** Each question starts a fresh conversation.
- **Tool-use decisions are probabilistic.** The model may occasionally search when it doesn't need to, or skip a search it should have done.
- **Citation checks are structural, not semantic.** We verify that `[2]` exists and its URL is real, but not that source 2 actually supports the sentence.
- **"From general knowledge" labelling relies on the model following instructions.**
- **One search provider, no source-quality ranking.** Results are used in the provider's order.
- **Synchronous and single-user.** No streaming, no async, no concurrency.

## 11. What I learned

- Tool calling is not magic: the model returns a JSON *request*, and my code decides whether and how to run it.
- The "agent" is the loop around the LLM. Most of the engineering is validation, limits, error handling and bookkeeping.
- The tool description and system prompt drive the model's decisions, so writing them is a design task.
- Errors should go back to the model as observations, because a model that's told what went wrong can often fix it.
- Anything that must be correct (URLs, citations, limits) belongs in code, not in the prompt.
- Agent control flow can be unit-tested with a fake LLM; model *behaviour* needs evaluation, which is a separate discipline.

## 12. Future improvements (V2 roadmap)

1. **Multiple tools**: add a tool by writing a class with `name`, `description`, `args_model`, `run()` and registering it. Then watch how tool choice gets harder as the list grows.
2. **URL/document reading**: a `read_url` tool that fetches a page and extracts clean text, so the agent can go deeper than snippets.
3. **RAG**: chunk and embed fetched pages or local documents, store them in a vector index, and add a `search_documents` tool.
4. **Conversation memory**: keep `messages` across questions, then add summarisation or trimming when it grows too large.
5. **Persistent state**: save runs (messages, sources, events) to SQLite/JSON so sessions can be resumed and inspected.
6. **Better source ranking**: dedupe by domain, prefer primary/official sources, add recency, rerank snippets against the question.
7. **Agent evaluation**: a small dataset of questions with expected behaviour (should it search? which facts?), measuring tool-use accuracy, citation faithfulness (LLM-as-judge), latency and cost.
8. **LangGraph**: rebuild the same loop as a graph (`llm` node → conditional edge → `tools` node) and compare what the framework gives you: checkpoints, human-in-the-loop, streaming.
9. **Production deployment**: FastAPI + streaming responses, async I/O, tracing (OpenTelemetry), rate limiting, caching, cost budgets, secret management, and prompt-injection defences for web content.
