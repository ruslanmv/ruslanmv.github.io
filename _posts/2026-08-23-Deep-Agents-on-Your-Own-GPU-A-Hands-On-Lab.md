---
title: "Deep Agents on Your Own GPU: A Hands-On Lab"
excerpt: "The three layers an agent is made of, where a plain ReAct loop breaks, and a step-by-step lab that fixes it with a deep agent — running entirely on your own graphics card with Ollama and uv."
description: ""
date: 2026-08-23
header:
  image: /assets/images/posts/2026-08-23-Deep-Agents-Ollama/deep-agents-ollama.jpg
  teaser: /assets/images/posts/2026-08-23-Deep-Agents-Ollama/wallpaper.jpg
  caption: "One clear head, three fresh helpers, one graphics card."
tags:
  - agents
  - deep-agents
  - ollama
  - local-llm
  - uv
  - langchain
  - python
---

Ask an AI agent to look up one fact and it does fine.

Ask it to *"research three inference engines and write me a comparison"* and something sad happens. It starts well, drifts around step ten, and hands you an answer that quietly forgot half of what you asked for.

That failure has a cause, and the cause has a fix. We're going to build both on your own laptop — the model runs on your graphics card, so there's no API key for it and no per-token bill.

Here's the plan: understand the **three layers** an agent is made of, see exactly where a plain **ReAct** agent breaks, then fix it with a **deep agent** and run every piece yourself.

Code: [github.com/ruslanmv/deep-agents-tutorial](https://github.com/ruslanmv/deep-agents-tutorial)

## 1. The Three Layers

Almost every confusing conversation about agents comes from mixing up three different things. Separate them and the rest of this post is easy.

<figure>
  <img src="/assets/images/posts/2026-08-23-Deep-Agents-Ollama/layers.svg" alt="Three stacked layers: the model at the bottom, the harness wrapping it, and the deep-agent layer on top." style="width:100%">
  <figcaption>Each layer wraps the one below it. Nothing here is magic.</figcaption>
</figure>

**Layer 1 is the model.** By itself it does exactly one thing: text in, text out. No memory of yesterday. It can't open a file, run a search, or check its own work. Brilliant and completely helpless.

**Layer 2 is the harness** — the code you wrap around the model to make it useful. It keeps the conversation, offers the model a set of tools, actually runs those tools, and decides when the job is done. Model is the engine; harness is the rest of the car.

**Layer 3 is the deep-agent layer.** A plan, a filesystem, sub-agents, a strong prompt. This is the part most people mean when they say "agent framework", and it's the part this post is really about.

### The harness, in six lines

Here's layer 2 in full. Not a simplification — this is the shape of it:

```
while not done:
    reply = model(conversation)
    if reply asks for a tool:
        result = run_that_tool()
        conversation.append(result)
    else:
        done = True
```

That loop is a real, working agent. It also has a name: **ReAct** — reason, act, repeat. Every agent framework you've read about is a fancier version of those six lines.

And once you can see the loop, you can see what's wrong with it. Everything the agent knows lives in `conversation`. Every search result, every file it read, every wrong turn — all of it piles back into the model on every single turn. That list has a size limit. Hit the limit and the oldest part falls off the edge.

## 2. ReAct vs Deep Agent

So put them side by side on the same task.

<figure>
  <img src="/assets/images/posts/2026-08-23-Deep-Agents-Ollama/react-vs-deep.svg" alt="A ReAct agent's single context fills with raw tool output until the original question falls off the edge; a deep agent's orchestrator context stays short because notes go to files and messy work goes to sub-agents." style="width:100%">
  <figcaption>Same question, same model, same GPU. The difference is where the mess goes.</figcaption>
</figure>

The ReAct agent on the left isn't badly written. It's just holding everything in its head. Around step ten the raw JSON from its own searches has crowded out your question, and here's the cruel part: **nothing tells you.** No error, no warning. The agent keeps going, confidently, on a task it no longer fully remembers.

The deep agent on the right does the same work but keeps almost none of it. Findings go to files. The messy searching happens inside sub-agents. What comes back to the orchestrator is one line per sub-question, so it still has room to think at the end — which is when it actually needs to.

Picture a colleague doing this research with no notebook, no to-do list, and no desk. They'd manage three or four steps and start dropping things. Not because they aren't smart. Because you gave them nowhere to put anything.

A deep agent is the same loop, with a desk. Four things go on that desk:

- **A to-do list** it writes and ticks off, so it knows what it set out to do ten steps later.
- **A filesystem**, so notes go on paper instead of in its head.
- **Sub-agents**, each with their own blank conversation.
- **A long, specific prompt**, for consistent habits over hundreds of steps.

### The sub-agent trick

The third one is the clever bit, and it's worth being precise about because it's where most of the win comes from.

<figure>
  <img src="/assets/images/posts/2026-08-23-Deep-Agents-Ollama/subagent-isolation.svg" alt="The orchestrator sends one instruction; the sub-agent starts from a blank conversation, runs many searches in its own window, and returns one paragraph plus files. Both share a filesystem." style="width:100%">
  <figcaption>One instruction in, one paragraph out. The noise never crosses back.</figcaption>
</figure>

When the orchestrator delegates, the helper starts from a **completely blank conversation** — just the one instruction it was given. It runs a dozen searches, writes what it found to a file, and comes back with a single paragraph.

The orchestrator never sees those dozen searches. It gets the paragraph and the file. All the noise stayed with the helper and was thrown away.

Two consequences worth remembering. Each sub-agent call is **stateless** — no memory between delegations. And the sub-agent **cannot see your original request**, only the description you hand it. Vague descriptions are the number one way this goes wrong.

This is the architecture behind Claude Code and the "deep research" buttons that appeared in every AI product last year. And doing it locally makes it *easier* to understand, not harder: a cloud model's window is so big you never feel the problem, while your GPU's is small enough that you can watch it break in seconds — then watch the fix work.

## 3. What You Need

The floor for this lab is an **RTX 4080 Laptop GPU, which has 12 GB of VRAM**. That's the smallest card I'd want to do this on, and everything in the repo is sized to fit it comfortably. If you have something bigger, you get a longer context window for free.

| Card | VRAM | Context window to try |
|---|---|---|
| **RTX 4080 Laptop** (our floor) | **12 GB** | **24576** |
| RTX 4080 / 5080 Desktop, 5080 Laptop, 4090 Laptop | 16 GB | 32768 |
| RTX 4090 Desktop, 5090 Laptop | 24 GB | 40960 |
| RTX 5090 Desktop | 32 GB | 40960 and a bigger model |

Don't take my word for the numbers — `make doctor` reads the real figures off your card in a minute or two and tells you what fits. Card memory also varies by laptop model, and your desktop is using some of it too.
{: .notice--info }

No NVIDIA card? It still runs, on your CPU. It's slow enough to be annoying for a full deep-agent run, but every lab in this post will complete and every lesson still lands.

You'll also want:

- **Python 3.11 or newer**
- **[uv](https://docs.astral.sh/uv/)** — the Python package manager we'll use
- **[Ollama](https://ollama.com/download)** — serves the model on your GPU
- A free **[Tavily](https://tavily.com)** key so the agent can search the web

That last one is the only cloud service left. The model is yours; the internet still isn't. And if you'd rather not sign up for anything at all, skip ahead to [Lab 3](#9-lab-3-the-built-in-toolbox) — it needs no key and no network.

## 4. Setting Up with uv

Four commands take you from a fresh clone to a verified machine. Here they are; the next three sections walk through each one.

<figure>
  <img src="/assets/images/posts/2026-08-23-Deep-Agents-Ollama/setup.svg" alt="Four cards in a row: make setup builds a .venv with pinned versions, make env writes a .env for your Tavily key, make model pulls qwen3:8b at about 5 GB, and make doctor runs seven checks against your real card." style="width:100%">
  <figcaption>Sections 4 to 6 in one picture. Run them in this order.</figcaption>
</figure>

If you haven't used uv before: it's a Python package manager that's fast and, more usefully for a tutorial, boring. One command builds the environment and gets the versions right.

Install it if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then grab the repo and set up:

```bash
git clone https://github.com/ruslanmv/deep-agents-tutorial
cd deep-agents-tutorial
make setup
```

That's it. Behind the scenes `make setup` runs `uv sync`, which reads `pyproject.toml`, creates a `.venv` folder in the project, and installs everything. Nothing gets installed system-wide, and nothing you already have gets touched.

Here's the whole `pyproject.toml`, so there's no mystery about what you just installed:

```toml
[project]
name = "deep-agents-tutorial"
version = "0.3.0"
requires-python = ">=3.11"
dependencies = [
    "deepagents>=0.7.0",
    "langchain>=1.0.0",
    "langchain-ollama>=1.0.0",
    "langchain-openai>=1.6.0",
    "tavily-python>=0.5.0",
    "langfuse>=3.0.0",
    "python-dotenv>=1.0.0",
]

# We're not building a library here, just a folder of scripts to run.
[tool.uv]
package = false
```

Seven dependencies. `deepagents` is the harness, `langchain-ollama` connects it to your GPU, `langchain-openai` is for the gateway in Lab 5, `tavily-python` does the searching, `langfuse` is for tracing.

uv also writes a `uv.lock` file with the exact version of every package, including the ones you didn't ask for. It's committed to the repo, so if this lab works for me it works for you.
{: .notice--info }

Now make your config file:

```bash
make env
```

That copies `.env.example` to `.env`. Open it and paste in your Tavily key. Everything else already has a sensible default.

To run anything, use `uv run` instead of `python`, and uv sorts out the environment for you:

```bash
uv run python src/00_doctor.py
```

Or just use the make targets, which do exactly that. `make help` lists them all.

## 5. Pulling the Model

Time to download a brain. Make sure Ollama is running — on Mac and Windows it starts with the app, on Linux you may need `ollama serve` in another terminal.

Then:

```bash
make model
```

Which runs:

```bash
ollama pull qwen3:8b
```

It's about 5 GB, so go make coffee. When it finishes you'll see your models listed:

```
NAME        ID              SIZE      MODIFIED
qwen3:8b    500a1f067a9f    5.2 GB    12 seconds ago
```

**Why this model?** Because a deep agent is *nothing but tool calls*, and not every small model can do them. Plenty of otherwise-lovely models respond to a tool-call request with a polite paragraph explaining what they would do. That's useless here. Qwen3 8B calls tools reliably and fits in 12 GB with room for a decent context window.

Want something else? You can:

```bash
make model MODEL=qwen3:14b
```

Then set `DEEP_AGENT_MODEL=qwen3:14b` in your `.env`. Good options: `qwen3:4b` if you're tight on memory, `llama3.1:8b` as an alternative at the same size, `qwen3:14b` if you have 16 GB or more.

Whatever you pick, it must support tool calling. The next step checks that for you, so you'll know within a minute if you picked wrong.
{: .notice--warning }

## 6. Checking Your Machine

Before writing any agent code, let's find out whether this is going to work at all:

```bash
make doctor
```

![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-01-23-01-55.png)





That is a real run on the reference machine. Note check 5 against check 6b: the
estimate said ~7.7 GiB, the measurement said 8.4 GiB. Estimates of KV-cache size
run a little low, which is exactly why 6b exists — it loads the model and asks
Ollama where the weights actually went. `100%` is what you want. Anything less
and the remainder is on the CPU, dragging the whole run down to that speed.
{: .notice--info }

Seven checks, in the order they'd otherwise ruin your afternoon. Check 6 is the important one — that's your model proving it can actually call a tool. Check 5 is the one that says whether your card can hold the model and the context at the same time.

If something says FAIL, jump to [Troubleshooting] (https://github.com/ruslanmv/deep-agents-tutorial/blob/master/TROUBLESHOOTING.md) section. Get a clean run here before moving on. Every one of these failures shows up later as weird agent behaviour rather than a clear error, which is a miserable way to spend an evening.

## 7. Lab 1: The Plain Agent

Let's build that ReAct loop from section 1 for real, and watch it break.

Open `src/01_shallow_agent.py`. The interesting part is short:

```python
from langchain.agents import create_agent

from local_model import build_local_model
from search import internet_search

agent = create_agent(
    model=build_local_model(),
    tools=[internet_search],
    system_prompt="You are a helpful research assistant.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is PagedAttention in vLLM?"}]
})
print(result["messages"][-1].content)
print(f"\n[messages in context: {len(result['messages'])}]")
```

One model, one tool, one loop. Run it:

```bash
make shallow
```

![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-01-23-02-36.png)



You'll get a good answer with sources, and something like `[messages in context: 4]` at the bottom. One question, a search or two, done. For this, the plain loop is perfect. Don't let anyone tell you that you need an agent framework for a question like this.

**Now break it.** Change the question to something with parts:

```python
"content": (
    "Research the top 3 open-source LLM inference engines in 2026, "
    "compare their strengths, and write a short report."
)
```

Run it again and watch that message count. It climbs and climbs, because every raw search result is still sitting in the conversation. Somewhere past twenty tool calls, the oldest messages start falling off the edge of the window — and Ollama does that **silently**. No error, no warning. The agent just quietly loses the beginning of the conversation, which is where you told it what you wanted.

Read the answer and you'll usually find one of the three engines is thin, or the "compare their strengths" part got skipped. It didn't ignore you. It genuinely forgot.

This is the moment worth sitting with. Nothing crashed. Nothing logged a problem. The agent was confidently wrong, and the only reason you know is that you read the output carefully.
{: .notice--warning }

## 8. Lab 2: Making It Deep

Same loop, now with a desk. Open `src/02_deep_agent.py`.

### The helper

First we describe the researcher. It's just a dictionary:

```python
RESEARCHER_PROMPT = """You are a focused researcher investigating exactly one question.

1. Call internet_search one to three times for the question you were given.
2. Call write_file to save your findings to the filename you were told to use.
   Use short markdown bullets with source URLs.
3. Reply with a compact summary of your findings.

The agent that called you only sees your final reply, not your searches."""

research_subagent = {
    "name": "researcher",
    "description": (
        "Investigates a single focused research question and writes its "
        "findings to a file. Use for every web-research sub-question."
    ),
    "system_prompt": RESEARCHER_PROMPT,
    "tools": [internet_search],
    # No "model" key: the helper reuses the main agent's model.
}
```

Notice how blunt that prompt is. Numbered steps, one instruction per line. Local models are good, but they follow plain orders much better than they follow flowing prose. Write for a competent new hire on their first day, not for a colleague who already knows how you work.

That comment at the end matters too. `deepagents` lets each helper use a different model, which sounds appealing — a big one to plan, a small fast one to fetch. On one graphics card it's a trap. Ollama can only hold so much, so it unloads one model to load the other, on **every single delegation**. Use one model for everybody and it stays put in VRAM.
{: .notice--warning }

### The main agent

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=ORCHESTRATOR_PROMPT,
    subagents=[research_subagent],
    backend=StateBackend(),
    middleware=[TodoListMiddleware()],
)
```

That one call gives the agent a to-do tool, a full set of file tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`), and a `task` tool for delegating. Your `internet_search` is added alongside them.

That `TodoListMiddleware()` line isn't optional decoration. In `deepagents` 0.7.x the to-do tool **isn't included by default** on most models, so without it your agent has no `write_todos` — and a prompt telling it to "make a plan first" is describing a tool that doesn't exist. Check 7 of `make doctor` verifies it's there.
{: .notice--warning }

### Run it

```bash
make deep
```
![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-01-23-03-21.png)

That is an unedited run on the reference machine: 49 seconds, six tool-calling
turns, three sub-agent delegations.

Look at the shape of it: a plan, three delegations, then a read and a write. The
three long gaps are the sub-agents — nine to twelve seconds each of searching
that the orchestrator never saw. It finished with **14 messages** in its own
context.

### What an 8B model does and doesn't obey

Here is where I have to be straight with you, because this is a real run and not
a demo I curated.

The prompt asks the researchers to write to `/notes_1.md`, `/notes_2.md` and
`/notes_3.md`. They wrote to `/research/top-3-open-source-llm-inference-engines-2026.md`
and `/results/2026_open_source_llm_inference_engines.md` instead — inventing their
own directories and names. Three delegations produced only **two** files, so one
researcher skipped the write entirely. And the orchestrator called `read_file`
once rather than three times before writing the report.

One sub-agent also drifted off the question. Asked about inference *engines*, it
came back with notes on Kimi K3 and GLM 5.2 — which are models, not engines.
That content is in the filesystem and never made it into the report, so the run
still produced a sound answer, but it was luck rather than discipline.

None of this is a bug in the harness. It is what an 8B model does with a
five-step procedure: it gets the shape right and the details approximate. The
architecture held — plan, delegate, isolate, synthesise — while exact filenames
and step counts did not.

Two honest conclusions. First, don't build anything load-bearing on a local 8B
following a long procedure; verify what it wrote rather than trusting that it
followed instructions. Second, if you need the details obeyed, that is what a
bigger model buys you — try `qwen3:14b` on 16 GB and compare. The value of
running this locally is that you can see the difference for yourself in under a
minute, for free.
{: .notice--warning }

Now look at that last number and compare it to Lab 1 on the same task. **Eighteen messages** — for three helpers and a dozen searches. All that digging happened in the helpers' conversations and never touched the main one.

That gap is the entire point of this post. And on a 24k window it isn't a nice optimisation, it's the difference between finishing the job and silently forgetting it.

![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-01-23-04-40.png)




### One gotcha when you print the files

The files come back as dictionaries, not strings:

```python
for path, data in sorted(result.get("files", {}).items()):
    content = data["content"]        # not data[:500]
    print(f"===== {path} =====\n{content[:500]}")
```

Each value is a `FileData` dict with `content`, `encoding` and timestamps. Slicing the dict directly throws `TypeError: unhashable type: 'slice'`, which is a rude way to end a run that otherwise went perfectly.
{: .notice--warning }

## 9. Lab 3: The Built-In Toolbox

So far we've used the toolbox without really looking at it. Let's look at it.

Call `create_deep_agent()` and the agent gets ten tools before you add any of your own. They fall into four groups.

**Planning.** Just `write_todos` — the agent writes a plan and ticks it off.

**Finding things.** `ls` to get oriented, `glob` to find files by name, `grep` to search their contents.

**Working with files.** `read_file`, `write_file`, `edit_file` for an exact string replacement, and `delete`.

**Getting help.** `task` hands a job to a sub-agent. `execute` runs a shell command, but only with a sandbox backend — more on that below.

That's the same shape of toolbox Claude Code gives itself, which is not a coincidence.

### The best-practice pattern: narrow before you read

`glob` → `grep` → `read_file`. That ordering is the single most useful habit a file-using agent can have.

<figure>
  <img src="/assets/images/posts/2026-08-23-Deep-Agents-Ollama/narrow-before-read.svg" alt="Left: reading twenty files fills the context window with files the agent never needed, and the one file that mattered falls past the end of the window. Right: ls, then glob to cut twenty files to twelve, then grep to cut twelve to three, then three reads — leaving most of the window free." style="width:100%">
  <figcaption>Same task, same window. The difference is how much of it you spend on files you didn't need.</figcaption>
</figure>

Read everything and let the model sort it out, and you fill the window with code the agent didn't need — then truncate the part it did.

Narrow first and it's three cheap calls instead of twenty expensive ones. Say so explicitly in your system prompt, because models don't always do it unprompted.

### A lab with no API key

Time to see it. This lab needs **no Tavily key and no network** — we hand the agent a tiny fake project in its own filesystem and ask for an audit:

```bash
make toolbox
```

`src/04_toolbox_agent.py` seeds four small files (a billing module with a couple of real bugs, a users module, a utils module, a README), then asks the agent to find every `TODO`/`FIXME`, review the files that have them, and write `/AUDIT.md`.

Note what it *doesn't* pass:

```python
agent = create_deep_agent(
    model=model,
    tools=[],          # no custom tools at all — built-ins only
    system_prompt="You are a meticulous code auditor. ...",
    subagents=[reviewer],
    backend=StateBackend(),
    middleware=[TodoListMiddleware()],
)
```

`tools=[]`. Everything the agent does in this lab comes out of the box.

### Seeding the filesystem

You put files into the agent's world by passing them to `invoke`:

```python
from deepagents.backends.utils import create_file_data

result = agent.invoke({
    "messages": [{"role": "user", "content": AUDIT_TASK}],
    "files": {p: create_file_data(c) for p, c in PROJECT.items()},
})
```

Use `create_file_data()` rather than writing `{"content": "..."}` by hand. A bare content dict looks fine and works for `ls`, `grep`, `read_file` and the rest — but `glob` sorts results by modification time, so it dies on a missing timestamp with a bare `'modified_at'` error while every other tool carries on. I lost a while to that one.
{: .notice--warning }

### What you see

The script prints which tools got called, in order:

![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-01-23-07-06.png)


That `order:` line is the whole lab. You can read the agent's thinking off it: plan, get oriented, narrow by filename, narrow by content, delegate the reading, write the result, tidy it up. Nobody wrote that sequence — it chose it.

Because the project is four files you can hold in your head, you can also check its homework. That's why the demo is deliberately tiny: a demo you can't verify isn't a demo, it's a vibe.

### About `execute`

The one tool that won't work here is `execute`. With the default `StateBackend` the files live in memory, not on a real disk, so there's no shell to run anything in. Ask for it and you get:

```
Error: Execution not available. This agent's backend does not support
command execution (SandboxBackendProtocol).
```

That's the correct answer, not a bug. If you want a real shell, swap the backend for a sandbox one — and think hard first, because an agent with a shell can do everything you can do.
{: .notice--warning }


![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-01-23-07-39.png)

## 10. Lab 4: Watching It Work

An agent that runs on its own for five minutes needs to be watchable. Since `deepagents` is built on LangGraph, [Langfuse](https://langfuse.com) traces it in two lines — this is the same setup as my [previous post](https://ruslanmv.com/blog/Langfuse-Observability-for-LLM-Applications):


This lab needs a Langfuse key. Get a free one at [https://cloud.langfuse.com](https://cloud.langfuse.com),
  then put it in .env (run `make env` first if you have no .env yet).

  The model itself is local — this is the one hosted piece.


```python
from langfuse.langchain import CallbackHandler

handler = CallbackHandler()
result = agent.invoke(
    {"messages": [{"role": "user", "content": "..."}]},
    config={"callbacks": [handler]},
)
```

Add your Langfuse keys to `.env` and run:

```bash
make observed
```
![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-01-23-13-15.png)


You get one trace with the whole tree in it: every plan, every search with its arguments, every helper as a nested span. You don't have to pass the handler down to the helpers — LangGraph carries it along automatically.

<figure>
  <img src="/assets/images/posts/2026-08-23-Deep-Agents-Ollama/trace-tree.svg" alt="A waterfall of one 96-second run: write_todos, then three researcher sub-agents each containing their own searches and model calls, then write_file. One nine-second model call inside the first sub-agent is highlighted as a model reload." style="width:100%">
  <figcaption>The shape of a healthy run — and one step that is eight seconds longer than its neighbours.</figcaption>
</figure>


![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-01-23-16-41.png)
Locally, what you're watching is different from the cloud. There's no bill, so you're not tracking spend. You're tracking **time and forgetting**:

- A step that took eight seconds longer than its neighbours? Ollama reloaded the model. Fix `keep_alive`.
- The agent repeating a search it already did? Its context got truncated. Raise `num_ctx`.

Both are obvious in a trace and nearly invisible in terminal output.

![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-01-23-14-41.png)


## 11. Lab 5: One URL for Every Model

So far the agent talks straight to Ollama on this machine. That's fine until you have models in more than one place — a gaming PC upstairs, a Colab GPU, your own OpenAI key for the jobs a local 8B can't manage.

[OllaBridge](https://github.com/ruslanmv/ollabridge) is my answer to that. It puts **one OpenAI-compatible URL** in front of everything you can reach, and compute nodes dial *out* to it over WebSockets, so no port forwarding and no VPN. Your laptop in a café can use the GPU at home.

<figure>
  <img src="/assets/images/posts/2026-08-23-Deep-Agents-Ollama/one-url.svg" alt="A deep agent points at localhost:11435/v1. Behind that one URL: Ollama on this laptop, a bigger GPU upstairs, a borrowed Colab GPU, and a hosted provider key. The remote nodes dial out to the gateway over WebSockets." style="width:100%">
  <figcaption>One URL. Which machine answers becomes a config question, not a code question.</figcaption>
</figure>

### Two commands

In one terminal, start the gateway. It's a service you run beside the project, not a dependency of it — so install it wherever you like, just not into `.venv`:

```bash
pip install ollabridge
ollabridge start --auth-mode local-trust --host 127.0.0.1
```

In another, run the lab:

```bash
make ollabridge
```

That's the whole lab. Nothing to add to `.env`, no key to copy, no file to edit.

Those two flags are why. `--auth-mode local-trust` tells the gateway to trust requests coming from this machine and skip the key check; `--host 127.0.0.1` keeps it listening on this machine only, which is what makes that trust reasonable. Leave the host off and OllaBridge binds to every interface and says so in yellow — fine on your own network, worth being deliberate about.

The endpoint comes up at `http://localhost:11435/v1`. Note **11435** — one above Ollama's own port, because both are running.

Here's what you should see:


![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-02-00-01-06.png)



A real deep agent — planning, grepping, writing files — over an OpenAI-compatible URL that could just as easily be pointing at the GPU box upstairs.

### What changed in your code

Nothing. `make ollabridge` sets `DEEP_AGENT_BACKEND=ollabridge` for that one command, and `build_local_model()` hands back a different client for each backend — `ChatOllama` for the direct path, `ChatOpenAI` for the gateway:

```python
model = ChatOpenAI(
    model="qwen3:8b",
    base_url="http://localhost:11435/v1",
    temperature=0.0,
    profile={"max_input_tokens": 24576},   # still needed, see section 12
)
```

Same model, same weights, same GPU. Different front door.

Want every lab on the gateway, not just this one? Then `.env` earns its keep: add `DEEP_AGENT_BACKEND=ollabridge` and `make deep` goes the same way.

Two overrides exist for when the defaults are wrong, and `.env` is where they go. `OLLABRIDGE_URL` if your gateway is on another machine or another port. `OLLABRIDGE_API_KEY` if you started it with `--auth-mode required` instead — then paste the key OllaBridge printed at startup. You'll know which case you're in, because step 1 says `HTTP 401 — the key was checked` and tells you both ways out.

### Why the script checks instead of assuming

Step 3 looks redundant when it passes. It isn't, and the reason is worth a minute.

A deep agent is nothing but tool calls. An OpenAI-compatible proxy that forwards chat but quietly ignores the `tools` field will return you HTTP 200 and a confident prose answer, with nothing in the logs to say the tools were dropped. The agent doesn't fail. It just stops being an agent.

That was the state of OllaBridge until recently, and fixing it took two changes, not one: forward `tools` on the way in, *and* let messages carry `tool_calls` with null content. Miss the second and the loop dies on turn two, when the agent sends its own tool call back. Both landed in **0.1.7**, which is what `pip install ollabridge` gives you today.

So the probe stays in. If you point this at some other gateway, step 3 tells you in one second what would otherwise cost you an afternoon.
{: .notice--info }

### A trick worth stealing either way

`num_ctx` is an Ollama parameter, and the OpenAI chat API has nowhere to put it. Through *any* OpenAI-compatible front door you can't set the context window per request — you get whatever the model was built with.

Bake it into the model instead. `Modelfile.example` in the repo:

```dockerfile
FROM qwen3:8b
PARAMETER num_ctx 24576
PARAMETER temperature 0
```

```bash
ollama create qwen3-agent -f Modelfile.example
```

Set `DEEP_AGENT_MODEL=qwen3-agent` and the window travels with the model — direct, through a gateway, from any SDK. Worth doing even if you never touch a gateway.


## 12. Three Settings That Will Bite You

This is the section I wish someone had handed me. Pointing `deepagents` at Ollama is one line. Making it *behave* comes down to three settings, and all three fail **quietly** when they're wrong. They live in `src/local_model.py`:

```python
model = ChatOllama(
    model="qwen3:8b",
    num_ctx=24576,
    keep_alive="30m",
    reasoning=False,
    temperature=0.0,
    profile={"max_input_tokens": 24576},
)
```

### `num_ctx` — the default is too small

Ollama's out-of-the-box context window is a few thousand tokens, and going over it truncates instead of erroring. That's the silent forgetting from Lab 1. Always set this yourself.

### `profile` — the one nobody expects

This is the one I'd never have guessed, and I only found it by reading the library's source.

`deepagents` has a lovely feature: when the conversation gets long, it summarises the older part to make room. It decides *when* to do that by asking the model how big its window is, via `model.profile`.

`ChatOllama` doesn't answer that question. So `deepagents` falls back to a safe default for a **cloud-sized** model and only summarises at **170,000 tokens**. Your window is 24,576. That threshold is never reached, the feature never fires, and Ollama's silent truncation gets there first — every single time.

<figure>
  <img src="/assets/images/posts/2026-08-23-Deep-Agents-Ollama/context-budget.svg" alt="With a profile, compaction fires at 85 percent of the window, about 20,900 tokens, and the run continues. Without one, the window fills completely and Ollama truncates at 24,576, while the default 170,000-token trigger sits far off the scale, never reached." style="width:100%">
  <figcaption>The whole bug is a scale problem: the default threshold sits about seven screens past the end of your window.</figcaption>
</figure>

You can watch it happen:

```python
from deepagents.middleware.summarization import compute_summarization_defaults

# Without a profile:
compute_summarization_defaults(ChatOllama(model="qwen3:8b", num_ctx=24576))
# {'trigger': ('tokens', 170000), ...}      <- never happens

# With one:
compute_summarization_defaults(
    ChatOllama(model="qwen3:8b", num_ctx=24576, profile={"max_input_tokens": 24576})
)
# {'trigger': ('fraction', 0.85), ...}      <- 85% of your real window
```

One dictionary moves the threshold from 170,000 tokens to about 20,900, which is where it belongs. Nothing warns you if you leave it out. Your agent just gets forgetful. Check 7 of `make doctor` exists purely to catch this.
{: .notice--warning }

### `keep_alive` — because there are so many calls

One deep-agent run is dozens of separate model calls. Without `keep_alive`, Ollama unloads the model between them and you pay the load time again and again. Pinning it in VRAM is the difference between a two-minute run and a ten-minute one.

The last two are smaller. `reasoning=False` turns off visible thinking blocks, which eat the context you just fought for. `temperature=0.0` because a long chain of tool calls wants a reliable agent, not a creative one.

Something misbehaving? Every symptom I hit while writing this — and its fix — is collected in [TROUBLESHOOTING.md](https://github.com/ruslanmv/deep-agents-tutorial/blob/main/TROUBLESHOOTING.md) in the repo.
{: .notice--info }

## 13. Advanced Topics

Everything so far took `create_deep_agent()` as given. Let's open it up.

There are exactly two levers, and knowing which one to reach for is most of the skill. **Middleware** is code that wraps individual calls — use it to watch or intervene as things happen. **Harness settings** are declarative defaults attached to a model — use them to decide what the agent is even offered.

Both are in one runnable lab, and it needs no API key:

```bash
make advanced
```

![](/assets/images/posts/2026-08-23-Deep-Agents-Ollama/screenshots/2026-09-02-00-03-03.png)

### 14.1 Middleware

Remember the six-line loop from section 1. Middleware is how you get *inside* it.

<figure>
  <img src="/assets/images/posts/2026-08-23-Deep-Agents-Ollama/middleware.svg" alt="Left: the middleware stack in order, with Filesystem, SubAgent, Summarization and PatchToolCalls built in, your Extra entry spliced in after them, and AnthropicPromptCaching as the tail; entries lower down sit closer to the model. Right: wrap_tool_call gets the request and a handler — call the handler and the tool runs, or return a ToolMessage instead and the tool never runs." style="width:100%">
  <figcaption>Where you sit in the stack decides what you can see. Whether you call the handler decides what runs.</figcaption>
</figure>

You write a class with hooks for the moments you care about. `before_agent` and `after_agent` fire once per run. `before_model` and `after_model` bracket each model call. And the two powerful ones, `wrap_model_call` and `wrap_tool_call`, wrap those calls — they hand you the request *and* the handler, and let you decide what to do with both. Every hook has an async twin (`awrap_model_call`, and so on).

Here's the shape:

```python
class ToolAuditMiddleware(AgentMiddleware):
    name = "ToolAuditMiddleware"

    def wrap_model_call(self, request, handler):
        # request.tools is what the model is about to be shown.
        self.offered.append([t.name for t in request.tools])
        return handler(request)

    def wrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        if name in self.blocked:
            # Don't call handler() — the tool simply never runs.
            return ToolMessage(
                content=f"Refused: '{name}' is blocked by policy.",
                tool_call_id=request.tool_call["id"], name=name,
            )
        started = time.perf_counter()
        result = handler(request)
        self.calls.append((name, (time.perf_counter() - started) * 1000))
        return result
```

Not calling `handler()` is the entire guardrail. The model asked, you said no, and your message becomes the tool result. In the lab the agent asks to delete a file, gets refused, and finishes the job anyway:

```
write_todos     0.7 ms
grep            1.0 ms
write_file      0.8 ms
refused by the guard: delete
```

That's auditing and policy in about twenty lines, and it works for any tool the agent can reach.

#### Where your middleware lands

Pass `middleware=[...]` and yours is spliced in after the built-in core stack and before the tail:

```
FilesystemMiddleware          <- built-in core
SubAgentMiddleware
SummarizationMiddleware
PatchToolCallsMiddleware
Extra                         <- yours lands here
AnthropicPromptCachingMiddleware   <- tail
```

One trick worth knowing: middleware is identified by its `name`, and reusing an existing name *replaces* that entry in place rather than appending. So `name = "SummarizationMiddleware"` swaps out the built-in compaction and keeps its position in the stack.

Position matters more than it looks. In the lab, the audit middleware reports that `delete` is still in the tool list even though the profile excludes it — because exclusion is enforced by a middleware sitting closer to the model than ours. We're seeing the request before it's filtered. Where you sit in the stack decides what you see.
{: .notice--warning }

### 14.2 Harness settings

A `HarnessProfile` is a set of defaults attached to a **model** rather than to one agent. Register it once and every `create_deep_agent()` using that model picks it up.

```python
from deepagents import HarnessProfile, GeneralPurposeSubagentProfile, register_harness_profile

register_harness_profile("ollama", HarnessProfile(
    system_prompt_suffix=HOUSE_RULES,
    excluded_tools=frozenset({"execute", "delete"}),
    tool_description_overrides={
        "grep": "Search file contents for LITERAL text (not a regex).",
    },
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
))
```

Those four cover most of what you'll want: text appended to the system prompt, tools withheld from the model, better wording for a tool the model keeps misusing, and turning off the catch-all sub-agent. Three more exist — `base_system_prompt`, plus `extra_middleware` and `excluded_middleware` for adding and removing stack entries.

Prompts assemble as **`USER → BASE → SUFFIX`**: yours first, then the profile's. That's how the built-in Sonnet profile has been appending `<use_parallel_tool_calls>` to your prompts this whole time without you asking.

#### Which key do you register under?

This is the part that will waste your afternoon.

Pass a model *string* like `"anthropic:claude-sonnet-4-6"` and that string is the key. Pass a pre-built instance — which is what `build_local_model()` does — and deepagents derives the key from the model, falling back to the provider name.

For `ChatOllama` running `qwen3:8b`, the provider is `ollama`. So registering under `"ollama"` covers every local model in this tutorial.

There's a trap in that sentence, and I walked into it. The provider is a property of the *client*, not of the model — so it changes when you switch the front door. Run this same lab with `DEEP_AGENT_BACKEND=ollabridge` and you get a `ChatOpenAI` whose provider is `openai`, because the gateway speaks the OpenAI API. Same weights, same GPU, different key. A profile filed under `"ollama"` matches nothing, and the tool exclusions and prompt suffix simply stop applying — the agent runs on happily with the general-purpose sub-agent back and `delete` in its toolbox. So the lab picks its key from the active backend:

```python
backend = os.environ.get("DEEP_AGENT_BACKEND", "ollama").strip().lower()
key = "openai" if backend == "ollabridge" else "ollama"
```

Get it wrong and deepagents tells you rather than failing silently:

```
No harness profile matched pre-built model ... (provider='scripted');
using defaults. If you registered a profile for this model, ensure the key
matches the model's resolved provider and identifier.
```

I hit that exact line writing this lab. Read it as "your profile is not being applied", not as noise.
{: .notice--info }

#### Two guard rails you'll be glad of

You cannot strip the scaffolding the harness depends on, and it tells you the moment you declare the profile rather than later at run time:

```
ValueError: HarnessProfile.excluded_middleware is invalid:
  - required scaffolding cannot be excluded: FilesystemMiddleware
```

And an exclusion that matches nothing is an error too, not a shrug — a typo in a middleware name won't quietly do nothing. After a whole post about silent failures, it's nice to see a library refusing to let you *think* you configured something.

### 14.3 Which one should you reach for?

Two examples to calibrate on.

*"Nobody should be able to run shell commands through this model."* That's a harness setting. `excluded_tools={"execute"}` and it's gone everywhere, for every agent, without anyone remembering to add middleware.

*"I want to know which tools are slow, and block deletes outside `/tmp`."* That's middleware. It's per-call, it needs the arguments, and the answer depends on runtime state.

When you want both, use both — which is exactly what `make advanced` does.

## 14. Wrapping Up

Here's what we did.

We started with the six-line loop that every agent framework is built on, and saw exactly where it falls over: everything it knows lives in one list, and that list has an edge.

Then we gave it a desk — a plan, some files, and helpers with their own fresh conversations. The same task that had confused the plain loop finished in fourteen clear messages. And we watched it pick its own way through the toolbox: plan, look around, narrow, read, delegate, write.

And we did it on a laptop GPU, which turned out to be the best part. On a cloud model the context problem is theoretical. On 12 GB you can watch it break, fix it, and watch the fix work, in about ten minutes, for free.

Next in this series we go one layer down and serve the models ourselves with vLLM.

**Congratulations!** You've got a deep agent that plans, delegates and remembers — and it's running on your desk. Happy coding!
