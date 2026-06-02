---
title: "Create a Simple AI Email Triage Agent with LangChain, LangGraph, and OpenAI"
excerpt: "Build an autonomous ReAct agent that investigates emails with custom tools."
description: "Build a simple AI email triage agent with LangChain, LangGraph, and OpenAI — custom tools, the ReAct loop, a complete runnable script, and a structured-output workflow you can reuse."
last_modified_at: 2026-06-02
header:
  image: "/assets/images/posts/2026-06-02-Create-AI-Agents-with-LangChain-and-OpenAI/hero.jpg"
  teaser: "/assets/images/posts/2026-06-02-Create-AI-Agents-with-LangChain-and-OpenAI/hero.jpg"
  caption: "Photo via Unsplash"
categories:
  - AI
  - Agents
tags:
  - LangChain
  - LangGraph
  - OpenAI
  - AI Agents
  - Python
toc: true
toc_label: "Contents"
---

Imagine you receive an email saying your account is locked, with a link to "verify now." A normal chatbot can only read the message and *guess* whether it looks suspicious. An **agent** can do more: it can check the sender, scan the URL, gather evidence, and only then make a decision.

That is the difference this tutorial teaches. We'll build a small **email triage agent** that uses Python tools to investigate an email before returning a verdict — with **LangChain**, **LangGraph**, and **OpenAI**, in well under 100 lines.

> This mirrors my open-source [Agentic AI Tutorial](https://github.com/ruslanmv/agentic-ai-tutorial) ([live version](https://ruslanmv.com/agentic-ai-tutorial/)), where the same problem is solved with LangChain, LangGraph, and CrewAI so you can compare frameworks side by side.

## What you'll build

You will build a **ReAct agent** that receives an email, checks the sender reputation, scans suspicious URLs, and returns a final verdict. The verdict includes a category, a confidence score, and a short rationale that explains *why* the email was classified that way.

## Prerequisites

- Python 3.10+
- An OpenAI API key (`export OPENAI_API_KEY=sk-...`)
- Basic familiarity with Python functions and type hints

## 1. Install the libraries

```bash
python -m pip install -U langchain langchain-openai langgraph pydantic
```

## 2. Define your tools

A tool is just a Python function with a docstring and type hints. LangChain turns the docstring into the description the model uses to decide *when* to call it.

For this tutorial, the tools are simple Python functions. They do **not** call a real threat-intelligence API yet — that keeps the example easy to understand. In a production system, these same functions could call a sender-reputation database, a URL scanner, or an internal security API. Notice that each tool returns a short **reason**, which both helps the model and makes the agent's decisions auditable.

```python
from langchain_core.tools import tool

@tool
def check_sender_reputation(email_address: str) -> str:
    """Check whether an email sender has a trusted or suspicious reputation."""
    known_bad_senders = {"billing@paypa1-support.com", "ceo@your-company.co"}
    email_address = email_address.lower()
    if email_address in known_bad_senders:
        return "score=5 verdict=suspicious reason=sender appears in known bad sender list"
    return "score=82 verdict=trusted reason=sender is not in the known bad sender list"

@tool
def scan_url(url: str) -> str:
    """Check whether a URL contains common phishing indicators."""
    suspicious_terms = ["paypa1", "login-verify", "bit.ly", "secure-update"]
    url_lower = url.lower()
    for term in suspicious_terms:
        if term in url_lower:
            return f"verdict=phishing reason=URL contains suspicious term: {term}"
    return "verdict=clean reason=no suspicious terms found"
```

## 3. Understand ReAct in one minute

**ReAct** means **Rea**son and **Act**. The model does not only answer immediately — it first decides whether it needs a tool. If it calls a tool, it reads the result (an *observation*) and then decides what to do next. This makes agents useful for tasks where the answer depends on external checks, calculations, databases, APIs, or business rules.

## 4. Create the agent

Modern LangChain ships a prebuilt ReAct agent in `langgraph`. You give it a model and a list of tools, and it manages the reason → act → observe loop for you. Keep the prompt focused, and ask the model to summarize the **evidence** rather than its private chain of thought.

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[check_sender_reputation, scan_url],
    prompt=(
        "You are an email security assistant. "
        "Use the available tools to collect evidence before making a decision. "
        "Classify the email as phishing, spam, or legitimate. "
        "In the final answer, briefly summarize the evidence, not your private reasoning."
    ),
)
```

## 5. Run it

```python
email = """
From: billing@paypa1-support.com
Subject: Your account is locked

Verify now at http://login-verify.paypa1-support.com to restore access.
"""

result = agent.invoke({"messages": [("user", f"Investigate this email:\n{email}")]})
print(result["messages"][-1].content)
```

**Expected output (abridged):**

```text
I checked the sender (score=5, verdict=suspicious) and scanned the URL
(verdict=phishing — contains "login-verify"). The domain also impersonates
PayPal ("paypa1"). Classification: phishing — do not click the link.
```

Behind the scenes the agent ran two **Reason → Act → Observe** cycles: it decided to check the sender, observed a low score, decided to scan the URL, observed "phishing," then produced its verdict.

![AI agent reasoning over code]({{ '/assets/images/posts/2026-06-02-Create-AI-Agents-with-LangChain-and-OpenAI/inline.jpg' | relative_url }})

## 6. The complete script

Here is everything in one file you can copy, run, and adapt — `agent_email_triage.py`:

```python
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def check_sender_reputation(email_address: str) -> str:
    """Check whether an email sender has a trusted or suspicious reputation."""
    known_bad_senders = {"billing@paypa1-support.com", "ceo@your-company.co"}
    email_address = email_address.lower()
    if email_address in known_bad_senders:
        return "score=5 verdict=suspicious reason=sender appears in known bad sender list"
    return "score=82 verdict=trusted reason=sender is not in the known bad sender list"


@tool
def scan_url(url: str) -> str:
    """Check whether a URL contains common phishing indicators."""
    suspicious_terms = ["paypa1", "login-verify", "bit.ly", "secure-update"]
    url_lower = url.lower()
    for term in suspicious_terms:
        if term in url_lower:
            return f"verdict=phishing reason=URL contains suspicious term: {term}"
    return "verdict=clean reason=no suspicious terms found"


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY before running this script.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_react_agent(
        model=llm,
        tools=[check_sender_reputation, scan_url],
        prompt=(
            "You are an email security assistant. "
            "Use the tools to collect evidence before making a decision. "
            "Classify the email as phishing, spam, or legitimate. "
            "Briefly explain the evidence behind your final verdict."
        ),
    )

    email = """
From: billing@paypa1-support.com
Subject: Your account is locked

Verify now at http://login-verify.paypa1-support.com to restore access.
"""

    result = agent.invoke({"messages": [("user", f"Investigate this email:\n{email}")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
```

Run it:

```bash
export OPENAI_API_KEY="your_api_key_here"
python agent_email_triage.py
```

## 7. Turn evidence into structured output

Free-text answers are hard to use downstream. A better, realistic workflow is two steps: **let the agent gather evidence with tools, then pass that evidence to a structured model** that returns a typed object you can trust.

```python
from pydantic import BaseModel, Field

class Verdict(BaseModel):
    category: str = Field(description="One of: phishing, spam, legitimate")
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(description="Short explanation based on evidence")

structured_llm = llm.with_structured_output(Verdict)

# Step 1 — the agent investigates and summarizes the evidence
agent_result = agent.invoke({
    "messages": [("user", f"Investigate this email and summarize the evidence:\n{email}")]
})
evidence = agent_result["messages"][-1].content

# Step 2 — a structured model formats the final, machine-readable verdict
verdict = structured_llm.invoke(f"""
Convert this investigation result into a structured verdict.

Email:
{email}

Investigation evidence:
{evidence}
""")

print(verdict)
# category='phishing' confidence=0.97 rationale='Suspicious sender and look-alike verification URL.'
```

The agent collects evidence; the structured model formats the final answer. That separation makes the result easy to use later in APIs, databases, dashboards, or automated workflows.

## When should you use an agent?

You do not need an agent for every AI feature. If your task is a simple rewrite, summary, or classification with **no external checks**, a direct model call is usually simpler and cheaper.

Use an agent when the model needs to **decide which tool to call**, combine multiple observations, query external systems, or handle tasks that may require different steps each time — exactly like investigating an email.

## Common errors

If you see an **`AuthenticationError`**, the OpenAI API key is probably missing or invalid. Set it in your terminal before running the script.

If **`create_react_agent` cannot be imported**, upgrade LangGraph (`pip install -U langgraph`). The prebuilt ReAct agent lives in LangGraph, so an older version may not include it.

If the **agent never calls a tool**, improve the tool docstring. The model uses the function name, type hints, and docstring to decide when a tool is useful.

If you hit **rate limits**, use a smaller model during development, add retries, or reduce repeated test runs.

## FAQ

**Do I have to use OpenAI?**
No. Swap `ChatOpenAI` for any LangChain chat model (Anthropic, IBM watsonx, local models via Ollama) — the agent code stays the same.

**What's the difference between LangChain and LangGraph here?**
LangChain provides the model and tool abstractions; LangGraph provides the prebuilt agent loop (and lets you build explicit, auditable state machines when you need more control).

**Can I add my own tools?**
Yes. Any Python function can become a tool if it has clear type hints and a useful docstring. For example, you could add a tool that checks a database, calls a URL-scanning API, searches internal documentation, or calculates a risk score.

## Conclusion

You have now built a small but complete AI agent. It can inspect an email, call tools, collect evidence, and return a verdict. The same pattern works for many real applications: customer-support agents, research assistants, security tools, data analysts, and workflow automation.

The important lesson is simple: **keep the tools clear, keep the prompt focused, and use structured output when another system needs to read the result.**

## Related tutorials

- [AI Agents guide hub](/ai-agents/)
- [Agent planning and reasoning with ReAct](/blog/Agent-Planning-and-Reasoning-with-ReAct)
- [AutoGen tutorial: build and orchestrate agents](/blog/AutoGen-Tutorial-Build-and-Orchestrate-AI-Agents)
- [Deploy AI agents to production with Docker](/blog/Deploy-AI-Agents-to-Production-with-Docker)
- [BeeAI Framework tutorial](/blog/BeeAI-Framework-Practical-Guide)
