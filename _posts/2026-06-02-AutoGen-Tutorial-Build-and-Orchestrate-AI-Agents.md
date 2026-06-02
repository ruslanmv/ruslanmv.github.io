---
title: "AutoGen Tutorial: Build a Simple Multi-Agent Email Triage System in Python"
excerpt: "Create a collaborative multi-agent workflow with Microsoft AutoGen."
description: "A hands-on Microsoft AutoGen tutorial in Python — build an assistant agent, add a tool, and orchestrate a multi-agent email-triage team with complete runnable scripts and clear roles."
last_modified_at: 2026-06-02
header:
  image: "/assets/images/posts/2026-06-02-AutoGen-Tutorial-Build-and-Orchestrate-AI-Agents/hero.jpg"
  teaser: "/assets/images/posts/2026-06-02-AutoGen-Tutorial-Build-and-Orchestrate-AI-Agents/hero.jpg"
  caption: "Photo via Unsplash"
categories:
  - AI
  - Agents
tags:
  - AutoGen
  - Multi-Agent
  - AI Agents
  - Python
  - Orchestration
toc: true
toc_label: "Contents"
---

Sometimes one AI assistant is enough. But some tasks become easier when different agents have different jobs.

For example, in email security, one agent can inspect the message, another can review the decision, and a final agent can decide what action to take. This is the idea behind **AutoGen**: you create small agents with clear roles, then let them collaborate.

In this tutorial, we will build a simple email-triage workflow with AutoGen. We will start with one agent, then add a tool, and finally create a small multi-agent team.

> This pairs with my [Agentic AI Tutorial](https://github.com/ruslanmv/agentic-ai-tutorial) ([live version](https://ruslanmv.com/agentic-ai-tutorial/)), which solves the same email-triage problem with role-based teams.

## What you'll build

By the end, you will have a small AutoGen workflow where one agent analyzes an email, one tool checks sender reputation, and a small team of agents agrees on a final action such as **quarantine**, **inbox**, or **manual review**.

## Prerequisites

- Python 3.10+
- An OpenAI API key (`export OPENAI_API_KEY=sk-...`)

## 1. Install AutoGen

```bash
python -m pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

## 2. A quick note on async

AutoGen's modern Python API is **asynchronous**. That means we define an `async` function, call the agent with `await`, and start the program with `asyncio.run(...)`. This may look slightly different from normal beginner Python, but the structure is simple once you see it — and every example below follows the same shape.

## 3. Your first agent

The classic pattern is an **AssistantAgent** (the brain) connected to a model client. Here's a complete, runnable script — it checks the API key, keeps the imports together, and uses a clear `main()`.

```python
import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY before running this script.")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

    analyst = AssistantAgent(
        name="security_analyst",
        model_client=model_client,
        system_message=(
            "You are a security analyst. "
            "Classify emails as phishing, spam, or legitimate. "
            "Explain the evidence briefly."
        ),
    )

    email = "Verify your account at http://login-verify.example"
    result = await analyst.run(task=f"Classify this email:\n\n{email}")
    print(result.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
```

**Expected output (abridged):**

```text
This is phishing. It creates urgency ("verify your account") and links to a
look-alike verification domain rather than the official site. Do not click.
```

## 4. Give an agent a tool

A **tool** gives the agent a way to check something *outside* the language model. The model can read the email, but it does not automatically know whether a sender exists in your internal fraud database. A tool can provide that evidence — and returning a short **reason** makes the agent's decisions auditable.

```python
def check_sender_reputation(email_address: str) -> str:
    """Check whether an email sender is trusted or suspicious."""
    known_bad_senders = {
        "billing@paypa1-support.com",
        "security-alert@fake-bank-login.com",
    }
    email_address = email_address.lower()
    if email_address in known_bad_senders:
        return "verdict=suspicious score=5 reason=sender appears in the known bad sender list"
    return "verdict=trusted score=82 reason=sender is not in the known bad sender list"
```

Now run an agent that actually uses the tool:

```python
import os, asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def run_with_tool():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])

    analyst = AssistantAgent(
        name="analyst",
        model_client=model_client,
        tools=[check_sender_reputation],
        system_message=(
            "Use the available tools to investigate the sender. "
            "Then classify the email as phishing, spam, or legitimate."
        ),
    )

    task = """
Classify this email:

From: billing@paypa1-support.com
Subject: Your account is locked

Please verify your account immediately.
"""
    result = await analyst.run(task=task)
    print(result.messages[-1].content)

asyncio.run(run_with_tool())
```

The agent will call `check_sender_reputation(...)`, read the result, and fold that evidence into its verdict.

## 5. Orchestrate a multi-agent team

Now we can split the work across three agents. The **triage** agent makes the first decision. The **reviewer** checks whether the decision makes sense. The **router** converts the approved decision into an action.

This is useful because each agent has a small job, and small roles are easier to debug than one large prompt that tries to do everything.

![Three agents collaborating, each with one clear role]({{ '/assets/images/posts/2026-06-02-AutoGen-Tutorial-Build-and-Orchestrate-AI-Agents/inline.jpg' | relative_url }})

A team runs the agents in a round-robin until a **stop condition** is met. Note the stop condition covers *all three* possible actions — `quarantine`, `inbox`, and `review` — so it always matches whatever the router decides.

```python
import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])

    triage = AssistantAgent(
        name="triage",
        model_client=model_client,
        system_message=(
            "You inspect the email first. "
            "Propose one label: phishing, spam, or legitimate. "
            "Give one short reason."
        ),
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=model_client,
        system_message=(
            "You review the triage decision. "
            "If you agree, say APPROVED and repeat the label. "
            "If you disagree, explain what should change."
        ),
    )

    router = AssistantAgent(
        name="router",
        model_client=model_client,
        system_message=(
            "You decide the final action after approval. "
            "Use only one of these actions: quarantine, inbox, or review."
        ),
    )

    stop = (
        TextMentionTermination("quarantine")
        | TextMentionTermination("inbox")
        | TextMentionTermination("review")
    )

    team = RoundRobinGroupChat(participants=[triage, reviewer, router], termination_condition=stop)

    email = """
From: billing@paypa1-support.com
Subject: Your account is locked

Verify your account now or access will be suspended.
"""
    result = await team.run(task=f"Classify this email:\n\n{email}")
    for message in result.messages:
        print(f"{message.source}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Expected flow:**

```text
triage:   Likely phishing — urgency + look-alike sender.
reviewer: Agree, impersonation risk is high. APPROVED: phishing
router:   Action: quarantine
```

Each agent stays focused on one job, which makes the system easier to debug, evaluate, and improve than a single mega-prompt.

## When should you use AutoGen?

You do not need AutoGen for every AI feature. If your task is a simple summary, rewrite, or classification, one model call may be enough.

AutoGen is more useful when the task benefits from **collaboration**. For example, one agent can draft, another can review, and another can decide the next step. This pattern helps with research workflows, code review, security triage, planning, and tasks where you want separate roles instead of one large prompt.

## Common design mistakes

A common mistake is **giving every agent the same job**. If all agents have similar prompts, the team becomes noisy and expensive. Each agent should have a clear responsibility.

Another mistake is **forgetting the stop condition**. Without a termination rule, a group chat can continue longer than expected and waste tokens.

A third mistake is **creating too many agents too early**. Start with one agent. Add a second only when there is a clear reason — such as review, critique, routing, or tool use.

## Common errors

- **`ModuleNotFoundError: autogen_agentchat`** — install the new packages (`autogen-agentchat`, `autogen-ext[openai]`); the older `pyautogen` API differs.
- **Nothing happens / no `await`** — the modern API is async; run inside `asyncio.run(...)`.
- **The team never stops** — always set a `termination_condition` that matches the words your agents actually produce.

## FAQ

**AutoGen vs LangChain/CrewAI?**
AutoGen excels at *conversational* multi-agent collaboration; LangChain/LangGraph give fine-grained control of a single agent's loop; CrewAI models role-based teams. All three are compared in the [Agentic AI Tutorial](https://github.com/ruslanmv/agentic-ai-tutorial).

**Can agents run code safely?**
AutoGen can run code through executors, including sandboxed options. For real projects, do not let generated code run directly on your machine without restrictions. Use a sandbox, limit permissions, and avoid exposing secrets inside the execution environment.

## Conclusion

You have now seen the basic AutoGen pattern: create an agent, give it a model, optionally give it tools, and then connect multiple agents into a team.

The main lesson is not that every project needs many agents. It's that complex workflows become easier to manage when each agent has one clear job. **Start simple, test each role, and only add more agents when they make the system easier to understand.**

## Related tutorials

- [AI Agents guide hub](/ai-agents/)
- [Create a simple AI email triage agent with LangChain](/blog/Create-AI-Agents-with-LangChain-and-OpenAI)
- [Agent planning and reasoning with ReAct](/blog/Agent-Planning-and-Reasoning-with-ReAct)
- [Deploy AI agents to production with Docker](/blog/Deploy-AI-Agents-to-Production-with-Docker)
- [watsonx Orchestrate tutorial](/blog/hello-watsonx-orchestrate)
