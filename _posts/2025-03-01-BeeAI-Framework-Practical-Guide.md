---
title: "BeeAI Framework: Your Guide from Zero to Hero"
excerpt: "BeeAI Framework: Complete Guide from Beginner to Expert"

header:
  image: "./../assets/images/posts/2025-03-01-BeeAI-Framework-Practical-Guide/bee2.jpg"
  teaser: "./../assets/images/posts/2025-03-01-BeeAI-Framework-Practical-Guide/bee2.jpg"
  caption: "Agents interact, learn, and evolve together, making multi-agent systems inherently dynamic and adaptive. — Michael Wooldridge"
  
---

Welcome to the **BeeAI Framework tutorial**! This comprehensive guide is meticulously crafted to take you from a complete beginner to a proficient developer, leveraging the powerful capabilities of the BeeAI Framework. Throughout this guide, you'll master key concepts and practical applications that will enable you to build intelligent, context-aware AI applications.

Initially, you will learn foundational concepts like creating and rendering **Prompt Templates** to dynamically generate prompts tailored for specific interactions. Following that, you'll explore **ChatModel Interaction**, discovering effective ways to interact with language models through message-based communications. You'll then delve into advanced techniques of **Memory Handling**, essential for managing conversation histories and maintaining contextual coherence in AI interactions.

Further, you'll gain expertise in enforcing **Structured Outputs** using robust Pydantic schemas, ensuring your AI's responses adhere to predefined formats, thus enhancing reliability and predictability. You will also understand how to utilize **System Prompts** to strategically guide the behavior of language models, optimizing their responses for your specific use cases.

The tutorial advances into sophisticated areas such as developing **ReAct Agents and Tools**, which empowers your AI agents with reasoning and actionable capabilities through seamless integration of external tools. Finally, you will master **Workflows**, effectively orchestrating multiple steps and complex agent interactions into streamlined, dynamic processes, including the sophisticated management of multi-agent systems.

Below is a comprehensive table of contents for easy navigation through your journey with the **BeeAI Framework**.

## Table of Contents

- [BeeAI Framework Basics](#beeai-framework-basics)
  - [1. Prompt Templates](#1-prompt-templates)
    - [Example: RAG Prompt Template](#example-rag-prompt-template)
  - [2. More Complex Templates](#2-more-complex-templates)
    - [Example: Template with a List of Search Results](#example-template-with-a-list-of-search-results)
  - [3. The ChatModel](#3-the-chatmodel)
    - [Example: Creating a User Message](#example-creating-a-user-message)
    - [Example: Sending a Message to the ChatModel](#example-sending-a-message-to-the-chatmodel)
  - [4. Memory Handling](#4-memory-handling)
    - [Example: Storing and Retrieving Conversation History](#example-storing-and-retrieving-conversation-history)
  - [5. Combining Templates and Messages](#5-combining-templates-and-messages)
    - [Example: Rendering a Template and Sending as a Message](#example-rendering-a-template-and-sending-as-a-message)
  - [6. Structured Outputs](#6-structured-outputs)
    - [Example: Enforcing a Specific Output Format](#example-enforcing-a-specific-output-format)
  - [7. System Prompts](#7-system-prompts)
    - [Example: Using a System Message](#example-using-a-system-message)

- [BeeAI ReAct Agents](#beeai-react-agents)
  - [1. Basic ReAct Agent](#1-basic-react-agent)
    - [Example: Setting Up a Basic ReAct Agent](#example-setting-up-a-basic-react-agent)
  - [2. Using Tools with the Agent](#2-using-tools-with-the-agent)
    - [Example: Using a Built-In Weather Tool](#example-using-a-built-in-weather-tool)
  - [3. Imported Tools](#3-imported-tools)
    - [Example: Long-Form Integration with Wikipedia](#example-long-form-integration-with-wikipedia)
    - [Example: Shorter Form Using the `@tool` Decorator](#example-shorter-form-using-the-tool-decorator)

- [BeeAI Workflows](#beeai-workflows)
  - [Overview](#overview)
  - [Core Concepts](#core-concepts)
    - [State](#state)
    - [Steps](#steps)
    - [Transitions](#transitions)
  - [Basic Usage](#basic-usage)
    - [Simple Workflow](#simple-workflow)
    - [Multi-Step Workflow](#multi-step-workflow)
  - [Advanced Features](#advanced-features)
    - [Workflow Nesting](#workflow-nesting)
    - [Multi-Agent Workflows: Orchestration with BeeAI](#multi-agent-workflows-orchestration-with-beeai)
      - [Orchestration with Watsonx.ai Backend](#orchestration-with-watsonxai-backend)
    - [Memory in Workflows](#memory-in-workflows)

- [Backend](#backend)
  - [Overview](#overview-1)
  - [Supported Providers](#supported-providers)
  - [Backend Initialization](#backend-initialization)
  - [Chat Model](#chat-model)
    - [Chat Model Configuration](#chat-model-configuration)
    - [Text Generation](#text-generation)
    - [Streaming Responses](#streaming-responses)
    - [Structured Generation](#structured-generation)
    - [Tool Calling](#tool-calling)

- [Embedding Model](#embedding-model)
  - [Embedding Model Initialization](#embedding-model-initialization)
  - [Embedding Model Usage](#embedding-model-usage)



## BeeAI Framework Basics

Dive into the foundational concepts of the BeeAI Framework, progressively building your knowledge and practical skills to confidently create intelligent, context-aware applications.

I will present some examples to  demonstrate the fundamental usage patterns of BeeAI in Python. They progressively increase in complexity, providing a well-rounded overview of the framework.

<img src="./../assets/images/posts/2025-03-01-BeeAI-Framework-Practical-Guide/docs_logo.svg" alt="docs_logo" style="zoom:50%;" />



## Setup Environment

This section outlines the steps to set up your environment for running BeeAI Framework Python code examples on Windows and Ubuntu 22.04.

### Prerequisites

- **Python 3.12+**:  Required for BeeAI Framework.
- **Anaconda or Miniconda (Recommended)**: For easier environment management.

### Step-by-step Setup

Follow the instructions for your operating system.

#### Windows

1. Install Python 3.12+:

   - Download from [python.org](https://www.python.org/downloads/windows/).
   - **Important:** Check "Add Python 3.12 to PATH" during installation.

2. Install Anaconda/Miniconda:

   - Download from [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
   - Run the installer with default settings.

3. **Open Anaconda Prompt**: Search in Start Menu.

4. Create Virtual Environment:

   ```bash
   python -m venv venv
   ```

5. Activate Virtual Environment:

   ```bash
   venv\Scripts\activate
   ```

6. Install BeeAI Framework & Dependencies:

   ```bash
   pip install beeai-framework
   # Install additional dependencies if needed by examples (e.g., visualization libraries)
   # pip install pandas networkx matplotlib plotly scikit-learn
   ```

7. Install Ollama:

   - Download the Windows installer from [ollama.com](https://ollama.com/download).
   - Run the installer.

8. Start Ollama Server: Open a new Anaconda Prompt and run:

   ```bash
   ollama serve &
   ```

9. Download Ollama Model:

   ```bash
   ollama pull granite3.1-dense:8b
   ```

10. Watsonx.ai Credentials (If using Watsonx):

    - Obtain Project ID, API Key, and API Endpoint URL from your Watsonx.ai service.

    - Set environment variables in Anaconda Prompt (or system-wide):

      ```bash
      set WATSONX_PROJECT_ID=YOUR_WATSONX_PROJECT_ID
      set WATSONX_API_KEY=YOUR_WATSONX_API_KEY
      set WATSONX_API_URL=YOUR_WATSONX_API_ENDPOINT_URL
      ```

#### Ubuntu 22.04

1. Install Python 3.12+:

   ```bash
   sudo apt update
   sudo apt install python3.12 python3.12-venv
   ```

2. Install Anaconda/Miniconda:

   - Download the Linux installer from [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
   - Run the `.sh` installer in your terminal.

3. **Activate Anaconda**: Close and reopen terminal or `source ~/.bashrc` / `source ~/.zshrc`.

4. Create Virtual Environment:

   ```bash
   python3.12 -m venv venv
   ```

5. Activate Virtual Environment:

   ```bash
   source venv/bin/activate
   ```

6. Install BeeAI Framework & Dependencies:

   ```bash
   pip install beeai-framework
   # Install additional dependencies if needed by examples
   # pip install pandas networkx matplotlib plotly scikit-learn
   ```

7. Install Ollama:

   

   ```bash
   curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
   ```

8. Start Ollama Server: In a new terminal, run:

   

   ```bash
   ollama serve &
   ```

9. Download Ollama Model:

   

   ```bash
   ollama pull granite3.1-dense:8b
   ```

10. Watsonx.ai Credentials (If using Watsonx):

    - Obtain Watsonx.ai credentials.

    - Set environment variables in your terminal (or shell config file):

      ```bash
      export WATSONX_PROJECT_ID=YOUR_WATSONX_PROJECT_ID
      export WATSONX_API_KEY=YOUR_WATSONX_API_KEY
      export WATSONX_API_URL=YOUR_WATSONX_API_ENDPOINT_URL
      ```

**Notes:**

- **Virtual Environments**:  Always activate your virtual environment.
- **Ollama Server**: Keep Ollama server running in background.
- **Watsonx Credentials**: Securely manage your Watsonx API keys using environment variables.
- **Troubleshooting**: Double-check each step if you encounter issues. Refer to BeeAI documentation for further assistance.

Your environment is now configured to run **BeeAI Framework examples.**

### 1. Prompt Templates

One of the core constructs in the BeeAI framework is the `PromptTemplate`. It allows you to dynamically insert data into a prompt before sending it to a language model. BeeAI uses the Mustache templating language for prompt formatting.

#### Example: RAG Prompt Template

```python
from pydantic import BaseModel
from beeai_framework.utils.templates import PromptTemplate

# Define the structure of the input data that will be passed to the template.
class RAGTemplateInput(BaseModel):
    question: str
    context: str

# Define the prompt template.
rag_template: PromptTemplate = PromptTemplate(
    schema=RAGTemplateInput,
    template="""
Context: {{context}}
Question: {{question}}

Provide a concise answer based on the context. Avoid statements such as 'Based on the context' or 'According to the context' etc. """,
)

# Render the template using an instance of the input model.
prompt = rag_template.render(
    RAGTemplateInput(
        question="What is the capital of France?",
        context="France is a country in Europe. Its capital city is Paris, known for its culture and history.",
    )
)

# Print the rendered prompt.
print(prompt)
```

---

### 2. More Complex Templates

The `PromptTemplate` class also supports more complex structures. For example, you can iterate over a list of search results to build a prompt.

#### Example: Template with a List of Search Results

```python
from pydantic import BaseModel
from beeai_framework.utils.templates import PromptTemplate

# Individual search result schema.
class SearchResult(BaseModel):
    title: str
    url: str
    content: str

# Input specification for the template.
class SearchTemplateInput(BaseModel):
    question: str
    results: list[SearchResult]

# Define the template that iterates over the search results.
search_template: PromptTemplate = PromptTemplate(
    schema=SearchTemplateInput,
    template="""
Search results:
{{#results.0}}
{{#results}}
Title: {{title}}
Url: {{url}}
Content: {{content}}
{{/results}}
{{/results.0}}

Question: {{question}}
Provide a concise answer based on the search results provided.""",
)

# Render the template with sample data.
prompt = search_template.render(
    SearchTemplateInput(
        question="What is the capital of France?",
        results=[
            SearchResult(
                title="France",
                url="[https://en.wikipedia.org/wiki/France](https://en.wikipedia.org/wiki/France)",
                content="France is a country in Europe. Its capital city is Paris, known for its culture and history.",
            )
        ],
    )
)

# Print the rendered prompt.
print(prompt)
```

---

### 3. The ChatModel

Once you have your prompt templates set up, you can begin interacting with a language model. BeeAI supports various LLMs through the `ChatModel` interface.

#### Example: Creating a User Message

```python
from beeai_framework.backend.message import UserMessage

# Create a user message to start a chat with the model.
user_message = UserMessage(content="Hello! Can you tell me what is the capital of France?")
```

#### Example: Sending a Message to the ChatModel

```python
from beeai_framework.backend.chat import ChatModel, ChatModelInput, ChatModelOutput

# Create a ChatModel instance that interfaces with Granite 3.1 (via Ollama).
model = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Send the user message and get the model's response.
output: ChatModelOutput = await model.create(ChatModelInput(messages=[user_message]))

# Print the model's response.
print(output.get_text_content())
```

---

### 4. Memory Handling

Memory is a convenient way to store the conversation history (a series of messages) that the model uses for context.

#### Example: Storing and Retrieving Conversation History

```python
from beeai_framework.backend.message import AssistantMessage
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Create an unconstrained memory instance.
memory = UnconstrainedMemory()

# Add a series of messages to the memory.
await memory.add_many(
    [
        user_message,
        AssistantMessage(content=output.get_text_content()),
        UserMessage(content="If you had to recommend one thing to do there, what would it be?"),
    ]
)

# Send the complete message history to the model.
output: ChatModelOutput = await model.create(ChatModelInput(messages=memory.messages))
print(output.get_text_content())
```

---

### 5. Combining Templates and Messages

You can render a prompt from a template and then send it as a message to the ChatModel.

#### Example: Rendering a Template and Sending as a Message

```python
# Some context that the model will use (e.g., from Wikipedia on Ireland).
context = """The geography of Ireland comprises relatively low-lying mountains surrounding a central plain, with several navigable rivers extending inland.
Its lush vegetation is a product of its mild but changeable climate which is free of extremes in temperature.
Much of Ireland was woodland until the end of the Middle Ages. Today, woodland makes up about 10% of the island,
compared with a European average of over 33%, with most of it being non-native conifer plantations.
The Irish climate is influenced by the Atlantic Ocean and thus very moderate, and winters are milder than expected for such a northerly area,
although summers are cooler than those in continental Europe. Rainfall and cloud cover are abundant.
"""

# Reuse the previously defined RAG template.
prompt = rag_template.render(RAGTemplateInput(question="How much of Ireland is forested?", context=context))

# Send the rendered prompt to the model.
output: ChatModelOutput = await model.create(ChatModelInput(messages=[UserMessage(content=prompt)]))
print(output.get_text_content())
```

---

### 6. Structured Outputs

Sometimes you want the LLM to generate output in a specific format. You can enforce this using structured outputs with a Pydantic schema.

#### Example: Enforcing a Specific Output Format

```python
from typing import Literal
from pydantic import Field
from beeai_framework.backend.chat import ChatModelStructureInput

# Define the output structure for a character.
class CharacterSchema(BaseModel):
    name: str = Field(description="The name of the character.")
    occupation: str = Field(description="The occupation of the character.")
    species: Literal["Human", "Insectoid", "Void-Serpent", "Synth", "Ethereal", "Liquid-Metal"] = Field(
        description="The race of the character."
    )
    back_story: str = Field(description="Brief backstory of this character.")

# Create a user message instructing the model to generate a character.
user_message = UserMessage(
    "Create a fantasy sci-fi character for my new game. This character will be the main protagonist, be creative."
)

# Request a structured response from the model.
response = await model.create_structure(ChatModelStructureInput(schema=CharacterSchema, messages=[user_message]))
print(response.object)
```

---

### 7. System Prompts

System messages can guide the overall behavior of the language model.

#### Example: Using a System Message

```python
from beeai_framework.backend.message import SystemMessage

# Create a system message that instructs the LLM to respond like a pirate.
system_message = SystemMessage(content="You are pirate. You always respond using pirate slang.")

# Create a new user message.
user_message = UserMessage(content="What is a baby hedgehog called?")

# Send both messages to the model.
output: ChatModelOutput = await model.create(ChatModelInput(messages=[system_message, user_message]))
print(output.get_text_content())
```

---

## BeeAI ReAct Agents

The BeeAI ReAct agent implements the “Reasoning and Acting” pattern, separating the process into distinct steps. This section shows how to build an agent that uses its own memory for reasoning and even integrates tools for added functionality.

### 1. Basic ReAct Agent

#### Example: Setting Up a Basic ReAct Agent

```python
from typing import Any
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput, BeeRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.emitter.types import EmitterOptions
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Construct the BeeAgent without external tools.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[], memory=UnconstrainedMemory()))

# Define a function to process agent events.
async def process_agent_events(event_data: dict[str, Any], event_meta: EventMeta) -> None:
    if event_meta.name == "error":
        print("Agent 🤖 : ", event_data["error"])
    elif event_meta.name == "retry":
        print("Agent 🤖 : ", "retrying the action...")
    elif event_meta.name == "update":
        print(f"Agent({event_data['update']['key']}) 🤖 : ", event_data["update"]["parsedValue"])

# Attach an observer to log events.
async def observer(emitter: Emitter) -> None:
    emitter.on("*.*", process_agent_events, EmitterOptions(match_nested=True))

# Run the agent with a sample prompt.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What chemical elements make up a water molecule?")
).observe(observer)
```

---

### 2. Using Tools with the Agent

Agents can be extended with tools so that they can perform external actions, like fetching weather data.

#### Example: Using a Built-In Weather Tool

```python
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create an agent that includes the OpenMeteoTool.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[OpenMeteoTool()], memory=UnconstrainedMemory()))

# Run the agent with a prompt about the weather.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What's the current weather in London?")
).observe(observer)
```

---

### 3. Imported Tools

You can also import tools from other libraries. Below are two examples that show how to integrate Wikipedia search via LangChain.

#### Example: Long-Form Integration with Wikipedia

```python
from typing import Any
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.tools import Tool
from beeai_framework.tools.tool import StringToolOutput

# Define the input schema for the Wikipedia tool.
class LangChainWikipediaToolInput(BaseModel):
    query: str = Field(description="The topic or question to search for on Wikipedia.")

# Adapter class to integrate LangChain's Wikipedia tool.
class LangChainWikipediaTool(Tool):
    name = "Wikipedia"
    description = "Search factual and historical information from Wikipedia about given topics."
    input_schema = LangChainWikipediaToolInput

    def __init__(self) -> None:
        super().__init__()
        wikipedia = WikipediaAPIWrapper()
        self.wikipedia = WikipediaQueryRun(api_wrapper=wikipedia)

    def _run(self, input: LangChainWikipediaToolInput, _: Any | None = None) -> None:
        query = input.query
        try:
            result = self.wikipedia.run(query)
            return StringToolOutput(result=result)
        except Exception as e:
            print(f"Wikipedia search error: {e!s}")
            return f"Error searching Wikipedia: {e!s}"

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create an agent that uses the custom Wikipedia tool.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[LangChainWikipediaTool()], memory=UnconstrainedMemory()))

# Run the agent with a query about the European Commission.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="Who is the current president of the European Commission?")
).observe(observer)
```

#### Example: Shorter Form Using the `@tool` Decorator

```python
from langchain_community.tools import WikipediaQueryRun  # noqa: F811
from langchain_community.utilities import WikipediaAPIWrapper  # noqa: F811
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.tools import Tool, tool

# Define a tool using the decorator.
@tool
def langchain_wikipedia_tool(expression: str) -> str:
    """
    Search factual and historical information, including biography, history, politics, geography, society, culture,
    science, technology, people, animal species, mathematics, and other subjects.
    
    Args:
        expression: The topic or question to search for on Wikipedia.
    
    Returns:
        The information found via searching Wikipedia.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return StringToolOutput(wikipedia.run(expression))

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create an agent that uses the decorated Wikipedia tool.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[langchain_wikipedia_tool], memory=UnconstrainedMemory()))

# Run the agent with a query about the longest living vertebrate.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What is the longest living vertebrate?")
).observe(observer)
```

---

## BeeAI Workflows 

Workflows allow you to combine what you’ve learned into a coherent multi-step process. A workflow is defined by a state (a Pydantic model) and steps (Python functions) that update the state and determine the next step. Workflows in BeeAI provide a flexible and extensible component for managing and executing structured sequences of tasks, especially useful for orchestration of complex agent behaviors and multi-agent systems.

### Overview

Workflows provide a flexible and extensible component for managing and executing structured sequences of tasks. They are particularly useful for:

- **Dynamic Execution**: Steps can direct the flow based on state or results
- **Validation**: Define schemas for data consistency and type safety
- **Modularity**: Steps can be standalone or invoke nested workflows
-  **Observability**: Emit events during execution to track progress or handle errors

---

### Core Concepts

#### State

State is the central data structure in a workflow. It's a Pydantic model that:

- Holds the data passed between steps
- Provides type validation and safety
- Persists throughout the workflow execution

#### Steps

Steps are the building blocks of a workflow. Each step is a function that:

- Takes the current state as input
- Can modify the state
- Returns the name of the next step to execute or a special reserved value

#### Transitions

Transitions determine the flow of execution between steps. Each step returns either:

- The name of the next step to execute
- `Workflow.NEXT` - proceed to the next step in order
- `Workflow.SELF` - repeat the current step
- `Workflow.END` - end the workflow execution

---

### Basic Usage

#### Simple Workflow

The example below demonstrates a minimal workflow that processes steps in sequence. This pattern is useful for straightforward, linear processes where each step builds on the previous one.

```python
import asyncio
import sys
import traceback

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow


async def main() -> None:
    # State
    class State(BaseModel):
        input: str

    workflow = Workflow(State)
    workflow.add_step("first", lambda state: print("Running first step!"))
    workflow.add_step("second", lambda state: print("Running second step!"))
    workflow.add_step("third", lambda state: print("Running third step!"))

    await workflow.run(State(input="Hello"))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

#### Multi-Step Workflow

This advanced example showcases a workflow that implements multiplication through repeated addition—demonstrating control flow, state manipulation, nesting, and conditional logic.

```python
import asyncio
import sys
import traceback
from typing import Literal, TypeAlias

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow, WorkflowReservedStepName

WorkflowStep: TypeAlias = Literal["pre_process", "add_loop", "post_process"]


async def main() -> None:
    # State
    class State(BaseModel):
        x: int
        y: int
        abs_repetitions: int | None = None
        result: int | None = None

    def pre_process(state: State) -> WorkflowStep:
        print("pre_process")
        state.abs_repetitions = abs(state.y)
        return "add_loop"

    def add_loop(state: State) -> WorkflowStep | WorkflowReservedStepName:
        if state.abs_repetitions and state.abs_repetitions > 0:
            result = (state.result if state.result is not None else 0) + state.x
            abs_repetitions = (state.abs_repetitions if state.abs_repetitions is not None else 0) - 1
            print(f"add_loop: intermediate result {result}")
            state.abs_repetitions = abs_repetitions
            state.result = result
            return Workflow.SELF
        else:
            return "post_process"

    def post_process(state: State) -> WorkflowReservedStepName:
        print("post_process")
        if state.y < 0:
            result = -(state.result if state.result is not None else 0)
            state.result = result
        return Workflow.END

    multiplication_workflow = Workflow[State, WorkflowStep](name="MultiplicationWorkflow", schema=State)
    multiplication_workflow.add_step("pre_process", pre_process)
    multiplication_workflow.add_step("add_loop", add_loop)
    multiplication_workflow.add_step("post_process", post_process)

    response = await multiplication_workflow.run(State(x=8, y=5))
    print(f"result: {response.state.result}")

    response = await multiplication_workflow.run(State(x=8, y=-5))
    print(f"result: {response.state.result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

This workflow demonstrates several powerful concepts:

- Implementing loops by returning `Workflow.SELF`
- Conditional transitions between steps
- Progressive state modification to accumulate results
- Sign handling through state transformation
- Type-safe step transitions using Literal types

---

### Advanced Features

#### Workflow Nesting

Workflow nesting allows complex behaviors to be encapsulated as reusable components, enabling hierarchical composition of workflows. This promotes modularity, reusability, and better organization of complex agent logic.

```python
import asyncio
import sys
import traceback
from typing import Literal, TypeAlias

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow, WorkflowReservedStepName

WorkflowStep: TypeAlias = Literal["pre_process", "add_loop", "post_process"]


async def main() -> None:
    # State
    class State(BaseModel):
        x: int
        y: int
        abs_repetitions: int | None = None
        result: int | None = None

    def pre_process(state: State) -> WorkflowStep:
        print("pre_process")
        state.abs_repetitions = abs(state.y)
        return "add_loop"

    def add_loop(state: State) -> WorkflowStep | WorkflowReservedStepName:
        if state.abs_repetitions and state.abs_repetitions > 0:
            result = (state.result if state.result is not None else 0) + state.x
            abs_repetitions = (state.abs_repetitions if state.abs_repetitions is not None else 0) - 1
            print(f"add_loop: intermediate result {result}")
            state.abs_repetitions = abs_repetitions
            state.result = result
            return Workflow.SELF
        else:
            return "post_process"

    def post_process(state: State) -> WorkflowReservedStepName:
        print("post_process")
        if state.y < 0:
            result = -(state.result if state.result is not None else 0)
            state.result = result
        return Workflow.END

    multiplication_workflow = Workflow[State, WorkflowStep](name="MultiplicationWorkflow", schema=State)
    multiplication_workflow.add_step("pre_process", pre_process)
    multiplication_workflow.add_step("add_loop", add_loop)
    multiplication_workflow.add_step("post_process", post_process)

    response = await multiplication_workflow.run(State(x=8, y=5))
    print(f"result: {response.state.result}")

    response = await multiplication_workflow.run(State(x=8, y=-5))
    print(f"result: {response.state.result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

#### Multi-Agent Workflows: Orchestration with BeeAI

The multi-agent workflow pattern enables the orchestration of specialized agents that collaborate to solve complex problems. Each agent focuses on a specific domain or capability, with results combined by a coordinator agent.  BeeAI Framework's workflow engine is perfectly suited for creating sophisticated multi-agent systems.

The following example demonstrates how to orchestrate a multi-agent system using BeeAI workflows with Ollama backend. We will create a "Smart assistant" workflow composed of three specialized agents: `WeatherForecaster`, `Researcher`, and `Solver`.

```python
import asyncio
import sys
import traceback

from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    workflow = AgentWorkflow(name="Smart assistant")
    workflow.add_agent(
        name="WeatherForecaster",
        instructions="You are a weather assistant. Use tools to provide weather information.",
        tools=[OpenMeteoTool()],
        llm=llm,
        execution=AgentExecutionConfig(max_iterations=3, total_max_retries=10, max_retries_per_step=3),
    )
    workflow.add_agent(
        name="Researcher",
        instructions="You are a researcher assistant. Use search tools to find information.",
        tools=[DuckDuckGoSearchTool()],
        llm=llm,
    )
    workflow.add_agent(
        name="Solver",
        instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
        llm=llm,
    )

    prompt = "What is the weather in New York?"
    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content=prompt))
    response = await workflow.run(messages=memory.messages)
    print(f"result (Ollama Backend): {response.state.final_answer}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

This pattern demonstrates:

- **Role specialization** through focused agent configuration. `WeatherForecaster` is designed specifically for weather-related queries, while `Researcher` is for general information retrieval.
- **Efficient tool distribution** to relevant specialists. The `WeatherForecaster` agent is equipped with the `OpenMeteoTool`, and `Researcher` with `DuckDuckGoSearchTool`, ensuring each agent has the right tools for its job.
- **Parallel processing** of different aspects of a query.  Although not explicitly parallel in this example, the workflow structure is designed to easily support parallel execution of agents if needed.
- **Synthesis of multiple expert perspectives** into a cohesive response. The `Solver` agent acts as a coordinator, taking responses from other agents and synthesizing them into a final answer.
- **Declarative agent configuration** using the `AgentWorkflow` and `add_agent` methods, which simplifies the setup and management of complex agent systems.

**Orchestration with Watsonx.ai Backend**

To demonstrate the versatility of BeeAI workflows, let's adapt the multi-agent workflow example to use Watsonx.ai as the backend LLM provider. First, ensure you have configured the Watsonx provider as described in the Backend section. Then, modify the `ChatModel.from_name` call to use a Watsonx model:

```python
import asyncio
import sys
import traceback

from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow

async def main() -> None:
    # Initialize Watsonx ChatModel
    llm = ChatModel.from_name("watsonx:ibm/granite-3-8b-instruct") # Replace with your Watsonx model

    workflow = AgentWorkflow(name="Smart assistant (Watsonx)")
    workflow.add_agent(
        name="WeatherForecaster",
        instructions="You are a weather assistant.",
        tools=[OpenMeteoTool()],
        llm=llm,
        execution=AgentExecutionConfig(max_iterations=3, total_max_retries=10, max_retries_per_step=3),
    )
    workflow.add_agent(
        name="Researcher",
        instructions="You are a researcher assistant.",
        tools=[DuckDuckGoSearchTool()],
        llm=llm,
    )
    workflow.add_agent(
        name="Solver",
        instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
        llm=llm,
    )

    prompt = "What is the weather in London?"
    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content=prompt))
    response = await workflow.run(messages=memory.messages)
    print(f"result (Watsonx Backend): {response.state.final_answer}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

In this modified example, we simply changed the `ChatModel.from_name` call to `watsonx:ibm/granite-3-8b-instruct`.  Assuming you have correctly set up your Watsonx environment variables, this code will now orchestrate the same multi-agent workflow but powered by Watsonx.ai. This highlights the provider-agnostic nature of BeeAI workflows, allowing you to easily switch between different LLM backends without significant code changes.

#### Memory in Workflows

Integrating memory into workflows allows agents to maintain context across interactions, enabling conversational interfaces and stateful processing. This example demonstrates a simple conversational echo workflow with persistent memory.

```python
import asyncio
import sys
import traceback

from pydantic import BaseModel, InstanceOf

from beeai_framework.backend.message import AssistantMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.workflows.workflow import Workflow
from examples.helpers.io import ConsoleReader


async def main() -> None:
    # State with memory
    class State(BaseModel):
        memory: InstanceOf[UnconstrainedMemory]
        output: str | None = None

    async def echo(state: State) -> str:
        # Get the last message in memory
        last_message = state.memory.messages[-1]
        state.output = last_message.text[::-1]
        return Workflow.END

    reader = ConsoleReader()

    memory = UnconstrainedMemory()
    workflow = Workflow(State)
    workflow.add_step("echo", echo)

    for prompt in reader:
        # Add user message to memory
        await memory.add(UserMessage(content=prompt))
        # Run workflow with memory
        response = await workflow.run(State(memory=memory))
        # Add assistant response to memory
        await memory.add(AssistantMessage(content=response.state.output))

        reader.write("Assistant 🤖 : ", response.state.output)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

This pattern demonstrates:

- Integration of memory as a first-class citizen in workflow state
- Conversation loops that preserve context across interactions
- Bidirectional memory updating (reading recent messages, storing responses)
- Clean separation between the persistent memory and workflow-specific state

---

## ⚙️ Backend

## Overview

Backend is an umbrella module that encapsulates a unified way to work with the following functionalities:

- Chat Models via (ChatModel class)
- Embedding Models (coming soon)
- Audio Models (coming soon)
- Image Models (coming soon)

BeeAI framework's backend is designed with a provider-based architecture, allowing you to switch between different AI service providers while maintaining a consistent API.


---

## Supported providers



The table below lists supported providers, their dependencies, and required environment variables. Ensure these variables are properly configured before using each provider.

| Provider         | Chat Support | Dependency            | Required Environment Variables                               |
| ---------------- | ------------ | --------------------- | ------------------------------------------------------------ |
| Ollama           | ✅            | ollama-ai-provider    | `OLLAMA_CHAT_MODEL`,`OLLAMA_BASE_URL`                        |
| OpenAI           | ✅            | openai                | `OPENAI_CHAT_MODEL`,`OPENAI_API_BASE`,`OPENAI_API_KEY`,`OPENAI_ORGANIZATION` |
| Watsonx          | ✅            | @ibm-cloud/watsonx-ai | `WATSONX_CHAT_MODEL`,`WATSONX_API_KEY`,`WATSONX_PROJECT_ID`,`WATSONX_SPACE_ID`,`WATSONX_VERSION`,`WATSONX_REGION` |
| Groq             | ✅            |                       | `GROQ_CHAT_MODEL`,`GROQ_API_KEY`                             |
| Amazon Bedrock   | ✅            | boto3                 | `AWS_CHAT_MODEL`,`AWS_ACCESS_KEY_ID`,`AWS_SECRET_ACCESS_KEY`,`AWS_REGION_NAME` |
| Google Vertex AI | ✅            |                       | `VERTEXAI_CHAT_MODEL`,`VERTEXAI_PROJECT`,`GOOGLE_APPLICATION_CREDENTIALS`,`GOOGLE_APPLICATION_CREDENTIALS_JSON`,`GOOGLE_CREDENTIALS` |
| Azure OpenAI     | ❌            | Coming soon!          | `AZURE_OPENAI_CHAT_MODEL`,`AZURE_OPENAI_API_KEY`,`AZURE_OPENAI_API_ENDPOINT`,`AZURE_OPENAI_API_RESOURCE`,`AZURE_OPENAI_API_VERSION` |
| Anthropic        | ✅            |                       | `ANTHROPIC_CHAT_MODEL`,`ANTHROPIC_API_KEY`                   |
| xAI              | ✅            |                       | `XAI_CHAT_MODEL`,`XAI_API_KEY`                               |




> If you don't see your provider raise an issue [here](https://github.com/i-am-bee/beeai-framework/discussions).

---

### Backend initialization

The Backend class serves as a central entry point to access models from your chosen provider.

**Watsonx Initialization**

To use Watsonx with BeeAI framework, you need to install the Watsonx adapter and set up your environment variables.

**Installation:**

```bash
pip install beeai-framework[watsonx]
```

**Environment Variables:**

Set the following environment variables. You can obtain these from your IBM Cloud account and Watsonx service instance.

- `WATSONX_API_KEY`: Your Watsonx API key.
- `WATSONX_PROJECT_ID`: Your Watsonx project ID.
- `WATSONX_REGION`: The region where your Watsonx service is deployed (e.g., `us-south`).
- `WATSONX_CHAT_MODEL`: The specific Watsonx chat model you want to use (e.g., `ibm/granite-3-8b-instruct`).

**Example Code:**

Here's how to initialize and use Watsonx ChatModel:

```python
import asyncio
import json
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework import ToolMessage
from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import MessageToolResultContent, UserMessage
from beeai_framework.cancellation import AbortSignal
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

# Setting can be passed here during initiation or pre-configured via environment variables
llm = WatsonxChatModel(
    "ibm/granite-3-8b-instruct",
    # settings={
    #     "project_id": "WATSONX_PROJECT_ID",
    #     "api_key": "WATSONX_API_KEY",
    #     "api_base": "WATSONX_API_URL",
    # },
)


async def watsonx_from_name() -> None:
    watsonx_llm = ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        # {
        #     "project_id": "WATSONX_PROJECT_ID",
        #     "api_key": "WATSONX_API_KEY",
        #     "api_base": "WATSONX_API_URL",
        # },
    )
    user_message = UserMessage("what states are part of New England?")
    response = await watsonx_llm.create(messages=[user_message])
    print(response.get_text_content())


async def watsonx_sync() -> None:
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def watsonx_stream() -> None:
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create(messages=[user_message], stream=True)
    print(response.get_text_content())


async def watsonx_stream_abort() -> None:
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.create(messages=[user_message], stream=True, abort_signal=AbortSignal.timeout(0.5))

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


async def watson_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create_structure(schema=TestSchema, messages=[user_message])
    print(response.object)


async def watson_tool_calling() -> None:
    watsonx_llm = ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
    )
    user_message = UserMessage("What is the current weather in Boston?")
    weather_tool = OpenMeteoTool()
    response = await watsonx_llm.create(messages=[user_message], tools=[weather_tool])
    tool_call_msg = response.get_tool_calls()[0]
    print(tool_call_msg.model_dump())
    tool_response = await weather_tool.run(json.loads(tool_call_msg.args))
    tool_response_msg = ToolMessage(
        MessageToolResultContent(
            result=tool_response.get_text_content(), tool_name=tool_call_msg.tool_name, tool_call_id=tool_call_msg.id
        )
    )
    print(tool_response_msg.to_plain())
    final_response = await watsonx_llm.create(messages=[user_message, tool_response_msg], tools=[])
    print(final_response.get_text_content())


async def main() -> None:
    print("*" * 10, "watsonx_from_name")
    await watsonx_from_name()
    print("*" * 10, "watsonx_sync")
    await watsonx_sync()
    print("*" * 10, "watsonx_stream")
    await watsonx_stream()
    print("*" * 10, "watsonx_stream_abort")
    await watsonx_stream_abort()
    print("*" * 10, "watson_structure")
    await watson_structure()
    print("*" * 10, "watson_tool_calling")
    await watson_tool_calling()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```



---

## Chat model

The ChatModel class represents a Chat Large Language Model and provides methods for text generation, streaming responses, and more. You can initialize a chat model in multiple ways:

**Method 1: Using the generic factory method**

```python
from beeai_framework.backend.chat import ChatModel

ollama_chat_model = ChatModel.from_name("ollama:llama3.1")
```

**Method 2: Creating a specific provider model directly**

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel

ollama_chat_model = OllamaChatModel("llama3.1")
```

### Chat model configuration

You can configure various parameters for your chat model:

*Coming soon*

### Text generation

The most basic usage is to generate text responses:

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage

ollama_chat_model = OllamaChatModel("llama3.1")
response = await ollama_chat_model.create(
    messages=[UserMessage("what states are part of New England?")]
)

print(response.get_text_content())
```

> [!NOTE]
>
> Execution parameters (those passed to model.create({...})) are superior to ones defined via config.

### Streaming responses

For applications requiring real-time responses:

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage

llm = OllamaChatModel("llama3.1")
user_message = UserMessage("How many islands make up the country of Cape Verde?")
response = await llm.create(messages=[user_message], stream=True)
```

### Structured generation

Generate structured data according to a schema:

```python
import asyncio
import json
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework import UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError


async def main() -> None:
    model = ChatModel.from_name("ollama:llama3.1")

    class ProfileSchema(BaseModel):
        first_name: str = Field(..., min_length=1)
        last_name: str = Field(..., min_length=1)
        address: str
        age: int = Field(..., min_length=1)
        hobby: str

    class ErrorSchema(BaseModel):
        error: str

    class SchemUnion(ProfileSchema, ErrorSchema):
        pass

    response = await model.create_structure(
        schema=SchemUnion,
        messages=[UserMessage("Generate a profile of a citizen of Europe.")],
    )

    print(
        json.dumps(
            response.object.model_dump() if isinstance(response.object, BaseModel) else response.object, indent=4
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

### Tool calling

Integrate external tools with your AI model:

```python
import asyncio
import json
import re
import sys
import traceback

from beeai_framework import Message, SystemMessage, Tool, ToolMessage, UserMessage
from beeai_framework.backend.chat import ChatModel, ChatModelParameters
from beeai_framework.backend.message import MessageToolResultContent
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import ToolOutput
from beeai_framework.tools.search import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    model = ChatModel.from_name("ollama:llama3.1", ChatModelParameters(temperature=0))
    tools: list[Tool] = [DuckDuckGoSearchTool(), OpenMeteoTool()]
    messages: list[Message] = [
        SystemMessage("You are a helpful assistant. Use tools to provide a correct answer."),
        UserMessage("What's the fastest marathon time?"),
    ]

    while True:
        response = await model.create(
            messages=messages,
            tools=tools,
        )

        tool_calls = response.get_tool_calls()

        tool_results: list[ToolMessage] = []

        for tool_call in tool_calls:
            print(f"-> running '{tool_call.tool_name}' tool with {tool_call.args}")
            tool: Tool = next(tool for tool in tools if tool.name == tool_call.tool_name)
            assert tool is not None
            res: ToolOutput = await tool.run(json.loads(tool_call.args))
            result = res.get_text_content()
            print(f"<- got response from '{tool_call.tool_name}'", re.sub(r"\s+", " ", result)[:90] + " (truncated)")
            tool_results.append(
                ToolMessage(
                    MessageToolResultContent(
                        result=result,
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.id,
                    )
                )
            )

        messages.extend(tool_results)

        answer = response.get_text_content()

        if answer:
            print(f"Agent: {answer}")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

---

## Embedding model

The `EmbedingModel` class provides functionality for generating vector embeddings from text.

### Embedding model initialization

You can initialize an embedding model in multiple ways:

#### Method 1: Using the generic factory method

The most straightforward way to initialize an embedding model is using the `EmbeddingModel.from_name()` factory method. This method automatically handles the creation of the appropriate provider-specific model based on the name you provide. BeeAI Framework supports various providers out of the box, and this method simplifies their instantiation.

```python
from beeai_framework.backend.embedding import EmbeddingModel

async def factory_method_example():
    # Initialize an embedding model from Ollama (ensure Ollama is running)
    ollama_embedding_model = EmbeddingModel.from_name("ollama:nomic-embed-text")
    print(f"Provider ID: {ollama_embedding_model.provider_id}") # Output: ollama
    print(f"Model ID: {ollama_embedding_model.model_id}") # Output: nomic-embed-text

    # Initialize an embedding model from Watsonx (ensure Watsonx credentials are configured)
    watsonx_embedding_model = EmbeddingModel.from_name("watsonx:ibm/granite-embedding-107m-multilingual")
    print(f"Provider ID: {watsonx_embedding_model.provider_id}") # Output: watsonx
    print(f"Model ID: {watsonx_embedding_model.model_id}") # Output: ibm/granite-embedding-107m-multilingual

await factory_method_example()
```

#### Method 2: Creating a specific provider model directly

For more granular control or when you need to configure provider-specific parameters, you can directly instantiate the embedding model class for your chosen provider. This method allows you to pass in specific configurations as needed.

```python
from beeai_framework.adapters.openai.embedding import OpenAIEmbeddingModel

async def direct_provider_example():
    # Initialize OpenAI Embedding Model directly
    openai_embedding_model = OpenAIEmbeddingModel(
        model_id="text-embedding-3-small",
        config={
            "dimensions": 512, # Optional: Specify embedding dimensions
            "max_embeddings_per_call": 5, # Optional: Limit embeddings per API call
        },
        provider_options={ # Optional: Provider-specific options like custom endpoint, API keys etc.
            # "base_url": "your_custom_endpoint", # Uncomment and set your custom endpoint if needed
            # "api_key": "YOUR_OPENAI_API_KEY", # Uncomment and set your OpenAI API key if not using env vars
            "compatibility": "openai", # Ensure compatibility setting if using custom endpoints
            # "headers": {"CUSTOM_HEADER": "..."}, # Add custom headers if required
        },
    )
    print(f"Provider ID: {openai_embedding_model.provider_id}") # Output: openai
    print(f"Model ID: {openai_embedding_model.model_id}") # Output: text-embedding-3-small

await direct_provider_example()
```

### Embedding model usage

Generate embeddings for one or more text strings using the `create` method. This method accepts a list of text strings in the `values` parameter and returns an `EmbeddingResponse` object containing the generated embeddings.

```python
from beeai_framework.backend.embedding import EmbeddingModel

async def embedding_usage_example():
    # Initialize Ollama Embedding Model using factory method
    embedding_model = EmbeddingModel.from_name("ollama:nomic-embed-text")

    # Generate embeddings for a list of text strings
    response = await embedding_model.create(values=["Hello world!", "Hello Bee!"])

    print("Original Texts:", response.values)
    print("Generated Embeddings:", response.embeddings) # Embeddings will be a list of lists (vectors)
    print("Provider ID:", response.provider_id)
    print("Model ID:", response.model_id)

await embedding_usage_example()
```

### Advanced usage

If your preferred provider isn't directly supported, you can use the LangChain adapter as a bridge.
This allows you to leverage any provider that has LangChain compatibility, extending BeeAI Framework's reach significantly.

```python
import asyncio
import pathlib
import random
import sys
import traceback

import langchain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.file_management.list_dir import ListDirectoryTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from beeai_framework.adapters.langchain.embedding import LangChainEmbeddingModel
from beeai_framework.adapters.langchain.tools import LangChainTool
from beeai_framework.errors import FrameworkError

async def huggingface_embedding_model() -> None:
    """
    Example demonstrating the usage of a HuggingFace embedding model via LangChain adapter.
    """
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2") # Example HuggingFace model
    embedding_model = LangChainEmbeddingModel(hf_embeddings)

    texts_to_embed = ["This is the first sentence.", "Here is another sentence."]
    response = await embedding_model.create(values=texts_to_embed)

    print("Original Texts:", response.values)
    print("Embeddings (first vector, first 5 dimensions):", response.embeddings[0][:5]) # Print first 5 dimensions of the first embedding
    print("Provider ID:", response.provider_id)
    print("Model ID:", response.model_id) # Will indicate LangChain adapter usage

async def directory_list_tool() -> None:
    """Example demonstrating listing directory contents using LangChain's ListDirectoryTool."""
    list_dir_tool = ListDirectoryTool()
    tool = LangChainTool(list_dir_tool)
    dir_path = str(pathlib.Path(__file__).parent.resolve())
    response = await tool.run({"dir_path": dir_path})
    print(f"Listing contents of {dir_path}:\n{response}")


async def custom_structured_tool() -> None:
    """Example of creating and using a custom structured tool via LangChain adapter."""
    class RandomNumberToolArgsSchema(BaseModel):
        min: int = Field(description="The minimum integer", ge=0)
        max: int = Field(description="The maximum integer", ge=0)

    def random_number_func(min: int, max: int) -> int:
        """Generate a random integer between two given integers. The two given integers are inclusive."""
        return random.randint(min, max)

    generate_random_number = StructuredTool.from_function(
        func=random_number_func,
        # coroutine=async_random_number_func, <- if you want to specify an async method instead
        name="GenerateRandomNumber",
        description="Generate a random number between a minimum and maximum value.",
        args_schema=RandomNumberToolArgsSchema,
        return_direct=True,
    )

    tool = LangChainTool(generate_random_number)
    response = await tool.run({"min": 1, "max": 10})

    print(f"Your random number: {response}")


async def main() -> None:
    print("*" * 10, "Using HuggingFace Embedding Model via LangChain")
    await huggingface_embedding_model()
    print("*" * 10, "Using custom StructuredTool")
    await custom_structured_tool()
    print("*" * 10, "Using ListDirectoryTool")
    await directory_list_tool()


if __name__ == "__main__":
    langchain.debug = False
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

**Source:** `/examples/backend/providers/langchain.py`

To run this example, the optional packages:

-   `langchain-core`
-   `langchain-community`

need to be installed.

### Troubleshooting

Common issues and their solutions:

-   **Authentication errors**: Ensure all required environment variables are set correctly, especially API keys and provider-specific credentials.
-   **Model not found**: Verify that the model ID is correct and available for the selected provider. Double-check the model name and provider compatibility.
-   **Package dependencies**: For LangChain integration, make sure you have installed the necessary LangChain packages (`langchain-core`, `langchain-community`, and any provider-specific LangChain integrations like `langchain-openai`).

## Embedding Model

The `EmbedingModel` class represents an Embedding Model and can be initiated in one of the following ways.

```typescript
import { EmbedingModel } from "beeai-framework/backend/core";

const model = await EmbedingModel.fromName("ibm/granite-embedding-107m-multilingual");
console.log(model.providerId); // watsonx
console.log(model.modelId); // ibm/granite-embedding-107m-multilingual
```

or you can always create the concrete provider's embedding model directly

```typescript
import { OpenAIEmbeddingModel } from "beeai-framework/adapters/openai/embedding";

const model = new OpenAIEmbeddingModel(
    "text-embedding-3-large",
    {
        dimensions: 512,
        maxEmbeddingsPerCall: 5,
    },
    {
        baseURL: "your_custom_endpoint",
        compatibility: "compatible",
        headers: {
            CUSTOM_HEADER: "...",
        },
    },
);
```

### Usage

```typescript
import { EmbeddingModel } from "beeai-framework/backend/core";

const model = await EmbeddingModel.fromName("ollama:nomic-embed-text");
const response = await model.create({
    values: ["Hello world!", "Hello Bee!"],
});
console.log(response.values);
console.log(response.embeddings);
```

## Conclusion

**Congratulations!** You've learned how to turn text into powerful numerical representations, enabling AI to understand context, meaning, and relationships with accuracy. You're now capable of building intelligent applications that go beyond simple keyword matching and embrace semantic relevance.

Throughout this BeeAI journey, you've developed critical skills:

- **Prompt Templates**: Guiding language models precisely.
- **ChatModel Interaction**: Creating dynamic conversations.
- **Memory Handling**: Building context-aware interactions.
- **Structured Outputs**: Delivering clear, structured information.
- **ReAct Agents and Tools**: Developing reasoning agents that interact with the real world.
- **Workflows**: Coordinating multi-agent systems for complex tasks.
- **Backend Flexibility**: Deploying AI solutions across diverse platforms.
- **Embedding Models**: Enhancing applications with semantic understanding.

You're now equipped to architect advanced, intelligent systems that deeply understand and interact with the world. BeeAI Framework empowers you to turn your AI visions into reality.

### Connect:

**Email**: [contact@ruslanmv.com](mailto:contact@ruslanmv.com)


Special thanks to the contributors, researchers, supporters, and the open-source community!