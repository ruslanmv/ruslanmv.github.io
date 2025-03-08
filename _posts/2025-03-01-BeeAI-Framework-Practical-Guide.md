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

<script src="https://gist.github.com/ruslanmv/aff25c5ff488c6831764a3014e6aa5c7.js"></script>


### 2. More Complex Templates

The `PromptTemplate` class also supports more complex structures. For example, you can iterate over a list of search results to build a prompt.

#### Example: Template with a List of Search Results



<script src="https://gist.github.com/ruslanmv/45a946a43ea39f070dc9c458c8ad4261.js"></script>




### 3. The ChatModel

Once you have your prompt templates set up, you can begin interacting with a language model. BeeAI supports various LLMs through the `ChatModel` interface.

#### Example: Creating a User Message

<script src="https://gist.github.com/ruslanmv/da57f48163837c90fff0ba15406d3749.js"></script>

#### Example: Sending a Message to the ChatModel

<script src="https://gist.github.com/ruslanmv/bfa85251871697ff5809abd2b3276640.js"></script>



### 4. Memory Handling

Memory is a convenient way to store the conversation history (a series of messages) that the model uses for context.

#### Example: Storing and Retrieving Conversation History

<script src="https://gist.github.com/ruslanmv/d17ac8d72326c087c17035d4b9b6db15.js"></script>



### 5. Combining Templates and Messages

You can render a prompt from a template and then send it as a message to the ChatModel.

#### Example: Rendering a Template and Sending as a Message

<script src="https://gist.github.com/ruslanmv/8124c8c4b379fd0d3f9bb025b004b757.js"></script>





### 6. Structured Outputs

Sometimes you want the LLM to generate output in a specific format. You can enforce this using structured outputs with a Pydantic schema.

#### Example: Enforcing a Specific Output Format



<script src="https://gist.github.com/ruslanmv/65183d1802a86c1ecc851b3bade1975e.js"></script>

### 7. System Prompts

System messages can guide the overall behavior of the language model.

#### Example: Using a System Message

<script src="https://gist.github.com/ruslanmv/926415f4b8cc1b14f1475f2e89d6a265.js"></script>



## BeeAI ReAct Agents

The BeeAI ReAct agent implements the “Reasoning and Acting” pattern, separating the process into distinct steps. This section shows how to build an agent that uses its own memory for reasoning and even integrates tools for added functionality.

### 1. Basic ReAct Agent

#### Example: Setting Up a Basic ReAct Agent



<script src="https://gist.github.com/ruslanmv/7f835119fc15624fc787894c70e0c01d.js"></script>

### 2. Using Tools with the Agent

Agents can be extended with tools so that they can perform external actions, like fetching weather data.

#### Example: Using a Built-In Weather Tool

<script src="https://gist.github.com/ruslanmv/133ac659fff3e57d5240847cc3c43ef3.js"></script>


### 3. Imported Tools

You can also import tools from other libraries. Below are two examples that show how to integrate Wikipedia search via LangChain.

#### Example: Long-Form Integration with Wikipedia

<script src="https://gist.github.com/ruslanmv/a0c35250a86f71073742ff26643a422a.js"></script>



#### Example: Shorter Form Using the `@tool` Decorator



<script src="https://gist.github.com/ruslanmv/5b90c8bef331e2a2322811080bee22c1.js"></script>



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

<script src="https://gist.github.com/ruslanmv/2bc268ac8d519ada5cc7831aec279571.js"></script>

#### Multi-Step Workflow

This advanced example showcases a workflow that implements multiplication through repeated addition—demonstrating control flow, state manipulation, nesting, and conditional logic.

<script src="https://gist.github.com/ruslanmv/54f7c75a222fbcf3f3584d266a2814f8.js"></script>



This workflow demonstrates several powerful concepts:

- Implementing loops by returning `Workflow.SELF`
- Conditional transitions between steps
- Progressive state modification to accumulate results
- Sign handling through state transformation
- Type-safe step transitions using Literal types



### Advanced Features

#### Workflow Nesting

Workflow nesting allows complex behaviors to be encapsulated as reusable components, enabling hierarchical composition of workflows. This promotes modularity, reusability, and better organization of complex agent logic.

<script src="https://gist.github.com/ruslanmv/de9a1d363b04f1ef4d2e6afacd5b6a9c.js"></script>

#### Multi-Agent Workflows: Orchestration with BeeAI

The multi-agent workflow pattern enables the orchestration of specialized agents that collaborate to solve complex problems. Each agent focuses on a specific domain or capability, with results combined by a coordinator agent.  BeeAI Framework's workflow engine is perfectly suited for creating sophisticated multi-agent systems.

The following example demonstrates how to orchestrate a multi-agent system using BeeAI workflows with Ollama backend. We will create a "Smart assistant" workflow composed of three specialized agents: `WeatherForecaster`, `Researcher`, and `Solver`.

<script src="https://gist.github.com/ruslanmv/dfc309e84f95a105823bed14cc83f93e.js"></script>



This pattern demonstrates:

- **Role specialization** through focused agent configuration. `WeatherForecaster` is designed specifically for weather-related queries, while `Researcher` is for general information retrieval.
- **Efficient tool distribution** to relevant specialists. The `WeatherForecaster` agent is equipped with the `OpenMeteoTool`, and `Researcher` with `DuckDuckGoSearchTool`, ensuring each agent has the right tools for its job.
- **Parallel processing** of different aspects of a query.  Although not explicitly parallel in this example, the workflow structure is designed to easily support parallel execution of agents if needed.
- **Synthesis of multiple expert perspectives** into a cohesive response. The `Solver` agent acts as a coordinator, taking responses from other agents and synthesizing them into a final answer.
- **Declarative agent configuration** using the `AgentWorkflow` and `add_agent` methods, which simplifies the setup and management of complex agent systems.

**Orchestration with Watsonx.ai Backend**

To demonstrate the versatility of BeeAI workflows, let's adapt the multi-agent workflow example to use Watsonx.ai as the backend LLM provider. First, ensure you have configured the Watsonx provider as described in the Backend section. Then, modify the `ChatModel.from_name` call to use a Watsonx model:

<script src="https://gist.github.com/ruslanmv/fd0aa5270484b09fc16337493f118717.js"></script>

In this modified example, we simply changed the `ChatModel.from_name` call to `watsonx:ibm/granite-3-8b-instruct`.  Assuming you have correctly set up your Watsonx environment variables, this code will now orchestrate the same multi-agent workflow but powered by Watsonx.ai. This highlights the provider-agnostic nature of BeeAI workflows, allowing you to easily switch between different LLM backends without significant code changes.

#### Memory in Workflows

Integrating memory into workflows allows agents to maintain context across interactions, enabling conversational interfaces and stateful processing. This example demonstrates a simple conversational echo workflow with persistent memory.



<script src="https://gist.github.com/ruslanmv/f5aafad0418d06fda2e563d72aa0b0c2.js"></script>

This pattern demonstrates:

- Integration of memory as a first-class citizen in workflow state
- Conversation loops that preserve context across interactions
- Bidirectional memory updating (reading recent messages, storing responses)
- Clean separation between the persistent memory and workflow-specific state

##  Backend



Backend is an umbrella module that encapsulates a unified way to work with the following functionalities:

- Chat Models via (ChatModel class)
- Embedding Models (coming soon)
- Audio Models (coming soon)
- Image Models (coming soon)

BeeAI framework's backend is designed with a provider-based architecture, allowing you to switch between different AI service providers while maintaining a consistent API.



## Supported providers



The table below lists supported providers, their dependencies, and required environment variables. Ensure these variables are properly configured before using each provider.

| Provider         | Chat Support | Dependency            | Required Environment Variables                               |
| ---------------- | ------------ | --------------------- | ------------------------------------------------------------ |
| Ollama           | Yes            | ollama-ai-provider    | `OLLAMA_CHAT_MODEL`,`OLLAMA_BASE_URL`                        |
| OpenAI           | Yes            | openai                | `OPENAI_CHAT_MODEL`,`OPENAI_API_BASE`,`OPENAI_API_KEY`,`OPENAI_ORGANIZATION` |
| Watsonx          | Yes           | @ibm-cloud/watsonx-ai | `WATSONX_CHAT_MODEL`,`WATSONX_API_KEY`,`WATSONX_PROJECT_ID`,`WATSONX_SPACE_ID`,`WATSONX_VERSION`,`WATSONX_REGION` |
| Groq             | Yes            |                       | `GROQ_CHAT_MODEL`,`GROQ_API_KEY`                             |
| Amazon Bedrock   | Yes            | boto3                 | `AWS_CHAT_MODEL`,`AWS_ACCESS_KEY_ID`,`AWS_SECRET_ACCESS_KEY`,`AWS_REGION_NAME` |
| Google Vertex AI | Yes            |                       | `VERTEXAI_CHAT_MODEL`,`VERTEXAI_PROJECT`,`GOOGLE_APPLICATION_CREDENTIALS`,`GOOGLE_APPLICATION_CREDENTIALS_JSON`,`GOOGLE_CREDENTIALS` |
| Azure OpenAI     | No            | Coming soon!          | `AZURE_OPENAI_CHAT_MODEL`,`AZURE_OPENAI_API_KEY`,`AZURE_OPENAI_API_ENDPOINT`,`AZURE_OPENAI_API_RESOURCE`,`AZURE_OPENAI_API_VERSION` |
| Anthropic        | Yes            |                       | `ANTHROPIC_CHAT_MODEL`,`ANTHROPIC_API_KEY`                   |
| xAI              | Yes           |                       | `XAI_CHAT_MODEL`,`XAI_API_KEY`                               |


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

<script src="https://gist.github.com/ruslanmv/ec2f547d126a02987f9262f67c978f5a.js"></script>


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



### Text generation

The most basic usage is to generate text responses:

<script src="https://gist.github.com/ruslanmv/426eeecca24b9562c73290d0e8953882.js"></script>

> [!NOTE]
>
> Execution parameters (those passed to model.create({...})) are superior to ones defined via config.



### Streaming responses

For applications requiring real-time responses:

<script src="https://gist.github.com/ruslanmv/3a82c392540fc61a3fe2af52e187050d.js"></script>



### Structured generation

Generate structured data according to a schema:

<script src="https://gist.github.com/ruslanmv/f85e39315d8ebc162335e93dbbddff3a.js"></script>



### Tool calling

Integrate external tools with your AI model:

<script src="https://gist.github.com/ruslanmv/b1743ddde0e9815aeb0820be49719da2.js"></script>



## Embedding model

The `EmbedingModel` class provides functionality for generating vector embeddings from text.



### Embedding model initialization

You can initialize an embedding model in multiple ways:

#### Method 1: Using the generic factory method

The most straightforward way to initialize an embedding model is using the `EmbeddingModel.from_name()` factory method. This method automatically handles the creation of the appropriate provider-specific model based on the name you provide. BeeAI Framework supports various providers out of the box, and this method simplifies their instantiation.

<script src="https://gist.github.com/ruslanmv/b7691da93a3034740f818a3a6f93dc4c.js"></script>



#### Method 2: Creating a specific provider model directly

For more granular control or when you need to configure provider-specific parameters, you can directly instantiate the embedding model class for your chosen provider. This method allows you to pass in specific configurations as needed.

<script src="https://gist.github.com/ruslanmv/8b60c3b8b619bcbe0e4e6c58d5e5b2b5.js"></script>

### Embedding model usage

Generate embeddings for one or more text strings using the `create` method. This method accepts a list of text strings in the `values` parameter and returns an `EmbeddingResponse` object containing the generated embeddings.

<script src="https://gist.github.com/ruslanmv/49427aff41205b4f7aa537c3856e47c7.js"></script>

### Advanced usage

If your preferred provider isn't directly supported, you can use the LangChain adapter as a bridge.
This allows you to leverage any provider that has LangChain compatibility, extending BeeAI Framework's reach significantly.

<script src="https://gist.github.com/ruslanmv/3ca54d4f3ae637d04988945bd58460ac.js"></script>



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



The `EmbedingModel` class represents an Embedding Model and can be initiated in one of the following ways, for example considering the node js.

<script src="https://gist.github.com/ruslanmv/350b0e43f60321c855587f9033d9554e.js"></script>





or you can always create the concrete provider's embedding model directly

<script src="https://gist.github.com/ruslanmv/b6f33d26f85cb4fe7afaf1df9b0bb840.js"></script>



### Usage

<script src="https://gist.github.com/ruslanmv/7c73f8ea8feee360c5f7789e30ec0152.js"></script>

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