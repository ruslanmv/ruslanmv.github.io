---
title: "Building Multi-Agent Systems with crewAI and Watsonx.ai"
excerpt: "How to build multi-agent system with  crewAI and WatsonX"

header:
  image: "./../assets/images/posts/2024-10-20-How-to-build-multi-agent-system-with crewAI-and-Watsonx/image.jpeg"
  teaser: "./../assets/images/posts/2024-10-20-How-to-build-multi-agent-system-with crewAI-and-Watsonx/crew.jpeg"
  caption: "Multimodal agents represent a significant step towards creating truly intelligent AI systems that can seamlessly integrate into our lives. Watsonx.ai"
  
---

Hello everyone, in this blog post, we’ll explore how to build a multi-agent system using **crewAI** integrated with **Watsonx.ai**. We'll walk through the concepts of crewAI and explain how to assemble agents and tasks to create dynamic workflows.

 By the end of this post, you will have a complete understanding of how to build your own multi-agent system using crewAI, with detailed explanations and Python code examples.

## Introduction

As AI systems grow in complexity, the need for structured and modular approaches to managing intelligent agents has become essential. **crewAI** provides a framework for creating and orchestrating multi-agent systems with ease. It integrates with LangChain and Watsonx.ai to leverage advanced language models and tools for building sophisticated, intelligent workflows.

This guide will explain the key components of crewAI, including **Agents**, **Tasks**, and the **Crew** itself. We’ll also show how to set up the environment, integrate WatsonxLLM for language modeling, and use the **Tavily Search Tool** for retrieving information from the web. Finally, we'll provide a detailed code walkthrough and example usage.

## What is crewAI?

**crewAI** is a Python library designed to build agent-based systems where each agent has a specific role and task. It allows you to create multi-agent systems by defining agents, assigning them tasks, and orchestrating their interactions to perform complex workflows efficiently.

crewAI is ideal for building applications that require multiple agents working together, such as chatbots, automated research tools, and information retrieval systems. It provides flexibility in defining agents, assigning roles, and integrating various tools for dynamic decision-making.

### Key Components of crewAI

1. **Agent**: An **Agent** in crewAI is an autonomous unit that performs specific tasks. Agents can be language models (like WatsonxLLM), functions, or tools that interact with users or systems.

2. **Task**: A **Task** defines a specific action or behavior for an agent. It links an agent to a goal and specifies the expected output.

3. **Crew**: A **Crew** is a collection of agents and tasks assembled to work together. The crew orchestrates the workflow, ensuring that each agent performs its task based on the user’s input and the decision-making logic implemented.

## Setting Up the Environment

Before we dive into the code, let's set up the environment properly to ensure all dependencies are installed.

### Step 1: Install Python 3.12

Make sure you have **Python 3.12** installed. Verify your Python version:

```bash
python3 --version
```

If you need to install Python 3.12, download it from [Python's official site](https://www.python.org/downloads/) or use a package manager like `pyenv`.

### Step 2: Create a Virtual Environment

Create a virtual environment (`.venv`) using Python 3.12:

```bash
python3.12 -m venv .venv
```

### Step 3: Activate the Virtual Environment

Activate the virtual environment:

- On **macOS/Linux**:

```bash
source .venv/bin/activate
```

- On **Windows**:

```bash
.venv\Scripts\activate
```

### Step 4: Create a `requirements.txt` File

Add the necessary dependencies for the project:

```
crewai
langsmith
python-dotenv
ibm_watsonx_ai
tavily-python
langchain-core
langchain-community
langchain-ibm
```

### Step 5: Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

### Step 6: Create a `.env` File

Create a `.env` file in your project directory to store API keys:

```
WATSONX_API_KEY=your-watsonx-api-key
PROJECT_ID=your-watsonx-project-id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
TAVILY_API_KEY=your-tavily-api-key
```

Replace placeholders with your actual credentials.

## Building a Multi-Agent System with crewAI

Below is the Python code that demonstrates how to build a multi-agent system using crewAI:

```python
from crewai import Crew, Task, Agent
from langchain_ibm import WatsonxLLM
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# Load environment variables from .env file
load_dotenv()

# Load API key and project ID from environment variables
watsonx_api_key = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("PROJECT_ID")
url = os.getenv("WATSONX_URL")

# WatsonxLLM parameters
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.MIN_NEW_TOKENS: 50,
    GenParams.TEMPERATURE: 0.7,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

if not watsonx_api_key or not project_id:
    raise ValueError("Please set the WATSONX_API_KEY and PROJECT_ID in your .env file.")

# Define LLM models using WatsonxLLM
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-1-70b-instruct",  
    url=url,
    apikey=watsonx_api_key,
    project_id=project_id,
    params=parameters
)

# Tool Initialization for the Researcher
tavily_tool = TavilySearchResults(max_results=5)  
```

### Explanation of the Code

- **Environment Variables**: We use `dotenv` to load API keys and project information securely from a `.env` file.
- **WatsonxLLM Setup**: We configure WatsonxLLM with the API key and parameters like temperature and token limits for language generation.
- **Tool Initialization**: The `TavilySearchResults` tool is initialized to retrieve up to 5 search results.

### Defining Agents

```python
# Router Agent definition
def router_decision(user_query):
    """Router agent that selects either 'Researcher' or 'Creator' based on the user input."""
    if "price" in user_query or "news" in user_query:
        return "Researcher"
    else:
        return "Creator"

router = Agent(
    llm=llm,
    role="Router",
    goal="Decide whether to use the Researcher or Creator agent based on the user query.",
    function=router_decision,
    backstory="You are a router that directs queries to the appropriate agent."
)

# Researcher Agent
researcher = Agent(
    llm=llm,
    role="Researcher",
    goal="Fetch the latest information available on the internet based on the user query.",
    backstory="You are an AI researcher skilled in retrieving the latest information.",
    tools=[tavily_tool]  
)

# Creator Agent
creator = Agent(
    llm=llm,
    role="Creator",
    goal="Generate informative and accurate knowledge-based responses to user queries.",
    backstory="You are an AI assistant specializing in generating knowledge-based answers."
)
```

**Explanation:**

- **Router Agent**: Decides whether the user query should be handled by the Researcher or Creator agent based on the keywords in the query.
- **Researcher Agent**: Uses the `TavilySearchResults` tool to fetch information from the web based on user queries.
- **Creator Agent**: Uses WatsonxLLM to generate responses based on general knowledge.

### Defining Tasks

```python
# Define tasks for the agents
task_router = Task(
    description="Route the user query to either the Researcher or Creator agent.",
    expected_output="Either 'Researcher' or 'Creator'",
    agent=router
)

task_researcher = Task(
    description="Fetch search results based on the user query.",
    expected_output="A summary of the search results.",
    agent=researcher
)

task_creator = Task(
    description="Generate a response to the user query based on general knowledge.",
    expected_output="An informative response.",
    agent=creator
)
```

**Explanation:**

- Each task specifies what an agent should do and the expected outcome.
- **Router Task**: Directs queries to the appropriate agent.
- **Researcher Task**: Retrieves web information.
- **Creator Task**: Generates knowledge-based responses.

### Assembling the Crew

```python
# Assemble the crew with the agents and tasks
crew = Crew(
    agents=[router, researcher, creator],
    tasks=[task_router, task_researcher, task_creator],
    verbose=True,
    output_log_file="crew_log.txt", 
    share_crew=False
)
```

- We assemble the crew with agents and tasks, specifying settings such as verbosity and log files.

### Executing the Workflow

```python
# Execute the workflow 
def crew_workflow(user_query):
    print("\n==== Starting Crew Workflow ====\n")
    
    # Step 1: Router Agent determines which agent to use
    print(f"[Router] Evaluating the query: '{user_query}'")
    selected_agent = router_decision(user_query)
    print(f"[Router] Selected Agent: {selected_agent}\n")
    
    # Step 2: Based on the selected agent, either Researcher or Creator is invoked
    if selected_agent == "

Researcher":
        print(f"[Researcher] Processing query: '{user_query}'")
        search_result = tavily_tool.invoke(user_query)  
        print(f"[Researcher] Result: {search_result}\n")
    elif selected_agent == "Creator":
        creator_prompt = f"Please provide an informative response for: '{user_query}'"
        result = llm.invoke(creator_prompt) 
        print(f"[Creator] Result: {result}\n")

# Example usage
crew_workflow("Fetch the bitcoin price over the past 5 days.") 
crew_workflow("Explain what Bitcoin is.")
```

This code simulates a workflow where the router determines which agent to use based on the user query, and the selected agent performs the task.

### Example Output

```bash
==== Starting Crew Workflow ====

[Router] Evaluating the query: 'Fetch the bitcoin price over the past 5 days.'
[Router] Selected Agent: Researcher

[Researcher] Processing query: 'Fetch the bitcoin price over the past 5 days.'
[Researcher] Result: [{'url': 'https://www.coindesk.com/price/bitcoin', 'content': "The price of Bitcoin (BTC) is $68,140.13 today as of Oct 24, 2024, 8:12 pm EDT, with a 24-hour trading volume of $35.97B. ... over the past decade ... and over the next four months, bitcoin's"}, {'url': 'https://www.investing.com/crypto/bitcoin/historical-data', 'content': "Get historical data for the Bitcoin prices. You'll find the historical Bitcoin market data for the selected range of dates. The data can be viewed in daily, weekly or monthly time intervals."}, {'url': 'https://www.worldcoinindex.com/coin/bitcoin', 'content': 'Bitcoin BTC price graph info 24 hours, 7 day, 1 month, 3 month, 6 month, 1 year. Prices denoted in BTC, USD, EUR, CNY, RUR, GBP. ... The Bitcoin price today is $66,813 USD with a 24 hour trading volume of $11.37B USD. Bitcoin (BTC) is up 0.26% in the last 24 hours. ... There are over 100,000 merchants and vendors accepting Bitcoin all over the'}, {'url': 'https://www.nasdaq.com/market-activity/cryptocurrency/btc/historical', 'content': 'Bitcoin (BTC) Historical prices - Nasdaq offers historical cryptocurrency prices & market activity data for US and global markets.'}, {'url': 'https://coinmarketcap.com/currencies/bitcoin/historical-data/', 'content': "The live Bitcoin price today is $68,022.51 USD with a 24-hour trading volume of $29,802,189,834.69 USD. We update our BTC to USD price in real-time. ... Bitcoin's price history is a testament to its evolution and growth over the years. It started with a value of almost nothing and has grown to be one of the most valuable assets in the world"}]


==== Starting Crew Workflow ====

[Router] Evaluating the query: 'Explain what Bitcoin is.'
[Router] Selected Agent: Creator

[Creator] Result: "
Answer: Bitcoin is a decentralized digital currency that allows for peer-to-peer transactions without the need for a central authority or intermediary. It was created in 2009 by an individual or group of individuals using the pseudonym Satoshi Nakamoto. Bitcoin is based on a decentralized technology called blockchain, which is a public ledger that records all transactions made with the currency. Transactions are verified by a network of computers around the world, rather than by a central bank or government. Bitcoin can be bought, sold,
```

This example demonstrates how the system processes different queries dynamically using the defined agents and tools.

## Conclusion

crewAI, combined with Watsonx.ai, provides a powerful way to create multi-agent systems that can handle complex workflows. By defining agents, tasks, and a crew, you can build sophisticated AI applications with dynamic decision-making capabilities.

For the complete code you can go [here](https://github.com/ruslanmv/How-to-build-multiagent-system-with-WatsonX/blob/master/crew/model.ipynb).



**Congratulations!** You have learned how to build a multi-agent system with **CrewAI** and **Watsonx.ai**