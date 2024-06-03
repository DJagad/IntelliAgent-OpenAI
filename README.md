# README

## Overview

This project demonstrates the usage of LangChain with OpenAI's GPT models to create an agent that can generate step-by-step plans, execute tasks, and replan based on progress. The core functionalities are structured around environmental variables for API keys, the use of OpenAI models for generating responses, and a workflow to manage the planning, execution, and replanning of tasks.

## Setup

### Prerequisites

- Python 3.8+
- Pip
- An OpenAI API key
- A Tavily API key
- A LangChain API key and project name

### Installation

1. Clone this repository:
    ```sh
    git clone <repository_url>
    cd <repository_name>
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file and add your API keys:
    ```sh
    OPENAI_API_KEY=<your_openai_api_key>
    TAVILY_API_KEY=<your_tavily_api_key>
    LANGCHAIN_API_KEY=<your_langchain_api_key>
    LANGCHAIN_PROJECT=<your_langchain_project_name>
    ```

## Usage

### Running the Agent

1. Load environment variables:
    ```python
    import os
    from dotenv import load_dotenv

    load_dotenv()
    ```

2. Set up API keys in environment variables:
    ```python
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    ```

3. Import necessary libraries and modules:
    ```python
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain import hub
    from langchain.agents import create_openai_functions_agent
    from langchain_openai import ChatOpenAI
    import langchain
    from langgraph.prebuilt import create_agent_executor
    from langchain_core.pydantic_v1 import BaseModel, Field
    from typing import List, Tuple, Annotated, TypedDict
    from collections import deque
    import operator
    from langchain.chains.openai_functions import create_structured_output_runnable
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains.openai_functions import create_openai_fn_runnable
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage
    ```

4. Initialize tools and agent:
    ```python
    tools = [TavilySearchResults(max_results=3)]
    prompt = langchain.hub.pull("hwchase17/openai-functions-agent")
    llm = ChatOpenAI(model="gpt-4-turbo-preview")

    agent_rumble = create_openai_functions_agent(llm=llm, prompt=prompt, tools=tools)
    prompt.messages
    ```

5. Create agent executor:
    ```python
    agent_executor = create_agent_executor(agent_runnable=agent_rumble, tools=tools)
    ```

6. Create a planner for structured output:
    ```python
    class Plan(BaseModel):
        """Plan to follow the step in the future"""
        steps: List[str] = Field(description="different steps to follow, should be in sorted order")

    planner_prompt = ChatPromptTemplate.from_template("""
        For the given objective, come up with a simple step by step plan. \
        This plan should involve the individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip the steps.
        {objective}
    """)

    planner = create_structured_output_runnable(
        Plan,
        ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
        planner_prompt
    )

    planner.invoke({"objective": "What is the hometown of the current Australia Open Winner?"})
    ```

7. Create a replanner for updating plans:
    ```python
    class Response(BaseModel):
        """Response back to User."""
        response: str

    replanner_prompt = ChatPromptTemplate.from_template("""
        For the given objective, come up with a simple step by step plan.\
        This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip the steps.

        Your Objective was this: 
        {input}

        Your Original Plan was this: 
        {plan}

        You have currently done the follow steps: 
        {past_steps}

        Update the plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan. 
    """)

    replanner = create_openai_fn_runnable(
        [Plan, Response],
        ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
        replanner_prompt
    )
    ```

8. Define asynchronous functions for execution, planning, and replanning:
    ```python
    async def execute_step(state: PlanExecute):
        task = state["plan"][0]
        agent_response = await agent_executor.ainvoke({"input": task, "chat_history": []})
        return {"past_steps": (task, agent_response["agent_outcome"].return_values["output"])}

    async def plan_step(state: PlanExecute):
        plan = await planner.ainvoke({"objective": state["input"]})
        return {"plan": plan.steps}

    async def replan_step(state: PlanExecute):
        output = await replanner.ainvoke(state)
        if isinstance(output, Response):
            return {"response": output.response}
        else:
            return {"plan": output.steps}
    ```

9. Define function to determine if workflow should end:
    ```python
    def should_end(state: PlanExecute):
        if "response" in state and state["response"]:
            return True
        else:
            return False
    ```

10. Set up the workflow using StateGraph:
    ```python
    workflow = StateGraph(PlanExecute)

    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        should_end,
        {
            True: END,
            False: "agent"
        }
    )

    app = workflow.compile()
    ```

11. Execute the workflow with an example input:
    ```python
    config = {"recursion_limit": 50}
    inputs = {"input": "Can you Code main.tf for Terraform for automating the instances for the AWS EC2?"}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            for m, n in v.items():
                if m == "response":
                    print(f"Final Response: {n}")
    ```

## License

This project is licensed under the MIT License.
