import os
import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from tools.search_tools import SearchTools
from dotenv import load_dotenv

load_dotenv()

serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

search = SearchTools.search_internet

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPER_API_KEY"] = ''
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "LANGGRAPH_EXECUTE_AGENT"


class ReWOO(TypedDict):
    task: str
    plan_string: str    
    steps: List
    results: dict
    result: str

model = ChatOpenAI(temperature=0)

#Regex to match expressions of the form E#..
regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
def get_plan(state: ReWOO):
    result = state["plan_string"]
    matches = re.findall(regex_pattern, result)
    print("Matches:", matches)
    return {"steps": matches, "plan_string": result}


def _get_current_task(state: ReWOO):
    if state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


def tool_execution(state: ReWOO):
    
    """Worker node that executes the tools of a given plan."""
    _step = _get_current_task(state)
    print("Step:", _step)
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    _results = state["results"] or {}
    for k, v in _results.items():
        tool_input = tool_input.replace(k, v)
    if tool == "Google":
        print("-----------------------Google tool result: ")
        result = search.invoke(tool_input)
        print("SerperAPI:", result)
    elif tool == "LLM":
        result = model.invoke(tool_input)
        print("LLM tool result: ", result)
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}



solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""


def solve(state: ReWOO):
    plan = ""
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
    prompt = solve_prompt.format(plan=plan, task=state["task"])
    print("Solve_prompt:", prompt)
    result = model.invoke(prompt)
    return {"result": result.content}



def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"
    
    


graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge("plan", "tool")
graph.add_edge("solve", END)
graph.add_conditional_edges("tool", _route)
graph.set_entry_point("plan")

app = graph.compile()


def execute_graph(checklist_content: str, task_description: str):
    initial_state = ReWOO(task=task_description, plan_string=checklist_content, steps=[], results={}, result="")
    return app.invoke(initial_state)

#Add verification step 