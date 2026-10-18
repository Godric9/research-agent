from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from agent import tools as impl
from agent.common import MODEL, SYSTEM_PROMPT
from agent.common import AgentResult, LimiteAtteinte


def build_tools(notes_root: Path, output_root: Path) -> list:
    @tool(description=impl.TOOLS_CALCULATOR["description"])
    def calculator(expression: str) -> str:
        return impl.calculator(expression)

    @tool(description=impl.TOOLS_READ_FILE["description"])
    def read_file(path: str) -> str:
        return impl.read_file(path, str(notes_root))

    @tool(description=impl.TOOLS_SEARCH_FILES["description"])
    def search_files(key: str) -> str:
        return impl.search_files(key, str(notes_root))

    @tool(description=impl.TOOLS_WRITE_FILE["description"])
    def write_file(path: str, content: str) -> str:
        return impl.write_file(path, content, str(output_root))

    return [calculator, read_file, search_files, write_file]


def create_agent_node(tools: list):
    model = ChatAnthropic(model=MODEL, max_tokens=2048).bind_tools(
        tools, parallel_tool_calls=False
    )

    def agent_node(state: MessagesState) -> dict:
        response = model.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    return agent_node


def build_graph(agent_node, tools: list):
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_graph(
    task, notes_root, output_root, max_iterations=8, log_path=None, agent_node=None
) -> AgentResult:
    tools = build_tools(notes_root, output_root)
    if agent_node is None:
        agent_node = create_agent_node(tools)
    graph = build_graph(agent_node, tools)

    try:
        result = graph.invoke(
            {"messages": [("user", task)]},
            {"recursion_limit": 2 * max_iterations},
        )
    except GraphRecursionError as e:
        raise LimiteAtteinte(f"Limit of {max_iterations} iterations reached.") from e

    return AgentResult(result["messages"][-1].content)