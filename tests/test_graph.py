import pytest
from langchain_core.messages import AIMessage

from agent import tools as impl
from agent.common import LimiteAtteinte
from agent.graph import build_graph, build_tools, run_graph


def test_build_tools_match_the_loop_schemas(tmp_path):
    built = {t.name: set(t.args) for t in build_tools(tmp_path, tmp_path)}
    expected = {
        name: set(schema["input_schema"]["properties"])
        for name, (_, schema) in impl.TOOLS.items()
    }
    assert built == expected


def test_graph_runs_a_tool_call(tmp_path):
    scenario = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "12 + 30"},
                        "id": "t1",
                    }
                ],
            ),
            AIMessage(content="42"),
        ]
    )
    graph = build_graph(
        lambda state: {"messages": [next(scenario)]}, build_tools(tmp_path, tmp_path)
    )
    result = graph.invoke({"messages": [("user", "12 + 30 ?")]})
    assert result["messages"][2].content == "42"  # ToolMessage from calculator
    assert result["messages"][-1].content == "42"


def test_run_graph_stops_at_max_iterations(tmp_path):
    n = 0

    def always_tool(state):
        nonlocal n
        n += 1
        call = {"name": "calculator", "args": {"expression": "1 + 1"}, "id": f"t{n}"}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    with pytest.raises(LimiteAtteinte):
        run_graph("loop", tmp_path, tmp_path, max_iterations=3, agent_node=always_tool)
    assert n == 3


def test_run_graph_returns_final_answer(tmp_path):
    fake = lambda state: {"messages": [AIMessage(content="done")]}
    result = run_graph("hi", tmp_path, tmp_path, agent_node=fake)
    assert result.answer == "done"
