from langchain_core.messages import AIMessage

from agent import tools as impl
from agent.graph import build_graph, build_tools


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
