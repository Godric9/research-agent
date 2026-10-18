from agent import tools as impl
from agent.graph import build_tools


def test_build_tools_match_the_loop_schemas(tmp_path):
    built = {t.name: set(t.args) for t in build_tools(tmp_path, tmp_path)}
    expected = {
        name: set(schema["input_schema"]["properties"])
        for name, (_, schema) in impl.TOOLS.items()
    }
    assert built == expected
