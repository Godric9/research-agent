from pathlib import Path

from agent.tools import TOOLS


def error(block, message):
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": message,
        "is_error": True,
    }


def run_tool(block, notes_root: Path, output_root: Path) -> dict:
    """Run one tool_use block and return the matching tool_result. Never raises."""
    if block.name not in TOOLS:
        return error(
            block, f"Unknown tool {block.name!r}. Available: {', '.join(TOOLS)}"
        )

    function, schema = TOOLS[block.name]

    expected = set(schema["input_schema"]["properties"])
    if set(block.input) != expected:
        return error(block, f"Invalid arguments. Expected exactly: {sorted(expected)}")

    args = dict(block.input)
    if block.name == "write_file":
        args["root"] = output_root
    elif block.name != "calculator":
        args["root"] = notes_root

    try:
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": function(**args),
        }
    except Exception as e:  # noqa: BLE001 - tool errors go back to the model
        return error(block, f"{type(e).__name__}: {e}")
