import json
from pathlib import Path

from agent.common import MODEL, SYSTEM_PROMPT, AgentResult, LimiteAtteinte
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


def log(path, **fields):
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")


def run_agent(
    task, client, notes_root, output_root, max_iterations=8, log_path=None
) -> AgentResult:
    messages = [{"role": "user", "content": task}]
    tools = [schema for _, schema in TOOLS.values()]
    for i in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "max_tokens":
            raise ValueError("Response truncated (max_tokens), nothing executed.")
        if response.stop_reason != "tool_use":
            texts = [b.text for b in response.content if b.type == "text"]
            return AgentResult("".join(texts))
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        results = [run_tool(b, notes_root, output_root) for b in tool_blocks]
        for b, r in zip(tool_blocks, results):
            log(
                log_path,
                iteration=i + 1,
                tool=b.name,
                args=b.input,
                result=str(r["content"])[:200],
                is_error=r.get("is_error", False),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        messages.append({"role": "user", "content": results})
    raise LimiteAtteinte(f"Limit of {max_iterations} iterations reached.")
