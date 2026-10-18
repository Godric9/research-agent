import copy
from types import SimpleNamespace as NS

import pytest

from agent.common import LimiteAtteinte
from agent.loop import run_agent, run_tool

USAGE = NS(input_tokens=10, output_tokens=5)


def block(name, args, id_="t1"):
    return NS(type="tool_use", id=id_, name=name, input=args)


def tool_response(*blocks, stop_reason="tool_use"):
    return NS(stop_reason=stop_reason, content=list(blocks), usage=USAGE)


def final_response(text):
    content = [NS(type="text", text=text)]
    return NS(stop_reason="end_turn", content=content, usage=USAGE)


def fake_client(*responses, seen=None):
    suite = iter(responses)

    def create(**kwargs):
        if seen is not None:
            seen.append(copy.deepcopy(kwargs["messages"]))
        return next(suite)

    return NS(messages=NS(create=create))


def test_run_tool_success(tmp_path):
    result = run_tool(
        block("calculator", {"expression": "12 + 30"}), tmp_path, tmp_path
    )
    assert result == {"type": "tool_result", "tool_use_id": "t1", "content": "42"}


def test_run_tool_unknown_tool_lists_available(tmp_path):
    result = run_tool(block("delete_file", {}), tmp_path, tmp_path)
    assert result["is_error"] is True
    assert "calculator" in result["content"]


def test_run_tool_empty_args(tmp_path):
    result = run_tool(block("read_file", {}), tmp_path, tmp_path)
    assert result["is_error"] is True


def test_run_tool_rejects_model_supplied_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x").write_text("secret", encoding="utf-8")
    notes = tmp_path / "notes"
    notes.mkdir()
    result = run_tool(
        block("read_file", {"path": "x", "root": str(outside)}), notes, tmp_path
    )
    assert result["is_error"] is True
    assert "secret" not in result["content"]


def test_run_tool_tool_error_becomes_result(tmp_path):
    result = run_tool(block("read_file", {"path": "absent.md"}), tmp_path, tmp_path)
    assert result["is_error"] is True
    assert "FileNotFoundError" in result["content"]


def test_run_tool_write_goes_to_output_root(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    out = tmp_path / "out"
    run_tool(block("write_file", {"path": "r.txt", "content": "hi"}), notes, out)
    assert (out / "r.txt").read_text(encoding="utf-8") == "hi"
    assert not (notes / "r.txt").exists()


def test_run_agent_returns_final_answer(tmp_path):
    client = fake_client(
        tool_response(block("calculator", {"expression": "12 + 30"})),
        final_response("The result is 42."),
    )
    result = run_agent("12 + 30 ?", client, tmp_path, tmp_path)
    assert result.answer == "The result is 42."


def test_run_agent_stops_at_max_iterations(tmp_path):
    seen = []
    call = tool_response(block("calculator", {"expression": "1 + 1"}))
    client = fake_client(call, call, call, seen=seen)
    with pytest.raises(LimiteAtteinte):
        run_agent("loop", client, tmp_path, tmp_path, max_iterations=3)
    assert len(seen) == 3


def test_run_agent_groups_results_in_one_user_message(tmp_path):
    seen = []
    client = fake_client(
        tool_response(
            block("calculator", {"expression": "1 + 1"}, id_="a"),
            block("calculator", {"expression": "2 + 2"}, id_="b"),
        ),
        final_response("done"),
        seen=seen,
    )
    run_agent("two tools", client, tmp_path, tmp_path)
    last = seen[1][-1]
    assert last["role"] == "user"
    assert [r["tool_use_id"] for r in last["content"]] == ["a", "b"]
    assert [r["content"] for r in last["content"]] == ["2", "4"]


def test_run_agent_max_tokens_executes_nothing(tmp_path):
    truncated = tool_response(
        block("write_file", {"path": "r.txt", "content": "x"}),
        stop_reason="max_tokens",
    )
    with pytest.raises(ValueError):
        run_agent("write", fake_client(truncated), tmp_path, tmp_path)
    assert not (tmp_path / "r.txt").exists()
