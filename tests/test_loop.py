from types import SimpleNamespace as NS

from agent.loop import run_tool


def block(name, args):
    return NS(type="tool_use", id="t1", name=name, input=args)


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
