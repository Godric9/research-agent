import pytest

from agent.common import ask_terminal


def raise_eof(_prompt=""):
    raise EOFError


def test_ask_terminal_eof_is_refusal(monkeypatch):
    monkeypatch.setattr("builtins.input", raise_eof)
    assert ask_terminal("r.txt", "x") is False


@pytest.mark.parametrize(
    "reply, expected",
    [("y", True), (" Y ", True), ("", False), ("yes", False), ("n", False)],
)
def test_ask_terminal_replies(monkeypatch, reply, expected):
    monkeypatch.setattr("builtins.input", lambda _prompt="": reply)
    assert ask_terminal("r.txt", "x") is expected
