import pytest

from agent.tools import calculator, read_file, search_files, write_file


# --- calculator: expected result ---
def test_calculator_addition():
    assert calculator("12 + 30") == "42"


def test_calculator_budget_total():
    assert float(calculator("(1249.90 + 387.35 + 2718.28) * 1.2")) == pytest.approx(
        5226.636
    )


# --- calculator: input must be rejected ---
@pytest.mark.parametrize(
    "expr", ["__import__('os')", "'a' * 10", "True + 1", "1e999", "2 +", "9 ** 9 ** 9"]
)
def test_calculator_rejects(expr):
    with pytest.raises(ValueError):
        calculator(expr)


def test_calculator_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        calculator("1 / 0")


# --- read_file: build files inside tmp_path ---
def test_read_file_reads_a_note(tmp_path):
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    assert read_file("a.md", str(tmp_path)) == "hello"


def test_read_file_rejects_traversal(tmp_path):
    with pytest.raises(PermissionError):
        read_file("../../etc/passwd", str(tmp_path))


def test_read_file_rejects_absolute_path(tmp_path):
    with pytest.raises(PermissionError):
        read_file("/etc/passwd", str(tmp_path))


def test_read_file_rejects_symlink(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    target = tmp_path / "secret.txt"  # OUTSIDE the root
    target.write_text("secret", encoding="utf-8")
    (root / "link.md").symlink_to(target)
    with pytest.raises(PermissionError):
        read_file("link.md", str(root))


# --- search_files ---
def test_search_files_two_results(tmp_path):
    (tmp_path / "a.md").write_text("the budget\nother", encoding="utf-8")
    (tmp_path / "b.txt").write_text("over budget", encoding="utf-8")
    assert len(search_files("budget", str(tmp_path)).splitlines()) == 2


def test_search_files_no_results(tmp_path):
    (tmp_path / "a.md").write_text("nothing", encoding="utf-8")
    assert search_files("xyz", str(tmp_path)) == "No results found"


# --- write_file: test of ABSENCE ---
def test_write_file_rejects_outside_root(tmp_path):
    root = tmp_path / "sortie"
    with pytest.raises(PermissionError):
        write_file("../outside.txt", "x", str(root))
    assert not (tmp_path / "outside.txt").exists()


def test_write_file_writes(tmp_path):
    write_file("r.txt", "hi", str(tmp_path))
    assert (tmp_path / "r.txt").read_text(encoding="utf-8") == "hi"