import ast
import math
import operator
from pathlib import Path

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

MAX_RESULTS = 50
MAX_CHARS = 5000


def _evaluate(node: ast.AST) -> float:
    """Recursively compute a node's value. Any node type outside the whitelist is rejected."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        v = node.value
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
        ):
            raise ValueError(f"rejected constant: {v!r}")
        return v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression (+, -, *, /, parentheses) and return the result.

    Raises ValueError if the expression is malformed or contains anything other than
    numbers and these operators, ZeroDivisionError on division by zero.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {expression!r}") from e
    return str(_evaluate(tree))


def read_file(path: str, root: str) -> str:
    """Read a file and return its contents as a string (truncated to MAX_CHARS).

    Raises PermissionError if the file is not within the root directory, FileNotFoundError if the file does not exist.

    Args:
        path (str): The path to the file to read, relative to root.
        root (str): The root directory that the file must be within."""
    root_path = Path(root).resolve()
    full_path = (root_path / path).resolve()
    if not full_path.is_relative_to(root_path):
        raise PermissionError(
            f"File path {full_path} is not within the root directory {root}"
        )

    if not full_path.is_file():
        raise FileNotFoundError(f"File not found: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read(MAX_CHARS + 1)
    if len(content) > MAX_CHARS:
        return content[:MAX_CHARS] + "\n(truncated)"
    return content


def search_files(key: str, root: str) -> str:
    """
    Search for the given key words in all files under the root.

    Returns a string "Found in <file_path>: <line> (l.<line_number>)" for each line that contains the key words, or "No results found" if no matches are found.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"Root path {root} is not a directory")

    key = key.lower()
    results = []

    for pattern in ("*.md", "*.txt"):
        for file_path in root_path.rglob(pattern):
            try:
                with open(file_path, encoding="utf-8") as f:
                    for line_number, line in enumerate(f, start=1):
                        if key in line.lower():
                            results.append(
                                f"{file_path.relative_to(root_path)}: {line.strip()} (l.{line_number})"
                            )
                            if len(results) >= MAX_RESULTS:
                                return "\n".join(results) + "\n(results truncated)"
            except (OSError, UnicodeDecodeError):
                print(
                    f"[SEARCH_FILES] (WARN): Error reading file {file_path}, skipping."
                )

    if not results:
        return "No results found"
    return "\n".join(results)


def write_file(path: str, content: str, root: str) -> str:
    """Write content in `root/path`. Creates parent directories if they don't exist.

    Raises PermissionError if the file is not within the root directory.
    Args:
        path (str): The path to the file to write, relative to root.
        content (str): The content to write to the file.
        root (str): The root directory that the file must be within.
    """
    root_path = Path(root).resolve()
    full_path = (root_path / Path(path)).resolve()
    if not full_path.is_relative_to(root_path):
        raise PermissionError(
            f"File path {full_path} is not within the root directory {root}"
        )

    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"written: {full_path.relative_to(root_path)}"


TOOLS_CALCULATOR = {
    "name": "calculator",
    "description": (
        "Evaluate an arithmetic expression (+, -, *, /, parentheses) and return the exact result. "
        "Use it for every calculation, even a simple one, instead of computing in your head. "
        "Decimals are written with a dot, not a comma. "
        "Does not support variables, functions, powers or dates."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "for example '(12.5 + 30) * 1.2'",
            }
        },
        "required": ["expression"],
    },
}

TOOLS_READ_FILE = {
    "name": "read_file",
    "description": (
        "Read one note from the user's notes folder and return its text (long notes are truncated). "
        "The path comes from a search_files result. "
        "Cannot read anything outside the notes folder."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "note path as returned by search_files, for example 'budget-projet.md'",
            }
        },
        "required": ["path"],
    },
}

TOOLS_SEARCH_FILES = {
    "name": "search_files",
    "description": (
        "Search the user's notes for a keyword (case-insensitive substring). "
        "Returns one line per match: '<note path>: <line> (l.<number>)', or 'No results found'. "
        "Matches show only the line containing the word, so call read_file to see the full note. "
        "Use a single short keyword, not a sentence."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "keyword to look for, for example 'budget'",
            }
        },
        "required": ["key"],
    },
}

TOOLS_WRITE_FILE = {
    "name": "write_file",
    "description": (
        "Write text to a file in the output folder. Overwrites an existing file. "
        "Requires the user's approval, which may be refused. "
        "Call it once, only when the final content is ready."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "file name in the output folder, for example 'resultat.txt'",
            },
            "content": {
                "type": "string",
                "description": "the full text to write",
            },
        },
        "required": ["path", "content"],
    },
}

TOOLS = {
    "calculator": (calculator, TOOLS_CALCULATOR),
    "read_file": (read_file, TOOLS_READ_FILE),
    "search_files": (search_files, TOOLS_SEARCH_FILES),
    "write_file": (write_file, TOOLS_WRITE_FILE),
}
