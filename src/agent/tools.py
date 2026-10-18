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
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
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


def read_file(path : str, root : str) -> str:
    """Read a file and return its contents as a string (truncated to MAX_CHARS).

    Raises PermissionError if the file is not within the root directory, FileNotFoundError if the file does not exist.

    Args:
        path (str): The path to the file to read, relative to root.
        root (str): The root directory that the file must be within."""
    root_path = Path(root).resolve()
    full_path = (root_path / path).resolve()
    if not full_path.is_relative_to(root_path):
        raise PermissionError(f"File path {full_path} is not within the root directory {root}")

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
                            results.append(f"{file_path.relative_to(root_path)}: {line.strip()} (l.{line_number})")
                            if len(results) >= MAX_RESULTS:
                                return "\n".join(results) + "\n(results truncated)"
            except (OSError, UnicodeDecodeError):
                print(f"[SEARCH_FILES] (WARN): Error reading file {file_path}, skipping.")

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
        raise PermissionError(f"File path {full_path} is not within the root directory {root}")

    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"written: {full_path.relative_to(root_path)}"


TOOLS_CALCULATOR = {
    "name": "calculator",
    "description": ("Evaluate an arithmetic expression (+, -, *, /, parentheses) and return the result."
                    "To use for every arithmetic expression, even if it is not explicitly requested."
                    "Décimal are represented with a dot (.) and not a comma (,)."
                    "And do not support variables, functions, or dates."
                    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression to evaluate."
            }
        },
        "required": ["expression"]
    },
}

TOOLS_READ_FILE = {
    "name": "read_file",
    "description": "Read a file and return its contents as a string (truncated to MAX_CHARS). Raises PermissionError if the file is not within the root directory, FileNotFoundError if the file does not exist.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to read, relative to root."
            },
            "root": {
                "type": "string",
                "description": "The root directory that the file must be within."
            }
        },
        "required": ["path", "root"]
    },
}

TOOLS_SEARCH_FILES = {
    "name": "search_files",
    "description": "Search for the given key words in all files under the root. Returns a string 'Found in <file_path>: <line> (l.<line_number>)' for each line that contains the key words, or 'No results found' if no matches are found.",
    "input_schema": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key words to search for."
            },
            "root": {
                "type": "string",
                "description": "The root directory to search in."
            }
        },
        "required": ["key", "root"]
    },
}

TOOLS_WRITE_FILE = {
    "name": "write_file",
    "description": "Write content in `root/path`. Creates parent directories if they don't exist. Raises PermissionError if the file is not within the root directory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to write, relative to root."
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file."
            },
            "root": {
                "type": "string",
                "description": "The root directory that the file must be within."
            }
        },
        "required": ["path", "content", "root"]
    },
}

TOOLS = {
    "calculator": (calculator, TOOLS_CALCULATOR),
    "read_file": (read_file, TOOLS_READ_FILE),
    "search_files": (search_files, TOOLS_SEARCH_FILES),
    "write_file": (write_file, TOOLS_WRITE_FILE),
    }