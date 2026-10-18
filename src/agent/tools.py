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
    """Read a file and return its contents as a string.
    
    Raises ValueError if the file is not within the root directory, FileNotFoundError if the file does not exist.
    
    Args:
        path (str): The path to the file to read.
        root (str): The root directory that the file must be within."""
    full_path = Path(path).resolve()
    if not full_path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"File path {full_path} is not within the root directory {root}")

    if not full_path.is_file():
        raise FileNotFoundError(f"File not found: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()