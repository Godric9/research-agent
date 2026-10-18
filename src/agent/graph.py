from pathlib import Path

from langchain_core.tools import tool

from agent import tools as impl


def build_tools(notes_root: Path, output_root: Path) -> list:
    @tool(description=impl.TOOLS_CALCULATOR["description"])
    def calculator(expression: str) -> str:
        return impl.calculator(expression)

    @tool(description=impl.TOOLS_READ_FILE["description"])
    def read_file(path: str) -> str:
        return impl.read_file(path, str(notes_root))

    @tool(description=impl.TOOLS_SEARCH_FILES["description"])
    def search_files(key: str) -> str:
        return impl.search_files(key, str(notes_root))

    @tool(description=impl.TOOLS_WRITE_FILE["description"])
    def write_file(path: str, content: str) -> str:
        return impl.write_file(path, content, str(output_root))

    return [calculator, read_file, search_files, write_file]
