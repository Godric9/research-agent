import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from agent.common import LimiteAtteinte
from agent.loop import run_agent


def main(argv=None, client=None) -> int:
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("task")
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args(argv)
    if args.max_iterations < 1:
        parser.error("--max-iterations must be >= 1")

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is missing.", file=sys.stderr)
            return 3
        client = anthropic.Anthropic()

    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs") / f"run-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    try:
        result = run_agent(
            args.task,
            client,
            Path("notes").resolve(),
            Path("sortie").resolve(),
            args.max_iterations,
            log_path,
        )
    except LimiteAtteinte as e:
        print(e, file=sys.stderr)
        return 1
    except (anthropic.APIError, ValueError) as e:
        print(f"API error: {e}", file=sys.stderr)
        return 5
    print(result.answer)
    return 4 if result.write_refused else 0


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())