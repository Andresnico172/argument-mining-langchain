"""Command line entry point: ``argmine path/to/text.txt``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from langchain_ollama import ChatOllama

from .export import to_markdown, to_mermaid
from .graph import build_graph


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="argmine", description="Extract arguments and conclusions from a text.")
    p.add_argument("input", nargs="?", help="Text file to analyse ('-' reads stdin).")
    p.add_argument("--model", default=os.getenv("ARGMINING_MODEL", "llama3.1"), help="Ollama model (default: llama3.1).")
    p.add_argument("--base-url", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--num-ctx", type=int, default=8192, help="Context window passed to Ollama.")
    p.add_argument("--max-revisions", type=int, default=2, help="Extractor/reviewer loop budget.")
    p.add_argument("--out", type=Path, default=Path("output"), help="Directory for result.json, report.md, argument_map.mmd.")
    p.add_argument("--show-graph", action="store_true", help="Print the pipeline graph as Mermaid and exit.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    llm = ChatOllama(model=args.model, base_url=args.base_url, temperature=0, num_ctx=args.num_ctx)
    app = build_graph(llm, max_revisions=args.max_revisions)

    if args.show_graph:
        print(app.get_graph().draw_mermaid())
        return
    if not args.input:
        sys.exit("argmine: an input file is required (or '-' for stdin)")

    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")

    # Stream node updates so the user sees each agent finish.
    state: dict = {}
    start = time.perf_counter()
    for update in app.stream({"text": text}, stream_mode="updates"):
        for node, delta in update.items():
            print(f"[{time.perf_counter() - start:6.1f}s] {node} done", file=sys.stderr)
            if node == "reviewer" and delta.get("feedback"):
                for issue in delta["feedback"]:
                    print(f"           ↳ {issue}", file=sys.stderr)
            state.update(delta or {})

    result = state["result"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "argument_map.mmd").write_text(to_mermaid(result), encoding="utf-8")
    report = to_markdown(result)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.out}/", file=sys.stderr)


def app() -> None:
    """``argmine-app``: start the Streamlit front end (extra args go to Streamlit)."""
    from streamlit.web import cli as streamlit_cli

    sys.argv = [
        "streamlit", "run", str(Path(__file__).with_name("app.py")),
        "--browser.gatherUsageStats=false", "--server.headless=true",
        "--server.address=localhost", "--client.toolbarMode=minimal", *sys.argv[1:],
    ]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
