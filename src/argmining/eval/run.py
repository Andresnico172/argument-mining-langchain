"""``argmine-eval``: run the pipeline over AAE essays and score it.

Predictions are cached one file per essay, so a long run can be stopped and
resumed, and scoring can be repeated (e.g. with another threshold) without
calling the model again (``--score-only``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ..schemas import ArgumentMap
from .aae import GoldEssay, load_essays
from .metrics import ROLES, aggregate, macro_f1, score_essay


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="argmine-eval", description="Evaluate on the Argument Annotated Essays corpus.")
    p.add_argument("--data", type=Path, default=Path("data/aae2"), help="Folder with the unzipped corpus.")
    p.add_argument("--split", default="TEST", choices=["TEST", "TRAIN", "ALL"])
    p.add_argument("--limit", type=int, default=20, help="Number of essays (0 = all in the split).")
    p.add_argument("--model", default=os.getenv("ARGMINING_MODEL", "llama3.1"))
    p.add_argument("--base-url", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--max-revisions", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5, help="Span overlap needed to match a gold component.")
    p.add_argument("--out", type=Path, default=None, help="Default: output/eval/<model>.")
    p.add_argument("--score-only", action="store_true", help="Score cached predictions without calling the model.")
    return p.parse_args(argv)


def predict(essays: list[GoldEssay], pred_dir: Path, args: argparse.Namespace) -> None:
    from langchain_ollama import ChatOllama

    from ..graph import build_graph

    llm = ChatOllama(model=args.model, base_url=args.base_url, temperature=0, num_ctx=args.num_ctx)
    app = build_graph(llm, max_revisions=args.max_revisions)
    pred_dir.mkdir(parents=True, exist_ok=True)

    todo = [e for e in essays if not (pred_dir / f"{e.id}.json").exists()]
    print(f"{len(essays) - len(todo)} cached, {len(todo)} to run", file=sys.stderr)
    for n, essay in enumerate(todo, start=1):
        start = time.perf_counter()
        try:
            result = app.invoke({"text": essay.text})["result"]
        except Exception as exc:  # keep going: one bad essay should not sink a long run
            print(f"[{n}/{len(todo)}] {essay.id} FAILED: {exc!r}", file=sys.stderr)
            continue
        seconds = time.perf_counter() - start
        record = {"essay": essay.id, "model": args.model, "seconds": seconds, "result": result.model_dump()}
        (pred_dir / f"{essay.id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{n}/{len(todo)}] {essay.id} {seconds:5.1f}s revisions={result.revisions}", file=sys.stderr)


def report(essays: list[GoldEssay], pred_dir: Path, args: argparse.Namespace) -> dict:
    per_essay, seconds, revisions = [], [], []
    for essay in essays:
        path = pred_dir / f"{essay.id}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        pred = ArgumentMap.model_validate(record["result"])
        per_essay.append(score_essay(pred, essay, args.threshold))
        seconds.append(record["seconds"])
        revisions.append(pred.revisions)

    total = aggregate(per_essay)
    rows = {name: {"precision": c.precision, "recall": c.recall, "f1": c.f1, "tp": c.tp,
                   "n_pred": c.n_pred, "n_gold": c.n_gold} for name, c in total.items()}
    return {
        "model": args.model,
        "split": args.split,
        "essays_scored": len(per_essay),
        "essays_requested": len(essays),
        "threshold": args.threshold,
        "components_macro_f1": macro_f1(total) if total else 0.0,
        "mean_seconds": sum(seconds) / len(seconds) if seconds else 0.0,
        "mean_revisions": sum(revisions) / len(revisions) if revisions else 0.0,
        "scores": rows,
    }


def to_markdown(summary: dict) -> str:
    s = summary["scores"]
    lines = [
        f"**{summary['model']}** on {summary['essays_scored']} AAE {summary['split'].lower()} essays "
        f"(span match ≥ {summary['threshold']:.0%}; {summary['mean_seconds']:.0f} s/essay, "
        f"{summary['mean_revisions']:.1f} revisions/essay)",
        "",
        "| Task | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    names = [("segments", "Component segmentation"), *[(r, f"  {r}") for r in ROLES],
             ("links", "Relations (untyped)"), ("relations", "Relations (support/attack)")]
    for key, label in names:
        if key in s:
            r = s[key]
            lines.append(f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    lines.append(f"| **Components, macro F1** | | | **{summary['components_macro_f1']:.3f}** |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    split = None if args.split == "ALL" else args.split
    essays = load_essays(args.data, split=split, limit=args.limit or None)
    out = args.out or Path("output/eval") / args.model.replace(":", "_").replace("/", "_")
    pred_dir = out / "predictions"

    if not args.score_only:
        predict(essays, pred_dir, args)

    summary = report(essays, pred_dir, args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = to_markdown(summary)
    (out / "metrics.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
