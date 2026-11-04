#!/usr/bin/env python3
"""Dump the exact prompts lm-eval constructs, without loading a model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lm_eval.tasks import TaskManager


HERE = Path(__file__).resolve().parent


def find_occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    return [m.start() for m in re.finditer(re.escape(needle.lower()), text.lower())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="based_triviaqa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--print-count", type=int, default=3)
    args = parser.parse_args()

    manager = TaskManager(include_path=str(HERE / "tasks"))
    loaded = manager.load([args.task])
    tasks = loaded["tasks"]
    if args.task not in tasks:
        raise SystemExit(f"Task not found: {args.task}; loaded={list(tasks)}")

    task = tasks[args.task]
    task.config.num_fewshot = args.num_fewshot
    task.build_all_requests(limit=args.limit)
    instances = [
        instance for instance in task.instances
        if instance.request_type == "generate_until"
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "instances": len(instances),
        "ends_with_question": 0,
        "question_occurs_more_than_once": 0,
        "internal_question_followed_by_context": 0,
        "meaningful_internal_question": 0,
        "generation_kwargs_mismatches": 0,
    }

    with out_path.open("w", encoding="utf-8") as handle:
        for index, instance in enumerate(instances):
            prompt, generation_kwargs = instance.args
            doc = instance.doc
            question = str(doc.get("question", "")).strip()
            occurrences = find_occurrences(prompt, question)
            final_question_start = len(prompt) - len(question)
            internal_occurrences = [
                pos for pos in occurrences if pos != final_question_start
            ]
            internal_followed_by_context = any(
                pos + len(question) < final_question_start
                for pos in internal_occurrences
            )

            ends_with_question = prompt.lower().endswith(question.lower())
            stats["ends_with_question"] += int(ends_with_question)
            stats["question_occurs_more_than_once"] += int(len(occurrences) > 1)
            stats["internal_question_followed_by_context"] += int(
                internal_followed_by_context
            )
            stats["meaningful_internal_question"] += int(
                internal_followed_by_context and len(question) >= 15
            )
            expected_kwargs = {
                "temperature": 0.0,
                "do_sample": False,
                "max_gen_toks": 48,
                "until": ["\n"],
            }
            stats["generation_kwargs_mismatches"] += int(
                generation_kwargs != expected_kwargs
            )

            snippet = ""
            if internal_occurrences:
                pos = internal_occurrences[0]
                start = max(0, pos - 100)
                end = min(len(prompt), pos + len(question) + 250)
                snippet = prompt[start:end].replace("\n", " ")

            row = {
                "index": index,
                "doc_id": doc.get("doc_id"),
                "question": question,
                "answers": doc.get("answers"),
                "generation_kwargs": generation_kwargs,
                "prompt_chars": len(prompt),
                "ends_with_question": ends_with_question,
                "question_occurrences": len(occurrences),
                "internal_question_followed_by_context": internal_followed_by_context,
                "snippet": snippet,
                "prompt": prompt,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Prompts saved to: {out_path}")

    if args.print_count > 0:
        hits = []
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["internal_question_followed_by_context"]:
                    hits.append(row)
                    if len(hits) >= args.print_count:
                        break
        for row in hits:
            print("\n" + "=" * 80)
            print(f"index={row['index']} doc_id={row['doc_id']} qlen={len(row['question'])}")
            print("question:", row["question"])
            print("generation_kwargs:", row["generation_kwargs"])
            print("snippet:", row["snippet"])


if __name__ == "__main__":
    main()
