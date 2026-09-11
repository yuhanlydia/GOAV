#!/usr/bin/env python3
"""Build a GOAV K=8 bank from PBPF generations and real EvalPlus executions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth, untrusted_check

from goav.bank import CandidateGroup, CheapTest, bank_hash, write_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbpf-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=164)
    args = parser.parse_args()
    problems = get_human_eval_plus()
    expected = get_groundtruth(problems, get_human_eval_plus_hash(), [])
    artifact_by_task = {}
    for root in args.pbpf_roots:
        for report in root.glob("HumanEval-*/report.json"):
            task_id = report.parent.name.replace("HumanEval-", "HumanEval/")
            artifact_by_task[task_id] = report.parent
    groups, labels = [], {}
    kinds = ["curated"] * 12 + ["generated"] * 12 + ["fault_targeted"] * 12 + ["metamorphic"] * 8 + ["control"] * 4
    for index, (task_id, problem) in enumerate(list(problems.items())[: args.num_tasks]):
        artifact = artifact_by_task[task_id]
        candidates = [json.loads(line) for line in (artifact / "bank/candidates.jsonl").read_text().splitlines()]
        candidates = [item for item in candidates if item["version"] == 0]
        if len(candidates) != 8:
            raise ValueError(f"{task_id} does not have exactly eight initial candidates")
        inputs = problem["plus_input"][:48]
        expected_rows = expected[task_id]["plus"][:48]
        ref_times = expected[task_id]["plus_time"][:48]
        matrix, full_labels, texts = [], [], []
        for candidate_index, candidate in enumerate(candidates):
            code = candidate["code"]
            status, details = untrusted_check(
                "humaneval", code, inputs, problem["entry_point"], expected_rows,
                problem["atol"], ref_times, fast_check=False,
            )
            row = np.asarray(details, dtype=np.int8)
            if row.shape != (48,):
                row = np.pad(row[:48], (0, max(0, 48 - len(row))), constant_values=0)
            matrix.append(row)
            full_status, _ = untrusted_check(
                "humaneval", code, problem["plus_input"], problem["entry_point"],
                expected[task_id]["plus"], problem["atol"], expected[task_id]["plus_time"],
                fast_check=True,
            )
            full_labels.append(float(full_status == "pass"))
            texts.append(code if code.strip() else f"# empty model response {candidate_index}")
        tests = tuple(CheapTest(f"evalplus-{i:02d}", kind, "evaluator-owned input") for i, kind in enumerate(kinds))
        group = CandidateGroup(
            task_id, "qwen25-coder-7b-c03e6d3", tuple(item["content_hash"] for item in candidates),
            tests, np.stack(matrix), prompt=problem["prompt"], candidate_texts=tuple(texts),
        )
        groups.append(group); labels[group.split_group] = np.asarray(full_labels)
        print(json.dumps({"task_id": task_id, "cheap_pass_rate": float(np.mean(matrix)), "full_passes": int(sum(full_labels))}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl, npz = args.output_dir / "bank.jsonl", args.output_dir / "bank.npz"
    write_bank(jsonl, npz, groups)
    digest = bank_hash(groups)
    sidecar = {"bank_hash": digest, "labels": [
        {"problem_id": p, "checkpoint_id": c, "values": values.tolist()}
        for (p, c), values in sorted(labels.items())
    ]}
    (args.output_dir / "sidecar.json").write_text(json.dumps(sidecar, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"bank_hash": digest, "tasks": len(groups)}))


if __name__ == "__main__":
    main()
