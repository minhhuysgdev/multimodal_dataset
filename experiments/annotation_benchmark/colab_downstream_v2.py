#!/usr/bin/env python3
"""Colab / local runner for downstream RQ2 v2 (marginal decoding).

Wraps run_downstream.py with presets, per-model batch/epochs/grad-accum,
--append, and --decoding. Designed to be called from the notebook:

  # In notebooks/downstream_colab_pro_v2.ipynb (after mount + paths):
  !python experiments/annotation_benchmark/colab_downstream_v2.py --preset smoke

  # Full LoRA sweep:
  !python experiments/annotation_benchmark/colab_downstream_v2.py --preset lora

  # Custom:
  !python experiments/annotation_benchmark/colab_downstream_v2.py \\
      --classifiers qwen3 --methods verified_d2,verified_d2_clean \\
      --epochs 10 --decoding marginal --append
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


PRESETS: dict[str, dict] = {
    "smoke": {
        "CLASSIFIERS": "tfidf,phobert",
        "METHODS": "verified_d2",
        "EPOCHS": 1,
    },
    "rq2_light": {
        "CLASSIFIERS": "tfidf,phobert,xlmr",
        "METHODS": "all",
        "EPOCHS": 10,
        "OUTDIR": "experiments/annotation_benchmark/results/downstream_v2_reeval",
    },
    "rq2_qwen": {
        "CLASSIFIERS": "qwen3",
        "METHODS": "all",
        "EPOCHS": 10,
        "OUTDIR": "experiments/annotation_benchmark/results/downstream_v2_reeval",
    },
    "rq2_mistral": {
        "CLASSIFIERS": "mistral",
        "METHODS": "all",
        "EPOCHS": 10,
        "OUTDIR": "experiments/annotation_benchmark/results/downstream_v2_reeval",
    },
    "rq2_full": {
        "CLASSIFIERS": "tfidf,phobert,xlmr,qwen3,mistral",
        "METHODS": "all",
        "EPOCHS": 10,
        "OUTDIR": "experiments/annotation_benchmark/results/downstream_v2_reeval",
    },
    "tfidf_all": {
        "CLASSIFIERS": "tfidf",
        "METHODS": "all",
        "EPOCHS": 5,
    },
    "light": {
        "CLASSIFIERS": "tfidf,phobert,xlmr",
        "METHODS": "all",
        "EPOCHS": 10,
    },
    "lora": {
        "CLASSIFIERS": "qwen3,mistral",
        "METHODS": "all",
        "EPOCHS": 10,
    },
    "all": {
        "CLASSIFIERS": "tfidf,phobert,xlmr,qwen3,mistral",
        "METHODS": "all",
        "EPOCHS": 10,
    },
    "qwen_only": {
        "CLASSIFIERS": "qwen3",
        "METHODS": "all",
        "EPOCHS": 10,
    },
    "cvhil_clean": {
        "CLASSIFIERS": "tfidf,phobert,xlmr,qwen3",
        "METHODS": "verified_d2,verified_d2_clean",
        "EPOCHS": 10,
    },
    "custom": {
        "CLASSIFIERS": "qwen3,mistral",
        "METHODS": "all",
        "EPOCHS": 10,
    },
}

BATCH_BY_CLF = {
    "tfidf": 0,
    "phobert": 16,
    "xlmr": 16,
    # T4-friendly defaults (A100 có thể tăng lên)
    "qwen3": 4,
    "mistral": 1,
}

EPOCHS_BY_CLF = {
    "tfidf": 0,
    "phobert": None,
    "xlmr": None,
    "qwen3": 10,
    "mistral": 10,
}

GRAD_ACCUM_BY_CLF = {
    "tfidf": 0,
    "phobert": 1,
    "xlmr": 1,
    "qwen3": 4,   # effective batch ≈ 16 với batch=4
    "mistral": 8, # effective batch ≈ 8 với batch=1
}

ALL_CLFS = ["tfidf", "phobert", "xlmr", "qwen3", "mistral"]
BASE_METHODS = [
    "zero_shot",
    "few_shot",
    "cot",
    "freeal",
    "candist",
    "verified_d2",
]

# Train file mapping (verified_d2 uses clean approved train, not *_dataset.json)
TRAIN_FILE = {
    "verified_d2": "d2_verified_train.json",
    "verified_d2_clean": "d2_verified_train.json",
}

DEFAULT_RQ2_OUTDIR = "experiments/annotation_benchmark/results/downstream_v2_reeval"


def resolve_repo_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("INTENT_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if Path("/content/drive/MyDrive/intent_kb").exists():
        return Path("/content/drive/MyDrive/intent_kb")
    # scripts/…/colab_downstream_v2.py → parents[2] = repo root
    here = Path(__file__).resolve()
    return here.parents[2]


def train_dataset_path(datasets_dir: Path, method: str) -> Path:
    name = TRAIN_FILE.get(method, f"{method}_dataset.json")
    return datasets_dir / name


def audit_verified_d2(datasets_dir: Path) -> None:
    path = datasets_dir / "verified_d2_dataset.json"
    if not path.exists():
        print("⚠️ Missing:", path)
        return
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("results", doc)
    paths = [
        (
            str((r.get("intent_3_level") or {}).get("level_1") or ""),
            str((r.get("intent_3_level") or {}).get("level_2") or ""),
            str((r.get("intent_3_level") or {}).get("level_3") or ""),
        )
        for r in rows
    ]
    l3_counts = Counter(p[2] for p in paths if p[2])
    print("verified_d2 rows       :", len(rows))
    print("unique full paths      :", len(set(paths)))
    print("unique L3              :", len(l3_counts))
    print("singleton L3 (count=1) :", sum(n == 1 for n in l3_counts.values()))
    if len(l3_counts) > 250:
        print("⚠️ CẢNH BÁO: số L3 lớn bất thường; dataset có thể gồm non-approved labels.")


def check_paths(repo: Path, methods: list[str]) -> None:
    datasets = repo / "data/datasets"
    d3 = repo / "data/gold/d3_dataset.json"
    print("REPO_ROOT:", repo)
    print("D3:", "✅" if d3.exists() else "❌", d3)
    for m in methods:
        p = train_dataset_path(datasets, m)
        print(f"  {'✅' if p.exists() else '❌'} {m} → {p.name}")


def run_jobs(
    repo: Path,
    *,
    classifiers: str,
    methods_arg: str,
    epochs: int,
    decoding: str,
    append: bool,
    outdir: Path | None,
    dry_run: bool = False,
    use_l1_head: bool = True,
    lambda_l1: float = 1.0,
    no_append: bool = False,
) -> tuple[list[str], list[str]]:
    results = outdir or (repo / "experiments/annotation_benchmark/results")
    results.mkdir(parents=True, exist_ok=True)
    log_dir = results / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if methods_arg == "all":
        method_list = list(BASE_METHODS)
    else:
        method_list = [m.strip() for m in methods_arg.split(",") if m.strip()]

    clf_list = [c.strip() for c in classifiers.split(",") if c.strip()]
    order = {k: i for i, k in enumerate(ALL_CLFS)}
    clf_list = sorted(clf_list, key=lambda c: order.get(c, 99))

    runner = repo / "experiments/annotation_benchmark/run_downstream.py"
    passed: list[str] = []
    failed: list[str] = []

    print(f"Jobs: {len(method_list) * len(clf_list)}")
    print(f"  methods  = {method_list}")
    print(f"  clfs     = {clf_list}")
    print(f"  decoding = {decoding}")
    print(f"  l1_head  = {use_l1_head} (lambda={lambda_l1})")
    print(f"  append   = {append and not no_append}")
    print(f"  outdir   = {results}")
    print("=" * 70)

    for clf in clf_list:
        ep = EPOCHS_BY_CLF.get(clf)
        if ep is None:
            ep = epochs
        bs = BATCH_BY_CLF.get(clf, 8)
        ga = GRAD_ACCUM_BY_CLF.get(clf, 1)

        for method in method_list:
            name = f"{method}__{clf}"
            job_log = log_dir / f"colab_{name}_{stamp}.log"
            cmd = [
                sys.executable,
                str(runner),
                "--method",
                method,
                "--classifiers",
                clf,
                "--decoding",
                decoding,
                "--outdir",
                str(results),
            ]
            if use_l1_head:
                cmd.append("--l1-head")
            else:
                cmd.append("--no-l1-head")
            cmd += ["--lambda-l1", str(lambda_l1)]
            if clf != "tfidf":
                cmd += ["--epochs", str(ep)]
                if bs > 0:
                    cmd += ["--batch-size", str(bs)]
                if ga > 0:
                    cmd += ["--grad-accum", str(ga)]
            if append and not no_append:
                cmd.append("--append")

            print(
                f"\n>>> {name}  epochs={ep if clf != 'tfidf' else '-'} "
                f"batch={bs if clf != 'tfidf' else '-'} "
                f"grad_accum={ga if clf != 'tfidf' else '-'}"
            )
            print("    ", " ".join(cmd))
            if dry_run:
                print("    (dry-run)")
                passed.append(name)
                continue

            t0 = time.time()
            with job_log.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(repo)
                )
            dur = int(time.time() - t0)
            if proc.returncode == 0:
                print(f"<<< OK {name} ({dur}s)")
                passed.append(name)
            else:
                print(f"<<< FAIL {name} ({dur}s) — xem {job_log}")
                try:
                    for line in job_log.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-8:]:
                        print("   ", line)
                except OSError:
                    pass
                failed.append(name)

    print("\n" + "=" * 70)
    print("PASS:", passed or "—")
    print("FAIL:", failed or "—")
    print("Results:", results / "downstream_performance.csv")
    return passed, failed


def check_upload(repo: Path) -> int:
    """Print ✅/❌ for files required by Colab v2. Return number of missing required files."""
    required = [
        ("B-code", "experiments/annotation_benchmark/colab_downstream_v2.py"),
        ("B-code", "experiments/annotation_benchmark/run_downstream.py"),
        ("B-code", "experiments/annotation_benchmark/common.py"),
        ("B-code", "experiments/evaluation/utils.py"),
        ("B-code", "experiments/evaluation/train_tfidf_svm.py"),
        ("B-code", "experiments/evaluation/train_transformer.py"),
        ("B-code", "experiments/evaluation/lora_utils.py"),
        ("B-code", "experiments/phase5_pipeline/pipeline_utils.py"),
        ("C-data", "data/datasets/zero_shot_dataset.json"),
        ("C-data", "data/datasets/few_shot_dataset.json"),
        ("C-data", "data/datasets/cot_dataset.json"),
        ("C-data", "data/datasets/freeal_dataset.json"),
        ("C-data", "data/datasets/candist_dataset.json"),
        ("C-data", "data/datasets/verified_d2_dataset.json"),
        ("C-data", "data/datasets/d2_verified_train.json"),
        ("C-data", "data/gold/d3_dataset.json"),
        ("C-data", "data/splits/d3_val_ids.txt"),
        ("C-data", "data/splits/d3_test_ids.txt"),
        ("A-nb", "notebooks/downstream_colab_pro_v2.ipynb"),
    ]
    optional = [
        ("D-opt", "data/unified_intents.csv"),
        ("D-opt", "data/intent_kb.intent_edges_28_05_2026.csv"),
        ("A-nb", "notebooks/COLAB_V2_UPLOAD_NOTE.md"),
    ]
    missing = 0
    print("REPO:", repo)
    print("=== REQUIRED (B+C+A) ===")
    for group, rel in required:
        path = repo / rel
        ok = path.exists()
        if not ok:
            missing += 1
        print(f"  {'✅' if ok else '❌'} [{group}] {rel}")
    print("=== OPTIONAL (D) ===")
    for group, rel in optional:
        path = repo / rel
        print(f"  {'✅' if path.exists() else '⚠️'} [{group}] {rel}")
    print()
    if missing:
        print(f"❌ Missing {missing} required file(s). Sync Drive rồi chạy lại.")
        print("   Xem note: notebooks/COLAB_V2_UPLOAD_NOTE.md")
    else:
        print("✅ Upload đủ để chạy downstream v2.")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="rq2_light",
        help="Job preset (overridden by --classifiers/--methods if set)",
    )
    parser.add_argument("--classifiers", default="", help="Override preset classifiers")
    parser.add_argument("--methods", default="", help="Override preset methods")
    parser.add_argument("--epochs", type=int, default=0, help="Override preset epochs (0=preset)")
    parser.add_argument(
        "--decoding",
        choices=["argmax", "marginal"],
        default="marginal",
    )
    parser.add_argument(
        "--l1-head",
        action="store_true",
        default=True,
        help="Pass --l1-head to run_downstream (default: on)",
    )
    parser.add_argument(
        "--no-l1-head",
        action="store_true",
        help="Pass --no-l1-head to run_downstream",
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=1.0,
        help="L1 auxiliary loss weight for transformers",
    )
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Do not pass --append (fresh CSV for first stage)",
    )
    parser.add_argument("--repo-root", type=str, default="")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Results dir (default: experiments/annotation_benchmark/results)",
    )
    parser.add_argument("--audit-only", action="store_true", help="Only print dataset audit")
    parser.add_argument(
        "--check-upload",
        action="store_true",
        help="Verify required Colab upload files then exit",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = resolve_repo_root(args.repo_root or None)
    os.chdir(repo)
    sys.path.insert(0, str(repo / "experiments/annotation_benchmark"))
    sys.path.insert(0, str(repo / "experiments/evaluation"))

    if args.check_upload:
        missing = check_upload(repo)
        raise SystemExit(1 if missing else 0)

    cfg = dict(PRESETS[args.preset])
    classifiers = args.classifiers.strip() or cfg["CLASSIFIERS"]
    methods_arg = args.methods.strip() or cfg["METHODS"]
    epochs = args.epochs if args.epochs > 0 else int(cfg["EPOCHS"])
    use_l1_head = not args.no_l1_head
    preset_outdir = cfg.get("OUTDIR")
    resolved_outdir = args.outdir
    if resolved_outdir is None and preset_outdir:
        resolved_outdir = repo / preset_outdir
    append = bool(args.append)
    if args.preset.startswith("rq2_") and not args.no_append and not args.append:
        # Stage B/C: user passes --append explicitly
        append = args.preset not in ("rq2_light", "rq2_full")

    print("PRESET     :", args.preset)
    print("CLASSIFIERS:", classifiers)
    print("METHODS    :", methods_arg)
    print("EPOCHS     :", epochs)
    print("DECODING   :", args.decoding)
    print("L1_HEAD    :", use_l1_head, f"(lambda={args.lambda_l1})")
    print("APPEND     :", append and not args.no_append)
    print("OUTDIR     :", resolved_outdir or (repo / "experiments/annotation_benchmark/results"))
    check_paths(repo, BASE_METHODS)
    print()
    audit_verified_d2(repo / "data/datasets")
    if args.audit_only:
        return

    run_jobs(
        repo,
        classifiers=classifiers,
        methods_arg=methods_arg,
        epochs=epochs,
        decoding=args.decoding,
        append=append,
        outdir=resolved_outdir,
        dry_run=args.dry_run,
        use_l1_head=use_l1_head,
        lambda_l1=args.lambda_l1,
        no_append=args.no_append,
    )


if __name__ == "__main__":
    main()
