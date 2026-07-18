#!/usr/bin/env python3
\"\"\"Generate verifier summary image and JSON from a labelled export.

Usage:
  python scripts/generate_verifier_report.py --input data/raw/hasaki_labelled_clean.json --outdir data/audit

The script looks for verifier fields in the exported labelled JSON:
  - verification (dict) with key 'decision' OR
  - verifier_decision / verifier_reason / confidence_before_verify at top-level of each result

If no verifier data is found, the script simulates numbers using conservative defaults
so you can preview the report layout.
\"\"\"
from __future__ import annotations
import argparse
import json
from pathlib import Path
import math
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    print(\"Missing plotting dependencies. Install pandas, matplotlib, seaborn.\", file=sys.stderr)
    raise


def load_results(path: Path):
    with path.open(\"r\", encoding=\"utf-8\") as f:
        doc = json.load(f)
    # support both {results: [...]} and raw list
    if isinstance(doc, dict) and \"results\" in doc:
        rows = doc[\"results\"]
    elif isinstance(doc, list):
        rows = doc
    else:
        # try common keys fallback
        rows = doc.get(\"results\") if isinstance(doc, dict) else []
    return rows


def summarize(rows: list[dict]):
    import pandas as pd
    df = pd.DataFrame(rows)
    # normalize verifier decision
    if \"verification\" in df.columns:
        df[\"verifier_decision\"] = df[\"verification\"].apply(lambda v: (v or {}).get(\"decision\") if isinstance(v, dict) else None)
    else:
        if \"verifier_decision\" not in df.columns:
            df[\"verifier_decision\"] = None
    total = len(df)
    verified = int(df[\"verifier_decision\"].notna().sum()) if total else 0
    retained = int((df[\"verifier_decision\"] == \"RETAIN\").sum()) if total else 0
    revised = int((df[\"verifier_decision\"] == \"REVISE\").sum()) if total else 0
    return {
        \"total\": total,
        \"verified\": verified,
        \"retained\": retained,
        \"revised\": revised,
        \"df\": df,
    }


def plot_and_save(summary: dict, out_dir: Path, source_name: str, simulate: bool = False):
    sns.set(style=\"whitegrid\")
    total = summary[\"total\"]
    verified = summary[\"verified\"]
    retained = summary[\"retained\"]
    revised = summary[\"revised\"]

    if verified == 0 and simulate:
        # reasonable simulation: verify ~15% of data, retain 80% of verified
        verified = max(1, int(total * 0.15))
        retained = int(math.floor(verified * 0.80))
        revised = verified - retained
        simulated = True
    else:
        simulated = False

    verify_coverage = verified / total if total else 0.0
    retain_rate = retained / verified if verified else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    axes[0].bar([\"total\"], [total], color=\"#777777\", label=\"total\")\n+    axes[0].bar([\"verified\"], [verified], color=\"#2b83ba\", label=\"verified\")\n+    axes[0].bar([\"retained\"], [retained], color=\"#66c2a5\", label=\"retained\")\n+    axes[0].bar([\"revised\"], [revised], color=\"#fc8d62\", label=\"revised\")\n+    axes[0].set_title(\"Counts: total / verified / retained / revised\")\n+    for i, v in enumerate([total, verified, retained, revised]):\n+        axes[0].text(i, v + max(1, total * 0.01), str(v), ha=\"center\")\n+\n+    # Pie chart for decisions\n+    labels = [\"Retain\", \"Revise\"]\n+    sizes = [retained, revised]\n+    axes[1].pie(sizes, labels=labels, autopct=\"%1.1f%%\", colors=[\"#66c2a5\", \"#fc8d62\"], startangle=140, textprops={\"fontsize\": 12})\n+    axes[1].set_title(f\"Verifier decisions (coverage={verify_coverage:.1%}, retained={retain_rate:.1%})\")\n+\n+    plt.suptitle(f\"Verifier summary — {source_name}\" + (\" (SIMULATED)\" if simulated else \"\"))\n+    plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n+\n+    png_path = out_dir / f\"verify_summary_{source_name.replace('/', '_')}.png\"\n+    fig.savefig(png_path, dpi=150)\n+    plt.close(fig)\n+\n+    summary_json = {\n+        \"source\": source_name,\n+        \"total\": total,\n+        \"verified\": verified,\n+        \"retained\": retained,\n+        \"revised\": revised,\n+        \"verify_coverage\": verify_coverage,\n+        \"retain_rate\": retain_rate,\n+        \"simulated\": simulated,\n+    }\n+    json_path = out_dir / f\"verify_summary_{source_name.replace('/', '_')}.json\"\n+    with json_path.open(\"w\", encoding=\"utf-8\") as f:\n+        json.dump(summary_json, f, ensure_ascii=False, indent=2)\n+\n+    return png_path, json_path\n+\n+\n+def main():\n+    parser = argparse.ArgumentParser()\n+    parser.add_argument(\"--input\", \"-i\", required=True, help=\"labelled JSON input file (export)\")\n+    parser.add_argument(\"--outdir\", \"-o\", default=\"data/audit\", help=\"output directory for image+json\")\n+    parser.add_argument(\"--simulate-if-empty\", action=\"store_true\", help=\"simulate verifier numbers if none present\")\n+    args = parser.parse_args()\n+\n+    inp = Path(args.input)\n+    if not inp.exists():\n+        print(f\"Input file not found: {inp}\", file=sys.stderr)\n+        sys.exit(2)\n+\n+    rows = load_results(inp)\n+    summary = summarize(rows)\n+    png, js = plot_and_save(summary, Path(args.outdir), inp.name, simulate=args.simulate_if_empty)\n+    print(\"Saved:\", png, js)\n+\n+\n+if __name__ == \"__main__\":\n+    main()\n+\n*** End Patch"}
