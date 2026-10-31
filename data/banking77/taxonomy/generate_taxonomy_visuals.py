#!/usr/bin/env python3
"""Generate HTML and Mermaid diagrams for taxonomy review."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIAGRAMS = ROOT / "diagrams"


def _l1_order() -> list[str]:
    return [
        "card",
        "cash_withdrawal",
        "transfer",
        "top_up",
        "currency_exchange",
        "refund_and_charges",
        "identity_verification",
        "account_and_access",
    ]


def write_mermaid_overview(rows: list[dict], out_path: Path) -> None:
    """High-level BANKING -> L1 -> L2 (counts only)."""
    l2_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    l1_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        l1_counts[r["L1"]] += 1
        l2_counts[r["L1"]][r["L2"]] += 1

    lines = ["flowchart TB", "  banking[\"BANKING root\"]"]
    for l1 in _l1_order():
        if l1 not in l1_counts:
            continue
        nid = l1.replace("_", "")
        lines.append(f'  banking --> {nid}["{l1}<br/>{l1_counts[l1]} L3"]')
        for l2, cnt in sorted(l2_counts[l1].items()):
            l2id = f"{nid}_{l2.replace('_', '')[:24]}"
            lines.append(f'  {nid} --> {l2id}["{l2}<br/>{cnt} intents"]')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mermaid_l1_detail(rows: list[dict], l1: str, out_path: Path) -> None:
    """L1 -> L2 -> L3 for one product line."""
    subset = [r for r in rows if r["L1"] == l1]
    if not subset:
        return
    l1_id = l1.replace("_", "")
    lines = ["flowchart TB", f'  root["{l1}"]']
    l2_groups: dict[str, list[dict]] = defaultdict(list)
    for r in subset:
        l2_groups[r["L2"]].append(r)

    for l2, items in sorted(l2_groups.items()):
        l2_id = f"l2_{l2.replace('_', '')[:20]}"
        lines.append(f'  root --> {l2_id}["{l2}"]')
        for r in sorted(items, key=lambda x: x["label_id"]):
            l3 = r["L3"].replace("?", "\\?")
            flag = f" ⚠ {r['flag']}" if r["flag"] else ""
            node = f"l3_{r['label_id']}"
            label = f"{r['label_id']}: {l3}{flag}"
            lines.append(f'  {l2_id} --> {node}["{label}"]')

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _menu_payload(rows: list[dict]) -> dict:
    by_l1: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_l1[r["L1"]][r["L2"]].append(r)
    l1_order = [x for x in _l1_order() if x in by_l1]
    l1_order.extend(sorted(set(by_l1) - set(l1_order)))
    menu = []
    for l1 in l1_order:
        families = []
        for l2 in sorted(by_l1[l1]):
            leaves = sorted(by_l1[l1][l2], key=lambda x: x["label_id"])
            families.append(
                {
                    "name": l2,
                    "intents": [
                        {
                            "id": item["label_id"],
                            "name": item["L3"],
                            "flag": item["flag"],
                            "rationale": item["rationale"],
                        }
                        for item in leaves
                    ],
                }
            )
        menu.append({"name": l1, "families": families})
    return {"levels": menu}


def write_review_html(rows: list[dict], audit: dict, out_path: Path) -> None:
    """Cascading menu: select L1, then L2, then list L3."""
    payload = _menu_payload(rows)
    flagged = audit.get("flagged_count", 0)
    data_json = json.dumps(payload, ensure_ascii=False)
    doc = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>BANKING77 taxonomy review</title>
  <style>
    :root {{
      --bg: #f8f5f0;
      --fg: #3c3732;
      --l1: #6c4f3b;
      --accent: #c58c32;
      --flag-bg: #fff3cd;
      --border: #ddd5c8;
      --active: #efe6d8;
    }}
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--fg);
      margin: 0;
      padding: 1rem 1.25rem 2rem;
      line-height: 1.4;
    }}
    h1 {{ font-size: 1.25rem; color: var(--l1); margin: 0 0 0.35rem; }}
    .meta {{ font-size: 0.85rem; margin: 0 0 0.75rem; }}
    .crumb {{
      font-size: 0.9rem; margin: 0 0 0.75rem; min-height: 1.3em;
    }}
    .crumb span {{ color: var(--l1); font-weight: 600; }}
    .toolbar {{
      display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center;
      margin-bottom: 0.85rem;
    }}
    #q {{
      flex: 1; min-width: 14rem; padding: 0.45rem 0.6rem;
      border: 1px solid var(--border); border-radius: 6px;
    }}
    .menu {{
      display: grid;
      grid-template-columns: minmax(11rem, 0.9fr) minmax(13rem, 1fr) minmax(16rem, 1.4fr);
      gap: 0.6rem;
      align-items: start;
    }}
    .col {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 8px;
      min-height: 18rem;
      overflow: auto;
      max-height: 70vh;
    }}
    .col h2 {{
      margin: 0; padding: 0.55rem 0.75rem;
      font-size: 0.75rem; letter-spacing: 0.04em;
      text-transform: uppercase; color: var(--l1);
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0; background: #fff;
    }}
    button.item {{
      display: flex; width: 100%; text-align: left;
      justify-content: space-between; gap: 0.5rem;
      border: 0; background: transparent; color: inherit;
      font: inherit; padding: 0.55rem 0.75rem; cursor: pointer;
    }}
    button.item:hover {{ background: #faf7f2; }}
    button.item.active {{ background: var(--active); font-weight: 600; }}
    button.item .count {{ color: #7a7066; font-weight: 400; font-size: 0.8rem; }}
    .leaf {{
      padding: 0.55rem 0.75rem; border-bottom: 1px solid #f0ebe4;
      font-size: 0.9rem;
    }}
    .leaf code {{
      background: #eee8df; padding: 0.05rem 0.35rem; border-radius: 4px;
      font-size: 0.78rem; margin-right: 0.35rem;
    }}
    .flag {{
      display: inline-block; margin-left: 0.35rem; font-size: 0.72rem;
      background: var(--flag-bg); padding: 0.05rem 0.35rem; border-radius: 4px;
    }}
    .rationale {{ display: block; font-size: 0.78rem; color: #6b635b; margin-top: 0.2rem; }}
    .empty {{ padding: 0.8rem 0.75rem; color: #7a7066; font-size: 0.88rem; }}
    .hits {{ margin: 0 0 0.75rem; font-size: 0.85rem; }}
    .hits button {{
      margin: 0.2rem 0.35rem 0 0; font: inherit; font-size: 0.8rem;
      border: 1px solid var(--border); background: #fff; border-radius: 999px;
      padding: 0.2rem 0.55rem; cursor: pointer;
    }}
    @media (max-width: 820px) {{
      .menu {{ grid-template-columns: 1fr; }}
      .col {{ max-height: none; min-height: 0; }}
    }}
  </style>
</head>
<body>
  <h1>BANKING77 taxonomy</h1>
  <p class="meta">Menu L1 → L2 → L3 · 77 intents · {flagged} intent có flag</p>
  <div class="toolbar">
    <input id="q" type="search" placeholder="Tìm L3, L2 hoặc id…" autocomplete="off"/>
    <label><input type="checkbox" id="flagsOnly"/> Chỉ hiện intent có flag</label>
  </div>
  <p class="crumb" id="crumb"></p>
  <div class="hits" id="hits" hidden></div>
  <div class="menu">
    <section class="col" aria-label="Level 1">
      <h2>Level 1</h2>
      <div id="col1"></div>
    </section>
    <section class="col" aria-label="Level 2">
      <h2>Level 2</h2>
      <div id="col2"></div>
    </section>
    <section class="col" aria-label="Level 3">
      <h2>Level 3</h2>
      <div id="col3"></div>
    </section>
  </div>
  <script id="taxonomy-data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('taxonomy-data').textContent);
    const col1 = document.getElementById('col1');
    const col2 = document.getElementById('col2');
    const col3 = document.getElementById('col3');
    const crumb = document.getElementById('crumb');
    const q = document.getElementById('q');
    const flagsOnly = document.getElementById('flagsOnly');
    const hits = document.getElementById('hits');
    let selectedL1 = data.levels[0] ? data.levels[0].name : null;
    let selectedL2 = null;
    function esc(value) {{
      return String(value).replace(/[&<>"']/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[ch]));
    }}

    function l1Node(name) {{
      return data.levels.find(x => x.name === name);
    }}
    function countL3(node) {{
      return node.families.reduce((n, f) => n + f.intents.length, 0);
    }}
    function intentVisible(intent) {{
      return !flagsOnly.checked || intent.flag;
    }}
    function familyVisible(family) {{
      return family.intents.some(intentVisible);
    }}

    function render() {{
      const l1 = l1Node(selectedL1);
      if (l1 && !l1.families.some(f => f.name === selectedL2 && familyVisible(f))) {{
        const first = l1.families.find(familyVisible);
        selectedL2 = first ? first.name : null;
      }}
      col1.innerHTML = '';
      data.levels.forEach(node => {{
        const visible = node.families.some(familyVisible);
        if (!visible) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'item' + (node.name === selectedL1 ? ' active' : '');
        const n = node.families.reduce((s, f) => s + f.intents.filter(intentVisible).length, 0);
        btn.innerHTML = '<span>' + esc(node.name) + '</span><span class="count">' + n + '</span>';
        btn.onclick = () => {{ selectedL1 = node.name; selectedL2 = null; render(); }};
        col1.appendChild(btn);
      }});
      col2.innerHTML = '';
      if (!l1) {{
        col2.innerHTML = '<p class="empty">Chọn Level 1</p>';
      }} else {{
        l1.families.filter(familyVisible).forEach(family => {{
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'item' + (family.name === selectedL2 ? ' active' : '');
          const n = family.intents.filter(intentVisible).length;
          btn.innerHTML = '<span>' + esc(family.name) + '</span><span class="count">' + n + '</span>';
          btn.onclick = () => {{ selectedL2 = family.name; render(); }};
          col2.appendChild(btn);
        }});
      }}
      col3.innerHTML = '';
      const family = l1 && l1.families.find(f => f.name === selectedL2);
      if (!family) {{
        col3.innerHTML = '<p class="empty">Chọn Level 2</p>';
      }} else {{
        family.intents.filter(intentVisible).forEach(intent => {{
          const row = document.createElement('div');
          row.className = 'leaf';
          const flag = intent.flag
            ? '<span class="flag">' + esc(intent.flag) + '</span>'
            : '';
          const why = intent.flag && intent.rationale
            ? '<span class="rationale">' + esc(intent.rationale) + '</span>'
            : '';
          row.innerHTML = '<code>' + esc(intent.id) + '</code><span>' + esc(intent.name) + '</span>' + flag + why;
          col3.appendChild(row);
        }});
      }}
      const parts = ['BANKING'];
      if (selectedL1) parts.push(selectedL1);
      if (selectedL2) parts.push(selectedL2);
      crumb.innerHTML = parts.map(p => '<span>' + esc(p) + '</span>').join(' → ');
    }}

    function searchHits(term) {{
      const out = [];
      if (!term) return out;
      data.levels.forEach(l1 => {{
        l1.families.forEach(l2 => {{
          l2.intents.forEach(intent => {{
            if (!intentVisible(intent)) return;
            const blob = (intent.id + ' ' + intent.name + ' ' + l2.name + ' ' + l1.name + ' ' + (intent.flag || '')).toLowerCase();
            if (blob.includes(term)) out.push({{ l1: l1.name, l2: l2.name, intent }});
          }});
        }});
      }});
      return out.slice(0, 12);
    }}

    function onSearch() {{
      const term = (q.value || '').trim().toLowerCase();
      const found = searchHits(term);
      hits.hidden = found.length === 0;
      hits.innerHTML = '';
      if (!term || !found.length) {{ render(); return; }}
      found.forEach(hit => {{
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = hit.l1 + ' / ' + hit.l2 + ' / ' + hit.intent.name;
        btn.onclick = () => {{
          selectedL1 = hit.l1;
          selectedL2 = hit.l2;
          render();
        }};
        hits.appendChild(btn);
      }});
      selectedL1 = found[0].l1;
      selectedL2 = found[0].l2;
      render();
    }}

    q.addEventListener('input', onSearch);
    flagsOnly.addEventListener('change', () => {{ onSearch(); }});
    render();
  </script>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")


def write_review_markdown(rows: list[dict], out_path: Path) -> None:
    """Short doc pointing to HTML + overview mermaid."""
    text = """# BANKING77 taxonomy — sơ đồ review

## Cách xem nhanh nhất

1. Mở file HTML trong trình duyệt (menu L1 → L2 → L3, tìm kiếm, lọc flag):

   [`banking77_taxonomy_review.html`](../banking77_taxonomy_review.html)

2. Sơ đồ tổng quan L1 → L2 (Mermaid): xem [`diagrams/banking77_overview.mmd`](diagrams/banking77_overview.mmd) trên GitHub hoặc paste vào [mermaid.live](https://mermaid.live).

3. Chi tiết từng L1 (L2 → L3 đủ 77 intent): các file `diagrams/banking77_<L1>.mmd`.

## Overview (Mermaid)

```mermaid
"""
    overview = (DIAGRAMS / "banking77_overview.mmd").read_text(encoding="utf-8")
    text += overview.strip() + "\n```\n"
    out_path.write_text(text, encoding="utf-8")


def generate_all(rows: list[dict], audit: dict) -> list[Path]:
    written: list[Path] = []
    html_path = ROOT / "banking77_taxonomy_review.html"
    write_review_html(rows, audit, html_path)
    written.append(html_path)

    overview = DIAGRAMS / "banking77_overview.mmd"
    write_mermaid_overview(rows, overview)
    written.append(overview)

    for l1 in _l1_order():
        path = DIAGRAMS / f"banking77_{l1}.mmd"
        write_mermaid_l1_detail(rows, l1, path)
        written.append(path)

    md_path = ROOT / "TAXONOMY_REVIEW.md"
    write_review_markdown(rows, md_path)
    written.append(md_path)
    return written


if __name__ == "__main__":
    data = json.loads((ROOT / "banking77_hierarchy.json").read_text(encoding="utf-8"))
    paths = generate_all(data["flat"], data["audit"])
    for p in paths:
        print("Wrote", p)
