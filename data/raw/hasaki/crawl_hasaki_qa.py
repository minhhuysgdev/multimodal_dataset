"""
Hasaki Q&A Crawler - Batch parallel version
Chia products thành 4 batches, mỗi batch chạy 1 browser riêng (multiprocessing)
Output: hasaki_questions.json
"""

import json
import time
import os
from datetime import datetime
from multiprocessing import Process
from playwright.sync_api import sync_playwright


def extract_questions_from_page(page, product):
    """Extract câu hỏi khách hàng từ 1 product page."""
    url = product["product_url"]
    pid = product["product_id"]
    name = product.get("product_name", product.get("product_name_raw", ""))

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=12000)
    except Exception:
        return []

    time.sleep(1)

    # Click Q&A tab
    page.evaluate("""
        () => {
            for (const a of document.querySelectorAll('a')) {
                if (a.textContent.trim().startsWith('Hỏi đáp') && a.className.includes('border-b')) {
                    a.click(); break;
                }
            }
        }
    """)
    time.sleep(1)

    # Extract
    raw = page.evaluate("""
        () => {
            const qs = [];
            const h2s = document.querySelectorAll('h2');
            let c = null;
            for (const h of h2s) {
                if (h.textContent.includes('Hỏi đáp')) {
                    c = h.closest('div[class*="shadow"]') || h.parentElement;
                    break;
                }
            }
            if (!c) return qs;

            for (const block of c.querySelectorAll('div[class*="border-b"]')) {
                const top = block.querySelector(':scope > div.text-sm');
                if (!top) continue;
                const userEl = top.querySelector('p.font-bold');
                const user = userEl ? userEl.textContent.trim() : '';
                let q = '';
                for (const p of top.querySelectorAll(':scope > p')) {
                    if (!p.classList.contains('font-bold') && p.textContent.trim().length > 2) {
                        q = p.textContent.trim(); break;
                    }
                }
                const dateEl = top.querySelector('div.text-xs p');
                const date = dateEl ? dateEl.textContent.trim() : '';
                if (q && user && user !== 'Hasaki') qs.push({user, question: q, date});
            }
            return qs;
        }
    """)

    return [{
        "product_id": pid,
        "product_name": name,
        "user": q["user"],
        "question": q["question"],
        "date": q["date"],
    } for q in raw]


def run_batch(batch_id, products, output_file):
    """Chạy 1 batch với 1 browser riêng."""
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="vi-VN",
        )
        page = ctx.new_page()

        for i, prod in enumerate(products):
            try:
                qs = extract_questions_from_page(page, prod)
                results.extend(qs)
            except Exception:
                pass

            if (i + 1) % 20 == 0 or i == len(products) - 1:
                # Save partial result
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"  [Batch {batch_id}] {i+1}/{len(products)} -> {len(results)} questions")

            time.sleep(0.3)

        browser.close()

    # Final save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  [Batch {batch_id}] DONE: {len(results)} questions -> {output_file}")


def main():
    with open("hasaki_products_full.json", "r", encoding="utf-8") as f:
        products = json.load(f)

    NUM_BATCHES = 4
    batch_size = len(products) // NUM_BATCHES + 1
    batches = [products[i:i + batch_size] for i in range(0, len(products), batch_size)]

    print(f"Total: {len(products)} products")
    print(f"Split into {len(batches)} batches: {[len(b) for b in batches]}")
    print(f"Running {len(batches)} browsers in parallel...\n")

    # Start batch processes
    processes = []
    output_files = []

    for i, batch in enumerate(batches):
        output_file = f"hasaki_qa_batch_{i}.json"
        output_files.append(output_file)
        proc = Process(target=run_batch, args=(i, batch, output_file))
        processes.append(proc)
        proc.start()

    # Wait for all
    for proc in processes:
        proc.join()

    # Merge results
    print(f"\nMerging results...")
    all_questions = []
    for f_path in output_files:
        if os.path.exists(f_path):
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_questions.extend(data)
            os.remove(f_path)

    with open("hasaki_questions.json", "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_questions)} questions -> hasaki_questions.json")
    products_with_qa = len(set(q["product_id"] for q in all_questions))
    print(f"Products with Q&A: {products_with_qa}/{len(products)}")

    if all_questions:
        print(f"\nSample questions:")
        for q in all_questions[:10]:
            print(f"  [{q['product_name'][:30]}] {q['question'][:80]}")


if __name__ == "__main__":
    main()
