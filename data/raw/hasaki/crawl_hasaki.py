"""
Hasaki Category Crawler (Playwright version)
Crawl products from 3 main categories:
- Sức khỏe làm đẹp (c3)
- Mỹ phẩm high end (c1907)
- Trang điểm (c23)

Output: JSON + CSV
"""

import json
import csv
import time
import re
from datetime import datetime
from playwright.sync_api import sync_playwright


class HasakiCrawler:
    def __init__(self):
        self.base_url = "https://hasaki.vn"
        self.products = []
        self.qa_data = []

    def crawl_category(self, page, category_path: str, category_name: str, num_pages: int = 2):
        """Crawl products from a category page."""
        products = []

        for pg in range(1, num_pages + 1):
            url = f"{self.base_url}/danh-muc/{category_path}.html"
            if pg > 1:
                url += f"?page={pg}"

            print(f"  Page {pg}: {url}")
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(3)

            # Scroll to load lazy content
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)

            # Get all product links from the page
            product_links = page.evaluate("""
                () => {
                    const links = [];
                    // Try multiple selectors
                    const selectors = [
                        'a[href*="/san-pham/"]',
                        'a[href*="/nhan-hang/"]',
                        '.product a',
                        '[class*="product"] a',
                        '[class*="Product"] a',
                    ];
                    const seen = new Set();
                    for (const sel of selectors) {
                        document.querySelectorAll(sel).forEach(a => {
                            const href = a.href;
                            if (href && href.includes('/san-pham/') && !seen.has(href)) {
                                seen.add(href);
                                // Try to get product name from nearby elements
                                const name = a.textContent.trim().substring(0, 200) ||
                                             a.getAttribute('title') || '';
                                // Try to get image
                                const img = a.querySelector('img');
                                const imgSrc = img ? (img.src || img.dataset.src || '') : '';
                                links.push({href, name, imgSrc});
                            }
                        });
                    }
                    return links;
                }
            """)

            if not product_links:
                print(f"    No product links found, trying broader search...")
                # Fallback: get all links
                all_links = page.evaluate("""
                    () => {
                        const links = [];
                        document.querySelectorAll('a').forEach(a => {
                            if (a.href && a.href.includes('hasaki.vn') && a.href !== window.location.href) {
                                links.push(a.href);
                            }
                        });
                        return [...new Set(links)].slice(0, 20);
                    }
                """)
                print(f"    Sample links on page: {all_links[:5]}")
                break

            print(f"    Found {len(product_links)} product links")

            for link in product_links:
                product_id = self._extract_product_id(link["href"])
                products.append({
                    "product_id": product_id,
                    "product_name": link["name"].split("\n")[0].strip() if link["name"] else "",
                    "product_url": link["href"],
                    "category_name": category_name,
                    "category_path": category_path,
                    "image_url": link["imgSrc"],
                    "crawled_at": datetime.now().isoformat(),
                })

            time.sleep(2)

        return products

    def crawl_product_detail(self, page, product_url: str, product_id: str):
        """Crawl detail + Q&A from a single product page."""
        try:
            page.goto(product_url, wait_until="networkidle", timeout=30000)
            time.sleep(2)

            detail = page.evaluate("""
                () => {
                    const getText = (sel) => {
                        const el = document.querySelector(sel);
                        return el ? el.textContent.trim() : '';
                    };

                    // Price
                    let price = '';
                    for (const sel of ['[class*="price"]', '[class*="Price"]', '.product-price', 'span[class*="price"]']) {
                        const el = document.querySelector(sel);
                        if (el && el.textContent.match(/\\d/)) {
                            price = el.textContent.trim();
                            break;
                        }
                    }

                    // Brand
                    let brand = '';
                    for (const sel of ['[class*="brand"]', '[class*="Brand"]', 'a[href*="/thuong-hieu/"]']) {
                        const el = document.querySelector(sel);
                        if (el) {
                            brand = el.textContent.trim();
                            break;
                        }
                    }

                    // Description
                    let description = '';
                    for (const sel of ['[class*="description"]', '[class*="Description"]', '.product-description', '[class*="detail"]']) {
                        const el = document.querySelector(sel);
                        if (el && el.textContent.trim().length > 20) {
                            description = el.textContent.trim().substring(0, 1000);
                            break;
                        }
                    }

                    // Images
                    const images = [];
                    document.querySelectorAll('img[src*="hasaki"]').forEach(img => {
                        const src = img.src || img.dataset.src;
                        if (src && !images.includes(src)) images.push(src);
                    });

                    // Q&A
                    const qaItems = [];
                    // Try finding Q&A containers
                    const qaSelectors = [
                        '[class*="qa"]', '[class*="QA"]', '[class*="question"]',
                        '[class*="Question"]', '[class*="hoi-dap"]', '[class*="review"]'
                    ];
                    for (const sel of qaSelectors) {
                        document.querySelectorAll(sel).forEach(el => {
                            const text = el.textContent.trim();
                            if (text.length > 10 && text.length < 1000) {
                                qaItems.push(text);
                            }
                        });
                    }

                    return {price, brand, description, images: images.slice(0, 10), qaItems: qaItems.slice(0, 20)};
                }
            """)

            return detail

        except Exception as e:
            print(f"    Error crawling detail: {e}")
            return None

    def _extract_product_id(self, url: str) -> str:
        match = re.search(r'/san-pham/([^/?#]+?)(?:\.html)?(?:\?|$|#)', url)
        if match:
            return match.group(1)
        return url.split("/")[-1].replace(".html", "")

    def run(self, pages_per_category=2, detail_limit=5):
        """Run the full crawl."""
        categories = [
            ("suc-khoe-lam-dep-c3", "Sức khỏe làm đẹp"),
            ("my-pham-high-end-c1907", "Mỹ phẩm high end"),
            ("trang-diem-c23", "Trang điểm"),
        ]

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                locale="vi-VN",
            )
            page = context.new_page()

            for cat_path, cat_name in categories:
                print(f"\n{'='*60}")
                print(f"CATEGORY: {cat_name}")
                print(f"{'='*60}")

                products = self.crawl_category(page, cat_path, cat_name, num_pages=pages_per_category)
                self.products.extend(products)
                print(f"  Total: {len(products)} products")

                # Crawl detail for first N products
                print(f"\n  Crawling details for top {detail_limit} products...")
                for i, prod in enumerate(products[:detail_limit]):
                    print(f"    [{i+1}/{detail_limit}] {prod['product_name'][:60]}...")
                    detail = self.crawl_product_detail(page, prod["product_url"], prod["product_id"])
                    if detail:
                        prod["price"] = detail["price"]
                        prod["brand"] = detail["brand"]
                        prod["description"] = detail["description"]
                        prod["image_urls"] = detail["images"]

                        # Save Q&A
                        for qa_text in detail["qaItems"]:
                            self.qa_data.append({
                                "product_id": prod["product_id"],
                                "product_name": prod["product_name"],
                                "qa_text": qa_text,
                                "product_url": prod["product_url"],
                                "crawled_at": datetime.now().isoformat(),
                            })

                    time.sleep(2)

            browser.close()

    def save(self, output_dir="."):
        """Save results to JSON and CSV."""
        # Products JSON
        products_file = f"{output_dir}/hasaki_products.json"
        with open(products_file, "w", encoding="utf-8") as f:
            json.dump(self.products, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(self.products)} products -> {products_file}")

        # Products CSV
        if self.products:
            csv_file = f"{output_dir}/hasaki_products.csv"
            keys = ["product_id", "product_name", "product_url", "category_name", "brand", "price", "image_url", "crawled_at"]
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.products)
            print(f"Saved {len(self.products)} products -> {csv_file}")

        # Q&A JSON
        if self.qa_data:
            qa_file = f"{output_dir}/hasaki_qa.json"
            with open(qa_file, "w", encoding="utf-8") as f:
                json.dump(self.qa_data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(self.qa_data)} Q&A items -> {qa_file}")

    def print_summary(self):
        print(f"\n{'='*60}")
        print("CRAWL SUMMARY")
        print(f"{'='*60}")
        print(f"Total Products: {len(self.products)}")
        print(f"Total Q&A Items: {len(self.qa_data)}")

        if self.products:
            cats = {}
            for p in self.products:
                cats[p["category_name"]] = cats.get(p["category_name"], 0) + 1
            print("\nBy Category:")
            for cat, count in cats.items():
                print(f"  - {cat}: {count}")


if __name__ == "__main__":
    print("HASAKI CRAWLER (Playwright)")
    print("Categories: Sức khỏe làm đẹp, Mỹ phẩm high end, Trang điểm\n")

    crawler = HasakiCrawler()
    crawler.run(pages_per_category=2, detail_limit=5)
    crawler.save(output_dir=".")
    crawler.print_summary()

    print("\nDone!")
