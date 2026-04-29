"""
Hasaki Full Crawler - Crawl nhiều pages hơn từ 3 categories
Output: hasaki_products_full.json
"""

import json
import csv
import time
import re
from datetime import datetime
from playwright.sync_api import sync_playwright


class HasakiFullCrawler:
    def __init__(self):
        self.base_url = "https://hasaki.vn"
        self.products = []

    def crawl_category(self, page, category_path: str, category_name: str, num_pages: int = 10):
        """Crawl products from a category, nhiều pages."""
        products = []
        seen_urls = set()

        for pg in range(1, num_pages + 1):
            url = f"{self.base_url}/danh-muc/{category_path}.html"
            if pg > 1:
                url += f"?page={pg}"

            print(f"  Page {pg}/{num_pages}: {url}")
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)

            # Scroll to trigger lazy load
            for _ in range(3):
                page.evaluate("window.scrollBy(0, 800)")
                time.sleep(0.5)

            product_links = page.evaluate("""
                () => {
                    const seen = new Set();
                    const links = [];
                    document.querySelectorAll('a[href*="/san-pham/"]').forEach(a => {
                        const href = a.href;
                        if (href && !seen.has(href)) {
                            seen.add(href);
                            const name = a.textContent.trim().substring(0, 300) || a.getAttribute('title') || '';
                            const img = a.querySelector('img');
                            const imgSrc = img ? (img.src || img.dataset.src || '') : '';
                            links.push({href, name, imgSrc});
                        }
                    });
                    return links;
                }
            """)

            if not product_links:
                print(f"    No products found - end of category")
                break

            new_count = 0
            for link in product_links:
                if link["href"] not in seen_urls:
                    seen_urls.add(link["href"])
                    new_count += 1
                    products.append({
                        "product_id": self._extract_id(link["href"]),
                        "product_name_raw": link["name"],
                        "product_url": link["href"],
                        "category_name": category_name,
                        "category_path": category_path,
                        "image_url": link["imgSrc"],
                        "crawled_at": datetime.now().isoformat(),
                    })

            print(f"    Found {len(product_links)} links, {new_count} new (total: {len(products)})")

            if new_count == 0:
                print(f"    No new products - end of category")
                break

            time.sleep(2)

        return products

    def _extract_id(self, url: str) -> str:
        match = re.search(r'/san-pham/([^/?#]+?)(?:\.html)?(?:\?|$|#)', url)
        return match.group(1) if match else url.split("/")[-1].replace(".html", "")

    def clean_product_name(self, raw_name: str) -> dict:
        """Parse raw product name thành name + brand."""
        name = raw_name.split("\n")[0].strip()

        # Remove discount/price prefix: -27 %360.000 ₫490.000 ₫
        name = re.sub(r'^-?\d+\s*%', '', name).strip()
        name = re.sub(r'^\d[\d.,]*\s*₫', '', name).strip()
        name = re.sub(r'^\d[\d.,]*\s*₫', '', name).strip()
        # Remove 'Tặng:...' prefix
        name = re.sub(r'^Tặng:.*?₫', '', name).strip()
        # Remove trailing rating: 4.9(228)79
        name = re.sub(r'\d+\.\d+\(\d+\)\d*$', '', name).strip()
        name = re.sub(r'\d+$', '', name).strip()

        # Detect brand at start
        brands_known = [
            "L'Oreal Professionnel", "La Roche-Posay", "Some By Mi",
            "The Ordinary", "Paula's Choice", "Dear Klairs",
            "CeraVe", "Cocoon", "Klairs", "Bioderma", "Maybelline",
            "DHC", "3CE", "Obagi", "MartiDerm", "Martiderm",
            "Vichy", "Eucerin", "Innisfree", "Neutrogena", "Garnier",
            "Nivea", "Rohto", "Senka", "Skin1004", "Pond's", "SVR",
            "Uriage", "Avene", "Heliocare", "Cetaphil", "Hada Labo",
            "Carslan", "Aperire", "Marvis", "Romand", "Merzy",
            "Espoir", "Peripera", "Laneige", "Sulwhasoo", "SK-II",
            "Shiseido", "Estee Lauder", "Clinique", "MAC",
            "NARS", "YSL", "Dior", "Chanel", "Lancome",
            "Kiehl's", "Origins", "Bobbi Brown", "Tom Ford",
            "Charlotte Tilbury", "Fenty Beauty", "Rare Beauty",
            "ColourPop", "NYX", "Wet n Wild", "Revlon", "Catrice",
            "Essence", "Heimish", "Torriden", "Mediheal",
            "Anessa", "Biore", "Sunplay", "Missha", "Etude",
            "Holika Holika", "COSRX", "Banila Co", "Mamonde",
        ]

        detected_brand = ""
        for brand in sorted(brands_known, key=len, reverse=True):
            if name.startswith(brand):
                detected_brand = brand
                name = name[len(brand):].strip()
                break

        if not detected_brand:
            # Try CamelCase pattern
            m = re.match(r'^([A-Z][a-z]+(?:[A-Z][a-z]+)+)', name)
            if m:
                detected_brand = m.group(1)
                name = name[len(detected_brand):].strip()

        # Remove English suffix after ml/g
        name = re.sub(r'(\d+(?:ml|g|L))[A-Z][a-z].*$', r'\1', name)

        return {"product_name": name, "brand": detected_brand}

    def run(self, pages_per_category=10):
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
            pg = context.new_page()

            for cat_path, cat_name in categories:
                print(f"\n{'='*60}")
                print(f"CATEGORY: {cat_name}")
                print(f"{'='*60}")

                products = self.crawl_category(pg, cat_path, cat_name, num_pages=pages_per_category)

                # Clean names
                for prod in products:
                    cleaned = self.clean_product_name(prod["product_name_raw"])
                    prod["product_name"] = cleaned["product_name"]
                    prod["brand"] = cleaned["brand"]

                self.products.extend(products)

            browser.close()

    def save(self, output_path="hasaki_products_full.json"):
        # Dedup by product_id
        seen = set()
        unique = []
        for p in self.products:
            if p["product_id"] not in seen:
                seen.add(p["product_id"])
                unique.append(p)
        self.products = unique

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.products, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(self.products)} unique products -> {output_path}")

        # CSV
        csv_path = output_path.replace(".json", ".csv")
        keys = ["product_id", "product_name", "brand", "category_name", "product_url", "image_url", "crawled_at"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.products)
        print(f"Saved {len(self.products)} products -> {csv_path}")

    def print_summary(self):
        from collections import Counter
        cats = Counter(p["category_name"] for p in self.products)
        brands = Counter(p["brand"] for p in self.products if p["brand"])

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(self.products)} unique products")
        print(f"\nBy category:")
        for c, n in cats.most_common():
            print(f"  {c}: {n}")
        print(f"\nTop 20 brands:")
        for b, n in brands.most_common(20):
            print(f"  {b}: {n}")
        print(f"  (no brand detected): {sum(1 for p in self.products if not p['brand'])}")


if __name__ == "__main__":
    print("HASAKI FULL CRAWLER")
    print("10 pages per category\n")

    crawler = HasakiFullCrawler()
    crawler.run(pages_per_category=10)
    crawler.save("hasaki_products_full.json")
    crawler.print_summary()
    print("\nDone!")
