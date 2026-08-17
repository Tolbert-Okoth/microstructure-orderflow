"""
Deep extraction and mathematical analysis of the 3 new research papers:
1. Kyle (1985) - Continuous Auctions and Insider Trading
2. Bouchaud, Farmer, Lillo (2008) - How Markets Slowly Digest Changes in Supply and Demand
3. arXiv:2307.00413 - Deep Limit Order Book / Market Making / Execution
"""
import pypdf
from pathlib import Path
import json

DATA_DIR = Path("data")
pdf_files = list(DATA_DIR.glob("*.pdf"))

results = {}

for pdf_path in sorted(pdf_files):
    print(f"Extracting {pdf_path.name}...")
    reader = pypdf.PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    full_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        full_text.append(f"--- PAGE {i+1} ---\n" + text)
    
    combined = "\n".join(full_text)
    results[pdf_path.name] = {
        "num_pages": num_pages,
        "char_count": len(combined),
        "first_page": reader.pages[0].extract_text()[:1500] if num_pages > 0 else "",
        "full_text": combined
    }
    print(f"  Extracted {num_pages} pages ({len(combined):,} chars)")

# Save full extracted text to json
with open(DATA_DIR / "extracted_microstructure_papers.json", "w", encoding="utf-8") as f:
    json.dump({k: {"num_pages": v["num_pages"], "char_count": v["char_count"], "first_page": v["first_page"]} for k, v in results.items()}, f, indent=2)

# Save text files for each paper
for fname, data in results.items():
    txt_path = DATA_DIR / (fname.replace(".pdf", ".txt"))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(data["full_text"])
    print(f"  Wrote text to {txt_path.name}")
