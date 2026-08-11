import re
import json
import pymupdf

PDF_PATH = "data/constitution_of_india.pdf"


def extract_text(pdf_path):
    doc = pymupdf.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    doc.close()

    return text


text = extract_text(PDF_PATH)

# Detect Article numbers such as:
# 1.  2.  14.  21.  300A.
article_pattern = r"(?m)^\s*(\d+[A-Z]?)\.\s+"

matches = list(re.finditer(article_pattern, text))

chunks = []

for i, match in enumerate(matches):

    article_number = match.group(1)

    start = match.start()

    if i + 1 < len(matches):
        end = matches[i + 1].start()
    else:
        end = len(text)

    article_text = text[start:end].strip()

    if len(article_text) < 30:
        continue

    chunks.append({
        "source": "constitution_of_india.pdf",
        "article": f"Article {article_number}",
        "text": article_text
    })


print(f"Found {len(chunks)} articles")

for chunk in chunks[:10]:
    print("\n--------------------")
    print(chunk["article"])
    print(chunk["text"][:500])


with open(
    "processed/constitution_articles.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nSaved to processed/constitution_articles.json")