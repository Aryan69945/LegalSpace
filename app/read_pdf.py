import fitz

pdf_path = "data/constitution_of_india.pdf"

doc = fitz.open(pdf_path)

print(f"Total pages: {len(doc)}")

for page_number, page in enumerate(doc):
    text = page.get_text("text")

    print(f"\n===== PAGE {page_number + 1} =====")
    print(text[:2000])
