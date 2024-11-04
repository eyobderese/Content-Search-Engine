import fitz  # PyMuPDF
import re


def extract_text_chunks_by_font_size(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []

    current_chunk = ["", ""]
    current_font_size = 11.0

    # Iterate through each page in the document
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]

                        # Check if this is the start of a new chunk
                        if font_size > current_font_size:
                            chunks.append(current_chunk)
                            current_chunk = [text, ""]
                        else:
                            print(current_chunk)
                            current_chunk[1] += " " + text

                        # Update current font size
                        current_font_size = font_size

    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(current_chunk[1].strip())

    return chunks


# Example usage
pdf_path = "./MindPlex.pdf"
chunks = extract_text_chunks_by_font_size(pdf_path)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")
