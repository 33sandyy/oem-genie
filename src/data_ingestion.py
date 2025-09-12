# src/data_ingestion.py
import os
import json
from typing import Dict, List
from pypdf import PdfReader
import pdfplumber

class PDFIngestor:
    def __init__(self, pdf_folder: str, max_pages: int = 999):
        self.pdf_folder = pdf_folder
        self.max_pages = max_pages

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        data = []
        reader = PdfReader(pdf_path)
        total_pages = min(len(reader.pages), self.max_pages)
        for i in range(total_pages):
            page = reader.pages[i]
            page_text = page.extract_text() or ""
            data.append({
                "page_number": i + 1,
                "text": page_text.strip(),
                "figures": []
            })

        # extract figure-like words with pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = min(len(pdf.pages), self.max_pages)
                for i in range(total_pages):
                    figures = []
                    try:
                        for obj in pdf.pages[i].extract_words():
                            txt = obj.get("text", "")
                            if "fig." in txt.lower() or "diagram" in txt.lower():
                                figures.append(txt)
                    except Exception:
                        pass
                    if i < len(data):
                        data[i]["figures"] = figures
        except Exception:
            pass

        return data

    def ingest_folder(self) -> Dict[str, List[Dict]]:
        all_docs = {}
        for filename in sorted(os.listdir(self.pdf_folder)):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_folder, filename)
                print(f"Ingesting {pdf_path} ...")
                all_docs[filename] = self.extract_text_from_pdf(pdf_path)
        return all_docs

    def save_to_json(self, docs: Dict, output_file: str):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
