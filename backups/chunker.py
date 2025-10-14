# # src/chunker.py
# import json
# from typing import Dict, List
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# class TextChunker:
#     def __init__(self, chunk_size: int = 800, overlap: int = 200):
#         # Use LangChain splitter for sentence-aware splits
#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=overlap
#         )

#     def create_chunks_from_docs(self, docs: Dict[str, List[Dict]]) -> List[Dict]:
#         """
#         Returns list of chunk dicts:
#         { "id","source","page_number","text","figures" }
#         """
#         chunks = []
#         for filename, pages in docs.items():
#             for page in pages:
#                 text = page.get("text", "").strip()
#                 if not text:
#                     continue
#                 pieces = self.splitter.split_text(text)
#                 for idx, p in enumerate(pieces):
#                     chunk = {
#                         "id": f"{filename}_p{page['page_number']}_c{idx}",
#                         "source": filename,
#                         "page_number": page["page_number"],
#                         "text": p,
#                         "figures": page.get("figures", [])
#                     }
#                     chunks.append(chunk)
#         return chunks

#     def save_chunks(self, chunks: List[Dict], output_file: str):
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(chunks, f, indent=2, ensure_ascii=False)

# src/chunker.py (replace your create_chunks_from_docs with this)
import json
import re
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def create_chunks_from_docs(self, docs: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Returns list of chunk dicts:
        { "id","source","page_number","text","figures","line_number" }
        """
        chunks = []
        for filename, pages in docs.items():
            for page in pages:
                text = page.get("text", "") or ""
                text = text.strip()
                if not text:
                    continue

                pieces = self.splitter.split_text(text)
                lines = page.get("lines", []) or []

                for idx, p in enumerate(pieces):
                    # ensure p is a string and normalized
                    p = (p or "").strip()

                    # 1) Try a soft substring match between page lines and chunk text
                    line_number = None
                    normalized_chunk = re.sub(r"\s+", " ", p.lower())

                    for line_idx, line in enumerate(lines, start=1):
                        if not line or not line.strip():
                            continue
                        normalized_line = re.sub(r"\s+", " ", line.strip().lower())
                        # check first N chars of the line are in the first M chars of the chunk
                        if normalized_line[:25] and normalized_line[:25] in normalized_chunk[:300]:
                            line_number = line_idx
                            break

                    # 2) Fallback: estimate line number if no match found
                    if line_number is None and lines:
                        # approximate how many lines per chunk on this page
                        approx_lines_per_chunk = max(1, len(lines) // max(1, len(pieces)))
                        line_number = (idx * approx_lines_per_chunk) + 1

                    # Build chunk dict
                    chunk = {
                        "id": f"{filename}_p{page['page_number']}_c{idx}",
                        "source": filename,
                        "page_number": page["page_number"],
                        "line_number": line_number,
                        "text": p,
                        "figures": page.get("figures", [])
                    }
                    chunks.append(chunk)
        return chunks

    def save_chunks(self, chunks: List[Dict], output_file: str):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

