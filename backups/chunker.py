import json
import re
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.data_ingestion import PDFIngestor

class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def create_chunks_from_docs(self, docs: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Returns list of chunk dicts:
        { "id","source","page_number","text","figures","start_line_number","end_line_number" }
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
                total_lines = len(lines)

                for idx, p in enumerate(pieces):
                    p = (p or "").strip()
                    if not p:
                        continue

                    # normalize chunk text once
                    normalized_chunk = re.sub(r"\s+", " ", p.lower())

                    matched_indices = []

                    # Search all page lines to find any line that appears in this chunk.
                    # We use full normalized_line if possible, otherwise a 25-char prefix as a fallback.
                    for line_idx, line in enumerate(lines, start=1):
                        if not line or not line.strip():
                            continue
                        normalized_line = re.sub(r"\s+", " ", line.strip().lower())
                        if not normalized_line:
                            continue

                        # Strong match: entire normalized line is in chunk
                        if normalized_line in normalized_chunk:
                            matched_indices.append(line_idx)
                            continue

                        # Fallback weaker match: first 25 chars of the line appear in the first 300 chars of chunk
                        # (addresses situations with truncated/overlap text)
                        prefix = normalized_line[:25]
                        if prefix and prefix in normalized_chunk[:300]:
                            matched_indices.append(line_idx)

                    if matched_indices:
                        start_line = min(matched_indices)
                        end_line = max(matched_indices)
                    else:
                        # Fallback: proportional estimate when no actual matches found
                        if total_lines > 0:
                            approx_lines_per_chunk = max(1, total_lines // max(1, len(pieces)))
                            start_line = (idx * approx_lines_per_chunk) + 1
                            end_line = min(total_lines, (idx + 1) * approx_lines_per_chunk)
                        else:
                            start_line = None
                            end_line = None

                    # Safety: ensure end_line is not less than start_line (if both exist)
                    if start_line is not None and end_line is not None and end_line < start_line:
                        end_line = start_line

                    chunk = {
                        "id": f"{filename}_p{page['page_number']}_c{idx}",
                        "source": filename,
                        "page_number": page["page_number"],
                        "start_line_number": start_line,
                        "end_line_number": end_line,
                        "text": p,
                        "figures": page.get("figures", [])
                    }
                    chunks.append(chunk)
        return chunks




    def save_chunks(self, chunks: List[Dict], output_file: str):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

