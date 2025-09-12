# src/chunker.py
import json
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        # Use LangChain splitter for sentence-aware splits
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def create_chunks_from_docs(self, docs: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Returns list of chunk dicts:
        { "id","source","page_number","text","figures" }
        """
        chunks = []
        for filename, pages in docs.items():
            for page in pages:
                text = page.get("text", "").strip()
                if not text:
                    continue
                pieces = self.splitter.split_text(text)
                for idx, p in enumerate(pieces):
                    chunk = {
                        "id": f"{filename}_p{page['page_number']}_c{idx}",
                        "source": filename,
                        "page_number": page["page_number"],
                        "text": p,
                        "figures": page.get("figures", [])
                    }
                    chunks.append(chunk)
        return chunks

    def save_chunks(self, chunks: List[Dict], output_file: str):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
