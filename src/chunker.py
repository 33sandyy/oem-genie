import json
from typing import Dict, List

class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 200, deduplicate: bool = True):
        """
        :param chunk_size: Number of lines per chunk
        :param overlap: Number of overlapping lines between consecutive chunks
        :param deduplicate: Whether to remove identical chunks after splitting
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.deduplicate = deduplicate

    def create_chunks_from_docs(self, docs: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Returns a list of chunk dictionaries:
        { "id", "source", "page_number", "text", "figures", "line_number", "end_line_number" }
        """
        chunks = []

        for filename, pages in docs.items():
            for page in pages:
                lines = page.get("lines", []) or []
                if not lines:
                    continue

                # Preprocess lines: remove empty lines
                lines = [line.strip() for line in lines if line.strip()]
                num_lines = len(lines)
                start_idx = 0

                while start_idx < num_lines:
                    end_idx = min(start_idx + self.chunk_size, num_lines)
                    chunk_lines = lines[start_idx:end_idx]
                    chunk_text = " ".join(chunk_lines)
                    chunk_line_number = start_idx + 1  # first line of chunk
                    end_line_number = chunk_line_number + len(chunk_lines) - 1

                    chunk = {
                        "id": f"{filename}_p{page['page_number']}_c{start_idx}",
                        "source": filename,
                        "page_number": page["page_number"],
                        "line_number": chunk_line_number,
                        "end_line_number": end_line_number,
                        "text": chunk_text,
                        "figures": page.get("figures", [])
                    }
                    chunks.append(chunk)

                    # Move to next chunk with overlap
                    start_idx += max(1, self.chunk_size - self.overlap)

        # Deduplicate chunks if requested
        if self.deduplicate:
            seen_texts = set()
            unique_chunks = []
            for c in chunks:
                if c["text"] not in seen_texts:
                    unique_chunks.append(c)
                    seen_texts.add(c["text"])
            chunks = unique_chunks

        return chunks

    def save_chunks(self, chunks: List[Dict], output_file: str):
        """
        Save chunks as a JSON file
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

