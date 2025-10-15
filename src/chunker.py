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
                
                if not lines:
                    # No line information available
                    for idx, p in enumerate(pieces):
                        p = (p or "").strip()
                        if not p:
                            continue
                        
                        chunk = {
                            "id": f"{filename}_p{page['page_number']}_c{idx}",
                            "source": filename,
                            "page_number": page["page_number"],
                            "start_line_number": None,
                            "end_line_number": None,
                            "text": p,
                            "figures": page.get("figures", [])
                        }
                        chunks.append(chunk)
                    continue

                # Build a mapping of character positions to line numbers
                char_to_line = self._build_char_to_line_mapping(lines)
                
                # Track position in the original text
                current_pos = 0
                
                for idx, chunk_text in enumerate(pieces):
                    chunk_text = (chunk_text or "").strip()
                    if not chunk_text:
                        continue
                    
                    # Find where this chunk appears in the original text
                    # Normalize whitespace for matching
                    normalized_text = re.sub(r'\s+', ' ', text.lower())
                    normalized_chunk = re.sub(r'\s+', ' ', chunk_text.lower())
                    
                    # Find the chunk in the text starting from current position
                    chunk_start = normalized_text.find(normalized_chunk, current_pos)
                    
                    if chunk_start == -1:
                        # Fallback: try finding from beginning
                        chunk_start = normalized_text.find(normalized_chunk)
                    
                    if chunk_start == -1:
                        # If still not found, use estimation
                        start_line, end_line = self._estimate_lines(idx, len(pieces), len(lines))
                    else:
                        chunk_end = chunk_start + len(normalized_chunk)
                        
                        # Get line numbers for this character range
                        start_line = char_to_line.get(chunk_start)
                        end_line = char_to_line.get(chunk_end - 1)
                        
                        # If we couldn't map to lines, find closest
                        if start_line is None:
                            start_line = self._find_closest_line(chunk_start, char_to_line)
                        if end_line is None:
                            end_line = self._find_closest_line(chunk_end - 1, char_to_line)
                        
                        # Update position for next chunk
                        # Account for overlap by going back a bit
                        current_pos = chunk_start + len(normalized_chunk) - (self.splitter._chunk_overlap // 2)
                        current_pos = max(0, current_pos)
                    
                    # Safety check
                    if start_line is not None and end_line is not None:
                        if end_line < start_line:
                            end_line = start_line
                        # Ensure line numbers are within bounds
                        start_line = max(1, min(start_line, len(lines)))
                        end_line = max(start_line, min(end_line, len(lines)))
                    
                    chunk = {
                        "id": f"{filename}_p{page['page_number']}_c{idx}",
                        "source": filename,
                        "page_number": page["page_number"],
                        "start_line_number": start_line,
                        "end_line_number": end_line,
                        "text": chunk_text,
                        "figures": page.get("figures", [])
                    }
                    chunks.append(chunk)
        
        return chunks

    def _build_char_to_line_mapping(self, lines: List[str]) -> Dict[int, int]:
        """Build a mapping from character position to line number."""
        char_to_line = {}
        current_pos = 0
        
        for line_idx, line in enumerate(lines, start=1):
            line_len = len(line)
            for i in range(line_len):
                char_to_line[current_pos + i] = line_idx
            # Account for newline character
            current_pos += line_len + 1
        
        return char_to_line

    def _find_closest_line(self, pos: int, char_to_line: Dict[int, int]) -> int:
        """Find the closest line number for a given character position."""
        if not char_to_line:
            return None
        
        # Try to find exact match first
        if pos in char_to_line:
            return char_to_line[pos]
        
        # Find closest position
        positions = sorted(char_to_line.keys())
        
        # Binary search for closest position
        left, right = 0, len(positions) - 1
        closest_pos = positions[0]
        min_diff = abs(pos - positions[0])
        
        while left <= right:
            mid = (left + right) // 2
            diff = abs(pos - positions[mid])
            
            if diff < min_diff:
                min_diff = diff
                closest_pos = positions[mid]
            
            if positions[mid] < pos:
                left = mid + 1
            elif positions[mid] > pos:
                right = mid - 1
            else:
                return char_to_line[positions[mid]]
        
        return char_to_line.get(closest_pos, 1)

    def _estimate_lines(self, chunk_idx: int, total_chunks: int, total_lines: int) -> tuple:
        """Estimate line numbers when exact matching fails."""
        if total_lines == 0 or total_chunks == 0:
            return None, None
        
        # Calculate lines per chunk
        lines_per_chunk = total_lines / total_chunks
        
        # Estimate start and end
        start_line = int(chunk_idx * lines_per_chunk) + 1
        end_line = int((chunk_idx + 1) * lines_per_chunk)
        
        # Ensure within bounds
        start_line = max(1, min(start_line, total_lines))
        end_line = max(start_line, min(end_line, total_lines))
        
        return start_line, end_line

    def save_chunks(self, chunks: List[Dict], output_file: str):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)