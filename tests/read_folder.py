import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import re

@dataclass
class TextBlock:
    """Represents a text block with position"""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def center_x(self):
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self):
        return (self.y0 + self.y1) / 2

class PDFVisualOrderExtractor:
    def __init__(self, column_threshold: float = 50):
        """
        Initialize the PDF extractor
        
        Args:
            column_threshold: Horizontal distance to consider as new column (in points)
        """
        self.column_threshold = column_threshold
    
    def detect_columns(self, blocks: List[TextBlock], page_width: float) -> List[List[TextBlock]]:
        """
        Detect columns and group blocks accordingly
        """
        if not blocks:
            return []
        
        # Sort blocks by x-coordinate to find column boundaries
        sorted_by_x = sorted(blocks, key=lambda b: b.x0)
        
        columns = []
        current_column = [sorted_by_x[0]]
        current_max_x = sorted_by_x[0].x1
        
        for block in sorted_by_x[1:]:
            # Check if this block starts a new column
            if block.x0 > current_max_x + self.column_threshold:
                columns.append(current_column)
                current_column = [block]
                current_max_x = block.x1
            else:
                current_column.append(block)
                current_max_x = max(current_max_x, block.x1)
        
        if current_column:
            columns.append(current_column)
        
        return columns
    
    def sort_blocks_in_reading_order(self, blocks: List[TextBlock], page_width: float) -> List[TextBlock]:
        """
        Sort text blocks in natural reading order: left-to-right, top-to-bottom
        """
        if not blocks:
            return []
        
        # Detect columns
        columns = self.detect_columns(blocks, page_width)
        
        # Sort each column by y-coordinate (top to bottom)
        sorted_blocks = []
        for column in columns:
            column_sorted = sorted(column, key=lambda b: b.y0)
            sorted_blocks.extend(column_sorted)
        
        return sorted_blocks
    
    def extract_text_blocks(self, page) -> List[TextBlock]:
        """
        Extract text blocks from a page with position information
        """
        blocks = []
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    
                    if line_text.strip():
                        bbox = line.get("bbox", block.get("bbox"))
                        blocks.append(TextBlock(
                            text=line_text.strip(),
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3]
                        ))
        
        return blocks
    
    def extract_figures(self, page) -> List[Dict[str, Any]]:
        """
        Extract figure/image information from a page
        """
        figures = []
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 1:  # Image block
                bbox = block.get("bbox")
                figures.append({
                    "type": "image",
                    "bbox": {
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3]
                    },
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1]
                })
        
        return figures
    
    def process_pdf(self, pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process entire PDF and extract content in visual reading order
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        
        result = {pdf_path.name: []}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = page.rect.width
            
            # Extract text blocks
            text_blocks = self.extract_text_blocks(page)
            
            # Sort blocks in reading order
            sorted_blocks = self.sort_blocks_in_reading_order(text_blocks, page_width)
            
            # Extract figures
            figures = self.extract_figures(page)
            
            # Combine text
            full_text = "\n".join(block.text for block in sorted_blocks)
            lines = [block.text for block in sorted_blocks]
            
            page_data = {
                "page_number": page_num + 1,
                "text": full_text,
                "lines": lines,
                "figures": figures
            }
            
            result[pdf_path.name].append(page_data)
        
        doc.close()
        return result
    
    def save_to_json(self, data: Dict, output_path: str):
        """
        Save extracted data to JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def process_folder(folder_path: str, output_path: str, column_threshold: float = 50):
    """
    Process all PDF files in a folder
    
    Args:
        folder_path: Path to folder containing PDF files
        output_path: Path to output JSON file
        column_threshold: Horizontal distance to consider as new column
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    # Find all PDF files
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'")
        return
    
    print(f"Found {len(pdf_files)} PDF files in '{folder_path}'")
    print("-" * 60)
    
    # Create extractor
    extractor = PDFVisualOrderExtractor(column_threshold=column_threshold)
    
    # Process all PDFs and combine results
    combined_result = {}
    
    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing: {pdf_file.name}")
            result = extractor.process_pdf(str(pdf_file))
            combined_result.update(result)
            
            # Print summary for this PDF
            pdf_name = pdf_file.name
            print(f"  ✓ Extracted {len(result[pdf_name])} pages")
            for page_data in result[pdf_name]:
                print(f"    Page {page_data['page_number']}: {len(page_data['lines'])} lines, {len(page_data['figures'])} figures")
        
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {str(e)}")
            continue
    
    # Save combined results
    if combined_result:
        extractor.save_to_json(combined_result, output_path)
        print(f"\n{'=' * 60}")
        print(f"All results saved to: {output_path}")
        print(f"Total PDFs processed: {len(combined_result)}")
    else:
        print("\nNo PDFs were successfully processed")


def main():
    """
    Main function to demonstrate usage
    """
    # Configuration
    FOLDER_PATH = "../data"  # Folder containing PDF files
    OUTPUT_PATH = "output.json"
    COLUMN_THRESHOLD = 50  # Adjust based on your PDF layout
    
    # Process all PDFs in folder
    process_folder(FOLDER_PATH, OUTPUT_PATH, COLUMN_THRESHOLD)


if __name__ == "__main__":
    main()