###################################################################################################################################################################
"""
DOCUMENTATION & HELPER LOGIC
----------------------------

1. is_noise(line)
   - Inp: Raw text line.
   - Out: Boolean.
   - Desc: Detects non-content lines like page footers, figure captions, or running headers to prevent them from polluting the text.

2. is_section_heading(line)
   - Inp: Raw text line.
   - Out: Tuple (section_number, title) or None.
   - Desc: Regex-based detector that identifies hierarchical headers (e.g., "**1.1.3** Technical Data") to signal the start of a new semantic block.

3. clean_text(text)
   - Inp: Raw text string.
   - Out: Cleaned string.
   - Desc: Removes trailing whitespace and collapses empty lines to ensure compact, readable text for the LLM.

4. extract_sections_from_pages(pages)
   - Inp: List of page objects (from pymupdf4llm).
   - Out: List of structured section dictionaries.
   - Desc: The core logic that iterates through every line of the PDF, grouping content under the most recent active header (stateful parsing).

5. create_section_object(section_info, content_lines, pages)
   - Inp: Header metadata, list of text lines, and page numbers.
   - Out: Final dictionary object for the section.
   - Desc: Assembles the raw data into a structured object, calculates metadata (like 'has_table'), and formats the section numbers.

6. process_pdf(pdf_path)
   - Inp: Path to the input PDF file.
   - Out: None (Writes output to disk).
   - Desc: Orchestrator function that converts PDF to Markdown, runs the section extractor, and saves the result.

7. save_sections(sections, out_file, doc_id)
   - Inp: List of section objects, output path, and document ID.
   - Out: None (Writes file).
   - Desc: Helper to serialize the extracted data into line-delimited JSON (JSONL) format.
"""
###################################################################################################################################################################

import json
import re
from pathlib import Path
from typing import List, Dict, Optional

import pymupdf4llm

from src.rag.utils.config_loader import load_config
from src.rag.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

RAW_DIR = Path(config["raw_data_path"])
OUT_DIR = Path(config["extracted_data_path"])
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Regex patterns
# -------------------------

# Match section headings like "1.1.3" or "**1.1.3** Technical data"
# Must have at least one dot to be a subsection
# SECTION_HEADING_RE = re.compile(r"^\*{0,2}(\d+\.\d+(?:\.\d+)*)\*{0,2}\s+(.+)$")
SECTION_HEADING_RE = re.compile(r"^\*{0,2}(\d+(?:\.\d+)+)\*{0,2}\s+\*\*(.+?)\*\*$")
PAGE_FOOTER_RE = re.compile(r"^(?:\d+\s*/\s*\d+|.*Omega\s+\d+/\d+.*?EN)\s*$")
FIGURE_RE = re.compile(r"^Fig\.\s+\d+$")
RUNNING_HEADER_RE = re.compile(r"^\*{2}[A-Z\s]+\*{2}$")


# -------------------------
# Helpers
# -------------------------

def is_noise(line: str) -> bool:
    """Check if line is noise."""
    line = line.strip()
    return bool(
        PAGE_FOOTER_RE.match(line) or 
        FIGURE_RE.match(line) or 
        RUNNING_HEADER_RE.match(line) or
        not line
    )


def is_section_heading(line: str) -> Optional[tuple]:
    """Check if line is section heading. Returns (section_num, title) or None."""
    match = SECTION_HEADING_RE.match(line.strip())
    if match:
        section_num = match.group(1)
        title = match.group(2).strip()
        # Only accept if it has proper section numbering (at least one dot)
        if '.' in section_num:
            return section_num, title
    return None


def clean_text(text: str) -> str:
    """Clean text."""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join([l for l in lines if l.strip()])


# -------------------------
# Section extraction
# -------------------------

def extract_sections_from_pages(pages: List[Dict]) -> List[Dict]:
    """Extract sections from all pages."""
    sections = []
    current_section = None
    content_buffer = []
    current_pages = set()
    
    for page_idx, page in enumerate(pages, start=1):
        raw_text = page.get("text", "")
        if not raw_text.strip():
            continue
        
        for line in raw_text.splitlines():
            if is_noise(line):
                continue
            
            heading_info = is_section_heading(line)
            
            # New section heading found
            if heading_info:
                # Save previous section
                if current_section:
                    sections.append(create_section_object(
                        current_section,
                        content_buffer,
                        current_pages
                    ))
                
                # Start new section
                section_num, title = heading_info
                current_section = {
                    "section_num": section_num,
                    "title": title,
                    "heading_line": line.strip()
                }
                content_buffer = []
                current_pages = {page_idx}
            
            # Content line
            elif current_section:
                content_buffer.append(line.strip())
                current_pages.add(page_idx)
    
    # Save last section
    if current_section:
        sections.append(create_section_object(
            current_section,
            content_buffer,
            current_pages
        ))
    
    return sections


def create_section_object(section_info: Dict, content_lines: List[str], pages: set) -> Dict:
    """Create section object with metadata."""
    # Combine heading and content
    full_text = section_info["heading_line"]
    if content_lines:
        full_text += "\n\n" + "\n".join(content_lines)
    
    full_text = clean_text(full_text)
    
    # Parse section number to get main section and subsection
    section_num = section_info["section_num"]
    parts = section_num.split(".")
    
    main_section = parts[0] if len(parts) >= 1 else ""
    subsection = ".".join(parts[1:]) if len(parts) > 1 else ""
    
    # Check if has table
    # has_table = any("|" in line for line in content_lines)
    
    has_table = any(
        line.count("|") >= 2 and "---" in next_line
        for line, next_line in zip(content_lines, content_lines[1:])
    )

    return {
        "text": full_text,
        "metadata": {
            "doc_id": "",  # Will be set later
            "section": main_section,
            "section_title": section_info["title"],
            "subsection": subsection,
            "section_full": section_num,
            "has_table": has_table,
            "pages": sorted(list(pages))
        }
    }


# -------------------------
# Main processing
# -------------------------

def process_pdf(pdf_path: Path):
    """Process PDF file."""
    doc_id = pdf_path.stem
    logger.info(f"Processing {doc_id}")
    
    # Extract markdown from PDF
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    
    # Extract sections
    sections = extract_sections_from_pages(pages)
    
    logger.info(f"Extracted {len(sections)} sections from {doc_id}")
    
    # Save to JSONL
    out_file = OUT_DIR / f"{doc_id}.jsonl"
    save_sections(sections, out_file, doc_id)
    
    logger.info(f"Saved to {out_file}")


def save_sections(sections: List[Dict], out_file: Path, doc_id: str):
    """Save sections to JSONL."""
    with out_file.open("w", encoding="utf-8") as f:
        for section in sections:
            # Add doc_id to metadata
            section["metadata"]["doc_id"] = doc_id
            
            record = {
                "text": section["text"],
                "metadata": section["metadata"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    """Main entry point."""
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDFs found in {RAW_DIR}")
        return
    
    for pdf in pdfs:
        try:
            process_pdf(pdf)
        except Exception as e:
            logger.error(f"Failed to process {pdf}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
