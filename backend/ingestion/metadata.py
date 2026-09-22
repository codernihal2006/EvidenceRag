from pathlib import Path
from .pdf_parser import ParsedDocument


def document_metadata(path: Path, parsed: ParsedDocument) -> dict:
    return {"document_name": path.name, "title": parsed.name, "pages": len(parsed.pages)}
