from pathlib import Path
from pydantic import BaseModel


class ParsedPage(BaseModel):
    page_number: int
    text: str
    title: str = ""


class ParsedDocument(BaseModel):
    name: str
    pages: list[ParsedPage]


def parse_pdf(path: Path) -> ParsedDocument:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to parse PDFs") from exc
    try:
        pdf = fitz.open(path)
        pages = []
        for number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            title = ""
            for line in text.splitlines()[:8]:
                if 3 < len(line.strip()) < 140 and not line.strip().endswith((".", ":")):
                    title = line.strip()
                    break
            pages.append(ParsedPage(page_number=number, text=text, title=title))
        if not any(page.text for page in pages):
            raise ValueError("The PDF contains no extractable text; scanned PDFs need OCR before indexing.")
        return ParsedDocument(name=path.name, pages=pages)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted PDF: {exc}") from exc
