from io import BytesIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


MAX_PDF_MB = 20
MAX_PDF_PAGES = 200
MAX_PDF_CHARACTERS = 300_000


class PdfAttachmentError(ValueError):
    """A PDF could not be used as a chat attachment."""


def extract_pdf_chunks(data: bytes) -> tuple[list[Document], int]:
    """Extract text by page and preserve one-based PDF page numbers in each chunk."""
    if len(data) > MAX_PDF_MB * 1024 * 1024:
        raise PdfAttachmentError(f"Choose a PDF up to {MAX_PDF_MB} MB.")

    documents = []
    total_characters = 0
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise PdfAttachmentError("Unlock the password-protected PDF before uploading it.")

        page_count = len(reader.pages)
        if page_count > MAX_PDF_PAGES:
            raise PdfAttachmentError(f"Choose a PDF with at most {MAX_PDF_PAGES} pages.")

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            total_characters += len(text)
            if total_characters > MAX_PDF_CHARACTERS:
                raise PdfAttachmentError("This PDF contains too much text. Upload a shorter section.")
            documents.append(Document(page_content=text, metadata={"page": page_number}))
    except PdfReadError as error:
        raise PdfAttachmentError("This file could not be read as a PDF. Choose a valid PDF.") from error

    if not documents:
        raise PdfAttachmentError("This PDF has no extractable text. Upload a PDF with selectable text.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    return splitter.split_documents(documents), page_count
