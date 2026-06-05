import os
import aiofiles
from pypdf import PdfReader
from docx import Document
from app.utils.text_preprocessor import TextPreprocessor
from app.exceptions import UnreadableDocumentError # Custom app error boundaries

class DocumentProcessorService:
    """
    Decoupled multi-format parsing engine running entirely air-gapped.
    Converts binary document streams securely into clean structural layout strings.
    """
    def __init__(self):
        self.preprocessor = TextPreprocessor()

    async def extract_text(self, file_path: str) -> str:
        """Master route handler evaluating file signatures securely."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document path missing: {file_path}")

        _, ext = os.path.splitext(file_path.lower())
        
        try:
            if ext == '.pdf':
                raw_text = await self._parse_pdf(file_path)
            elif ext == '.docx':
                raw_text = await self._parse_docx(file_path)
            elif ext in ['.md', '.txt']:
                raw_text = await self._parse_markdown(file_path)
            else:
                raise UnreadableDocumentError(f"Unsupported extension signature: {ext}")
            
            return self.preprocessor.clean_text(raw_text)
            
        except Exception as e:
            raise UnreadableDocumentError(f"Engine failed extraction boundary for {ext}: {str(e)}")

    async def _parse_pdf(self, file_path: str) -> str:
        """Extracts text sections locally via high-performance pypdf streaming."""
        text_content = []
        # Run inside standard thread pool context via aiofiles if doing heavy I/O
        reader = PdfReader(file_path)
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
        return "\n".join(text_content)

    async def _parse_docx(self, file_path: str) -> str:
        """Parses Microsoft OpenXML structures securely."""
        doc = Document(file_path)
        text_content = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(text_content)

    async def _parse_markdown(self, file_path: str) -> str:
        """Reads flat markdown text asynchronously."""
        async with aiofiles.open(file_path, mode='r', encoding='utf-8', errors='ignore') as f:
            return await f.read()