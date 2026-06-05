import re

class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Executes intensive local text normalization. Removes layout noise,
        standardizes spacing, and strips non-printable control characters.
        """
        if not text:
            return ""
        
        # Strip ASCII control characters/null bytes
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Standardize structural bullet points and weird dashes
        text = re.sub(r'[\u2014\u2015]', '—', text)
        text = re.sub(r'[\u2018\u2019]', "'", text)
        text = re.sub(r'[\u201c\u201d]', '"', text)
        
        # Force uniform spacing without removing necessary line breaks
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        return text.strip()

    @staticmethod
    def normalize_identifiers(text: str) -> str:
        """Normalizes tokens specifically for localized keyword/BM25 extraction."""
        text = text.lower()
        # Strip punctuation for strict keyword normalization
        text = re.sub(r'[^\w\s\-]', '', text)
        return text