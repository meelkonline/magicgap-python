import json
import openai
import concurrent.futures
import os
import re
import fitz  # PyMuPDF
import spacy
import tiktoken
from dotenv import load_dotenv

# Load .env variables
load_dotenv()


class SimplePdfChunker:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.nlp = spacy.load("en_core_web_sm")  # NLP model for sentence splitting & entity extraction
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT tokenizer

    def extract_text_by_paragraphs(self) -> list:
        """
        Extracts text from a PDF and splits it by paragraphs (based on blank lines).
        """
        doc = fitz.open(self.pdf_path)
        full_text = []

        for page in doc:
            page_text = page.get_text("text")

            # Clean extracted text
            page_text = self.clean_text(page_text)

            # Split into paragraphs (double newlines indicate section breaks)
            paragraphs = page_text.split("\n\n")

            for para in paragraphs:
                para = para.strip()

                # Check if this is a **title** (short sentence, no punctuation like "." or "!?")
                is_title = len(para) < 60 and not any(p in para for p in ".!?")

                if para:
                    # If it's a title, add an extra newline for clarity
                    full_text.append(para + ("\n" if is_title else ""))

        doc.close()
        return full_text

    def clean_text(self, text: str) -> str:
        """
        Cleans extracted text by:
        - Removing unwanted **single newlines inside paragraphs**.
        - Keeping **double newlines** for paragraph separation.
        - Fixing **hyphenated words split across lines**.
        """

        # 1️⃣ Remove hyphenated word breaks (common in justified PDF text)
        text = re.sub(r"-\n", "", text)

        # 2️⃣ Remove unwanted single newlines inside paragraphs
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # 3️⃣ Normalize spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def split_sentences(self, text):
        """Splits text into sentences ensuring no isolated titles."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def extract_entities(self, text):
        """Extracts main entities (persons, organizations, places, etc.) from a given text."""
        doc = self.nlp(text)
        entities = {ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "EVENT", "PRODUCT"}}
        return list(entities)

    def tokenize(self, text):
        """Returns the number of tokens in the text."""
        return len(self.tokenizer.encode(text))

    def chunk_document(self, max_tokens=256):
        """
        Chunks the document while ensuring:
        - No isolated titles (titles are kept within sections)
        - Chunks stop at sentence boundaries and do not exceed max_tokens
        - Entities are extracted for better retrieval
        """
        paragraphs = self.extract_text_by_paragraphs()
        chunks = []
        current_chunk = []
        current_token_count = 0
        section_title = ""

        for paragraph in paragraphs:
            is_title = len(paragraph) < 60 and not any(p in paragraph for p in ".!?")
            if is_title:
                section_title = paragraph  # Save the section title
                continue  # Skip isolated titles (they'll be added to chunks)

            sentences = self.split_sentences(paragraph)

            for sentence in sentences:
                sentence_tokens = self.tokenize(sentence)

                # If adding this sentence exceeds token limit, finalize the current chunk
                if current_token_count + sentence_tokens > max_tokens:
                    if current_chunk:
                        chunk_text = "\n".join(current_chunk)
                        chunks.append(f"{section_title}\n\n{chunk_text}" if section_title else chunk_text)

                    current_chunk = []
                    current_token_count = 0

                # Add sentence to the chunk
                current_chunk.append(sentence)
                current_token_count += sentence_tokens

        # Add last chunk if it exists
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append(f"{section_title}\n\n{chunk_text}" if section_title else chunk_text)

        return chunks
