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


class SmartPdfChunker:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is missing! Ensure it's set in the .env file.")

        self.client = openai.OpenAI(api_key=self.openai_api_key)

    def extract_text_by_chunks(self, max_chars=2500) -> list:
        """Extract text from the PDF and split it into smaller chunks."""
        doc = fitz.open(self.pdf_path)
        full_text = []
        temp_chunk = ""

        for page in doc:
            page_text = page.get_text()
            if len(temp_chunk) + len(page_text) > max_chars:
                full_text.append(temp_chunk)
                temp_chunk = page_text  # Start new chunk
            else:
                temp_chunk += "\n" + page_text

        if temp_chunk:
            full_text.append(temp_chunk)  # Add last chunk

        doc.close()
        return full_text

    def process_chunk(self, chunk):
        """Handles GPT-4 API call for a single chunk and formats output."""
        messages = [
            {"role": "system",
             "content": "You are an expert legal document analyst. Given a legal document in its "
                        "**original language**, process it without translating or altering the original wording."},
            {"role": "user", "content": (
                    "Segment this legal document text into coherent articles. "
                    "For each chunk, extract:\n"
                    "1. The article number (e.g., L.561-1).\n"
                    "2. The title or heading of the article.\n"
                    "3. The section or grouping it belongs to.\n"
                    "4. Any subsection (if applicable).\n"
                    "5. The full content of that article.\n\n"
                    "Return ONLY a valid JSON array with this format:\n"
                    "[\n"
                    "  {\n"
                    "    \"article\": \"L.561-1\",\n"
                    "    \"title\": \"Title of the article\",\n"
                    "    \"section\": \"Relevant section\",\n"
                    "    \"subsection\": \"Subsection (if applicable, if not remove this line)\",\n"
                    "    \"content\": \"Full article text here.\"\n"
                    "  }\n"
                    "]\n\n"
                    "DO NOT include any markdown or explanations.\n\n"
                    "**Text:**\n" + chunk
            )}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.0
            )

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Received empty response from OpenAI.")

            content = response.choices[0].message.content.strip()

            # Ensure JSON format
            chunk_data = json.loads(content)

            # 🔹 Format each chunk as a **single block of text**
            formatted_chunks = []
            for item in chunk_data:
                formatted_text = (
                    f"Article: {item.get('article', 'N/A')}\n"
                    f"Title: {item.get('title', 'N/A')}\n"
                    f"Section: {item.get('section', 'N/A')}\n"
                    f"Subsection: {item.get('subsection', 'N/A')}\n\n"
                    f"{item.get('content', 'N/A')}"
                )
                formatted_chunks.append(formatted_text)

            return formatted_chunks  # Returns a list of formatted articles

        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            return []

    def chunk_document(self) -> list:
        """
        Process PDF in parallel chunks for faster performance and return formatted text.
        """
        chunks = self.extract_text_by_chunks()
        extracted_articles = []

        # 🚀 Run API calls in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # parallel threads
            results = executor.map(self.process_chunk, chunks)

        # Combine results
        for result in results:
            extracted_articles.extend(result)

        return extracted_articles


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
