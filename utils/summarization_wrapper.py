from utils.preprocessing import preprocess
from utils.summarizer import summarize_text
from typing import List
import re

# Common section headers in research papers / articles
SECTION_HEADERS = [
    "abstract", "introduction", "background", "related work", 
    "methodology", "methods", "experiments", "results", 
    "discussion", "conclusion", "future work", "references"
]

def split_into_sections(text: str) -> List[str]:
    """
    Split text into sections using headers. Returns list of section texts.
    If no headers found, returns the full text as one section.
    """
    # Create regex pattern for headers, allowing optional numbering (1., 2.1, etc.)
    pattern = r"(^\d*(\.\d+)*\s*(" + "|".join(SECTION_HEADERS) + r")\s*$)"
    # Split text by pattern (case-insensitive, multiline)
    splits = re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    # Filter out None or empty strings
    sections = [s.strip() for s in splits if s and not re.match(pattern, s, flags=re.IGNORECASE)]
    
    # If no sections detected, fallback to entire text
    if not sections:
        return [text.strip()]
    return sections

def split_into_chunks(sentences: List[str], max_sentences: int = 10) -> List[str]:
    """
    Split a list of sentences into smaller chunks (to avoid GPU OOM on long text)
    """
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks

def summarize_wrapper(text: str, max_sentences_per_chunk: int = 10) -> str:
    """
    Full pipeline:
    1. Split into sections
    2. Preprocess section text
    3. Split into sentence chunks if long
    4. Summarize each chunk
    5. Merge chunk summaries per section
    6. Merge all section summaries into final summary
    """
    sections = split_into_sections(text)
    all_section_summaries = []

    for section_text in sections:
        # Preprocess and get sentences
        sentences = preprocess(section_text, return_sentences=True)
        if not sentences:
            continue

        # Determine length
        total_text = " ".join(sentences)
        token_estimate = len(total_text.split())

        if token_estimate < 512:
            section_summary = summarize_text(total_text)
        elif token_estimate < 2000:
            section_summary = summarize_text(total_text)
        else:
            chunks = split_into_chunks(sentences, max_sentences=max_sentences_per_chunk) # type: ignore
            chunk_summaries = [summarize_text(chunk) for chunk in chunks]
            merged_text = " ".join(chunk_summaries)
            # optional re-summary if merged section is still long
            if len(merged_text.split()) > 500:
                section_summary = summarize_text(merged_text)
            else:
                section_summary = merged_text

        all_section_summaries.append(section_summary)

    # Merge all section summaries
    final_summary = " ".join(all_section_summaries)
    return final_summary
