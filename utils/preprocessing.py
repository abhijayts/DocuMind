import spacy
import re
import gc
from typing import List

def get_spacy_model():
    """Lazily load spaCy model for efficiency."""
    nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def remove_reference_sections(text: str) -> str:
    """
    Removes References, Bibliography, Acknowledgements, or Appendix sections entirely.
    """
    # Match patterns like "References", "Bibliography", "Acknowledgments", "Appendix"
    ref_pattern = re.compile(
        r"(?:(?:^|\n)(?:\d*\s*(?:references|bibliography|acknowledgments?|appendix|works cited)\s*:?)[\s\S]*)",
        re.IGNORECASE
    )
    cleaned_text = re.sub(ref_pattern, "", text)
    return cleaned_text.strip()


def clean_text(text: str) -> str:
    """
    Clean text for summarization: remove URLs, emails, citations, section headers,
    figure/table references, digits, and normalize spacing.
    """
    # --- Remove reference sections ---
    text = remove_reference_sections(text)

    # --- Remove URLs, emails, mentions, hashtags ---
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", " ", text)

    # --- Remove inline citations: [1], (Smith et al., 2020), etc. ---
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", text)
    text = re.sub(r"\([A-Za-z ,;&]+ et al\., \d{4}\)", " ", text)
    text = re.sub(r"\([A-Za-z ,;&]+, \d{4}\)", " ", text)

    # --- Remove section headers like "1 Introduction" or "2.1 Methodology" ---
    text = re.sub(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z\s]{2,50}$", " ", text, flags=re.MULTILINE)

    # --- Remove figure/table/equation references (e.g. Fig. 1, Table 2) ---
    text = re.sub(r"(Fig|Figure|Table|Eq|Equation)\.?\s*\(?\d+\)?", " ", text)

    # --- Remove page numbers and footers ---
    text = re.sub(r"page\s*\d+", " ", text, flags=re.IGNORECASE)

    # --- Remove digits, stray special chars ---
    text = re.sub(r"[^A-Za-z\s]", " ", text)

    # --- Normalize spacing ---
    text = re.sub(r"\s+", " ", text)

    # Remove IEEE-style reference block lines:
    text = re.sub(r"^\s*\[\d+\]\s+.*?(?=\n\[|\Z)", " ", text, flags=re.MULTILINE | re.DOTALL)

    # Remove lines that consist mostly of non-letters (corruption)
    text = re.sub(r"^[^A-Za-z0-9]{5,}$", " ", text, flags=re.MULTILINE)

    # Remove repeated sequences of random characters
    text = re.sub(r"\b([a-zA-Z]{2,})\1{2,}\b", " ", text)

    # Remove long sequences of random consonants
    text = re.sub(r"\b[b-df-hj-np-tv-z]{5,}\b", " ", text, flags=re.IGNORECASE)

    # Remove lines of garbage (min length 20, low vowel count → likely junk)
    def is_garbage(line):
        letters = re.sub(r'[^A-Za-z]', '', line)
        if len(letters) < 20:
            return False
        vowel_ratio = sum(c in "aeiouAEIOU" for c in letters) / len(letters)
        return vowel_ratio < 0.2

    cleaned_lines = []
    for line in text.split("\n"):
        if not is_garbage(line):
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    return text.strip().lower()


def tokenize_and_lemmatize(text: str) -> List[str]:
    """Tokenize and lemmatize using spaCy, remove stopwords."""
    nlp = get_spacy_model()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    del nlp, doc
    gc.collect()
    return tokens


def preprocess(text: str, return_sentences: bool = False):
    """
    Preprocess text for DocuMind pipeline:
    - Removes references, citations, and noisy text.
    - Optionally returns cleaned sentences for chunked summarization.
    """
    cleaned = clean_text(text)
    nlp = get_spacy_model()

    if return_sentences:
        doc = nlp(cleaned)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 3]
        del nlp, doc
        gc.collect()
        return sentences

    tokens = tokenize_and_lemmatize(cleaned)
    del nlp
    gc.collect()
    return " ".join(tokens)
