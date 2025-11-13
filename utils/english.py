from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

def ensure_english(text: str) -> str:
    """
    Detect language and translate to English if needed.
    Handles short or empty text gracefully.
    """
    # Handle empty or short text
    if not text or len(text.strip()) < 10:
        return text

    try:
        lang = detect(text)
    except LangDetectException:
        return text

    if lang != "en":
        try:
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            return translated
        except Exception as e:
            return text

    return text
