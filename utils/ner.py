import spacy

def unload_spacy(nlp):
    """Unload spaCy from memory (frees RAM and GPU)."""
    if nlp is not None:
        del nlp

nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    unload_spacy(nlp)
    return entities
