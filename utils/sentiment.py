from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
# labels = ['Negative', 'Neutral', 'Positive']
# labels = ['Negative', 'Positive']  # For binary sentiment
LABELS = ['Anger', 'Disgust', 'Fear', 'Joy', 'Neutral', 'Sadness', 'Surprise']


def analyze_sentiment(text: str):
    """
    Load model dynamically, run sentiment analysis, 
    then unload it to free GPU VRAM.
    """
    # Load tokenizer and model
    # MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
    # MODEL_NAME = "siebert/sentiment-roberta-large-english"  # Alternative model for binary sentiment
    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

    # --- Lazy load model/tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)

    try:
        # Tokenize and infer
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

        predicted_class = np.argmax(probs)
        label = LABELS[predicted_class]
        confidence = float(probs[predicted_class])

    finally:
        # --- Cleanup ---
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()

    return label, round(confidence, 3)
