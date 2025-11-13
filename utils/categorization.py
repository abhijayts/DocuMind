from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANDIDATE_LABELS = [
    "Technology", "Healthcare", "Finance", "Education",
    "Politics", "Sports", "Entertainment", "Travel", "Cooking"
]


def categorize_text(text: str) -> Tuple[str, float]:
    """
    Dynamically load zero-shot classifier, run categorization,
    then unload model and free GPU memory.
    """
    MODEL_NAME = "facebook/bart-large-mnli"

    try:
        # --- Lazy load model and tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)

        # --- Build the pipeline dynamically ---
        zero_shot_classifier = pipeline(
            task="zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        # --- Process text ---
        MAX_LEN = 1000
        truncated_text = " ".join(text.split()[:MAX_LEN])

        result = zero_shot_classifier(
            truncated_text,
            candidate_labels=CANDIDATE_LABELS,
            multi_label=False
        )

        # --- Extract top label ---
        if isinstance(result, list):
            result = result[0]

        labels = result.get("labels", []) # type: ignore
        scores = result.get("scores", []) # type: ignore
        top_label = str(labels[0]) if labels else "Unknown"
        confidence = float(scores[0]) if scores else 0.0

    finally:
        # --- Cleanup ---
        del model, tokenizer, zero_shot_classifier
        torch.cuda.empty_cache()

    return top_label, round(confidence, 3)
