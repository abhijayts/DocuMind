from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LEDForConditionalGeneration,
    pipeline
)
import torch
import re
import textwrap


# --- Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------
#  Model Management Utilities
# --------------------------------------------------------------------
def load_model(model_name: str):
    """Load model dynamically (LED handled separately)."""
    if "led" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LEDForConditionalGeneration.from_pretrained(model_name).to(DEVICE) # type: ignore
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

    # Optional: use half precision for large models to save VRAM
    if torch.cuda.is_available() and "led" in model_name.lower():
        model.half()

    return tokenizer, model


def unload_model(model, tokenizer):
    """Free GPU memory after model use."""
    del model, tokenizer
    torch.cuda.empty_cache()


# --------------------------------------------------------------------
#  Text Handling Utilities
# --------------------------------------------------------------------
def chunk_text(text, tokenizer, max_tokens=800, overlap=100):
    """Split text into overlapping token chunks."""
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks


# --------------------------------------------------------------------
#  Main Summarization Function
# --------------------------------------------------------------------
def summarize_text(text: str) -> str:
    """Summarize text dynamically depending on length."""

    # --- Handle very short text with paraphraser ---
    if len(text.split()) < 20:
        try:
            paraphraser = pipeline(
                "text2text-generation",
                model="eugenesiow/bart-paraphrase",
                tokenizer="eugenesiow/bart-paraphrase",
                device=0 if torch.cuda.is_available() else -1
            )
            result = paraphraser(
                text, max_new_tokens=50, num_return_sequences=1, num_beams=4
            )[0]['generated_text']
            del paraphraser
            torch.cuda.empty_cache()
            return result.strip()
        except Exception:
            return text.strip()

    # --- Estimate token count with a lightweight tokenizer ---
    temp_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", use_fast=True)
    rough_token_count = len(temp_tokenizer.encode(text))
    del temp_tokenizer
    torch.cuda.empty_cache()

    # --- Model Selection ---
    if rough_token_count < 512:
        model_name, max_length, min_length = "sshleifer/distilbart-cnn-12-6", 80, 10
    elif rough_token_count < 2000:
        model_name, max_length, min_length = "facebook/bart-large-cnn", 300, 50
    else:
        model_name, max_length, min_length = "allenai/led-large-16384", 600, 100

    # --- Load Model Dynamically ---
    tokenizer, model = load_model(model_name)

    # ----------------------------------------------------------------
    # Long-form (LED or large text)
    # ----------------------------------------------------------------
    if "led" in model_name.lower() or rough_token_count > 1024:
        chunks = chunk_text(text, tokenizer, max_tokens=1024)
        summaries = []

        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding="longest"
            ).to(DEVICE)

            # LED-specific generation kwargs
            generation_kwargs = dict(
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

            if "led" in model_name.lower():
                global_attention_mask = torch.zeros_like(inputs["input_ids"])
                global_attention_mask[:, 0] = 1  # mark first token for global attention
                generation_kwargs["global_attention_mask"] = global_attention_mask # type: ignore

            with torch.no_grad():
                summary_ids = model.generate(**inputs, **generation_kwargs) # type: ignore

            summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
            torch.cuda.empty_cache()

        final_summary = " ".join(summaries)

    # ----------------------------------------------------------------
    # Short / Medium (BART / DistilBART)
    # ----------------------------------------------------------------
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="longest"
        ).to(DEVICE)

        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # ----------------------------------------------------------------
    # Post-process and Cleanup
    # ----------------------------------------------------------------
    unload_model(model, tokenizer)
    return final_summary
