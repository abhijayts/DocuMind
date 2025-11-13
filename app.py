import streamlit as st
import spacy_streamlit
import PyPDF2
import json
import docx
import io
from collections import defaultdict, Counter
from utils.preprocessing import preprocess
from utils.ner import extract_entities
from utils.sentiment import analyze_sentiment
from utils.categorization import categorize_text
from utils.summarization_wrapper import summarize_wrapper
from utils.english import ensure_english
from utils.ner import nlp

st.set_page_config(page_title="DocuMind", layout="wide")

st.title("🧠 DocuMind")
st.write("Upload a document or paste your text below to analyze it.")

# --- Text Input ---
text_input = st.text_area("📌 Paste your text here", height=200)

# --- File Upload ---
uploaded_file = st.file_uploader("📄 Or upload a document (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

# --- Function to extract text ---
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    elif file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    return None

# --- Determine source of input ---
final_text = ""

if uploaded_file:
    final_text = extract_text(uploaded_file)
elif text_input:
    final_text = text_input

# --- Ensure English Language ---
final_text = ensure_english(final_text) # type: ignore

# --- Named Entity Recognition (NER) ---
if final_text and final_text.strip():
    st.subheader("🔎 Named Entities")

    try:
        doc = nlp(final_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        if entities:
            # --- Entity Summary ---
            ent_counter = Counter([label for _, label in entities])
            st.write("🧾 Entity Summary:", dict(ent_counter))

            # --- Grouped Entities ---
            grouped_ents = defaultdict(list)
            for ent, label in entities:
                grouped_ents[label].append(ent)

            with st.expander("📂 Entity Details"):
                for label, ents in grouped_ents.items():
                    st.markdown(f"### 🏷 {label}")
                    unique_ents = sorted(set(ents))
                    for ent in unique_ents:
                        st.markdown(f"- {ent}")

            # --- Entity Visualization ---
            st.subheader("🎨 Entity Visualization")
            try:
                from spacy import displacy
                html = displacy.render(doc, style="ent", page=False)
                st.markdown(html, unsafe_allow_html=True)
            except Exception as viz_error:
                st.warning("⚠️ Could not render entity visualization automatically.")
                st.text(f"Visualization error: {viz_error}")

        else:
            st.info("No named entities found.")

    except Exception as e:
        st.error(f"Error during NER processing: {e}")

else:
    st.warning("Please enter or upload text before analyzing.")


# --- Sentiment Analysis ---
if final_text:
    st.subheader("📈 Sentiment Analysis")
    label, confidence = analyze_sentiment(final_text)
    emoji_map = {
    "Anger": "😠",
    "Disgust": "🤮",
    "Fear": "😰",
    "Joy": "😁",
    "Neutral": "😐",
    "Sadness": "😔",
    "Surprise": "😮",
    }
    emoji = emoji_map.get(label, "😐")
    st.write(f"**Sentiment:** {label} {emoji}")
    st.write(f"**Confidence Score:** {confidence}")

# --- Text Summarization using wrapper ---
if final_text:
    st.subheader("📚 Text Summarization")
    summary = summarize_wrapper(final_text)
    st.write(f"**Summary:** {summary}")

# --- Document Categorisation ---
if final_text:
    st.subheader("📂 Document Categorisation")
    category, confidence = categorize_text(final_text)
    st.write(f"**Predicted Category:** {category} 📌")
    st.write(f"**Confidence Score:** {confidence}")

if final_text:
    output = {
        "entities": entities if final_text else [],
        "sentiment": {"label": label, "confidence": confidence},
        "summary": summary,
        "category": {"label": category, "confidence": confidence},
    }
    st.download_button(
        label="💾 Download Analysis",
        data=json.dumps(output, indent=2),
        file_name="documind_output.json",
        mime="application/json"
    )
