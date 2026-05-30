# -----------------------------------
# INSTALL REQUIRED LIBRARIES FIRST
# -----------------------------------
# Run these commands in terminal BEFORE running app
# pip install transformers==4.38.2
# pip install sentence-transformers==2.6.1
# pip install huggingface-hub==0.23.0
# pip install accelerate==0.27.2
# pip install streamlit
# pip install PyPDF2 nltk langchain-text-splitters scikit-learn

# -----------------------------------
# IMPORTS
# -----------------------------------
import streamlit as st
import PyPDF2
import nltk
import re
import numpy as np

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

nltk.download('punkt')

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------
st.set_page_config(
    page_title="AI PDF Assistant",
    layout="wide"
)

st.title("📘 AI Document Summarizer + Chat with PDF")

# -----------------------------------
# LOAD MODELS
# -----------------------------------
@st.cache_resource
def load_models():
    # Summary Model
    summarizer = pipeline(
        task="text-generation",
        model="google/flan-t5-base"
    )

    # QA Model
    qa_generator = pipeline(
        task="text2text-generation",
        model="google/flan-t5-large" # Note: Consider 'base' if deployment gets too slow/crashes
    )

    # Embedding Model
    embedding_model = SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

    return summarizer, qa_generator, embedding_model

summarizer, qa_generator, embedding_model = load_models()

# -----------------------------------
# BETTER CHUNKING FUNCTION
# -----------------------------------
def create_chunks(text, chunk_size=1800):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        # overlap improves context
        start += 1400
    return chunks

# -----------------------------------
# PDF UPLOAD
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload PDF File",
    type=["pdf"]
)

# -----------------------------------
# PROCESS PDF
# -----------------------------------
if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "

    # -----------------------------------
    # CLEAN TEXT
    # -----------------------------------
    cleaned_text = re.sub(r'\s+', ' ', text)

    # Remove duplicates
    lines = cleaned_text.split('.')
    unique_lines = []
    seen = set()

    for line in lines:
        line = line.strip()
        if len(line) < 20:
            continue
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    cleaned_text = ". ".join(unique_lines)

    # -----------------------------------
    # DISPLAY EXTRACTED TEXT
    # -----------------------------------
    st.subheader("📄 Extracted PDF Text")
    formatted_text = cleaned_text.replace(". ", ".\n\n")
    st.text_area(
        "PDF Content",
        formatted_text[:10000],
        height=300
    )

    # -----------------------------------
    # STUDY NOTES GENERATION
    # -----------------------------------
    st.subheader("📝 AI Generated Study Notes")
    lines = formatted_text.split("\n")
    structured_notes = ""

    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        # Detect headings
        if len(line.split()) <= 8 and line[0].isupper():
            structured_notes += f"\n\n### {line}\n\n"
        else:
            structured_notes += f"• {line}\n\n"

    st.markdown(structured_notes)

    # -----------------------------------
    # CREATE CHUNKS & EMBEDDINGS
    # -----------------------------------
    qa_chunks = create_chunks(cleaned_text)
    chunk_embeddings = embedding_model.encode(qa_chunks)
    
    # -----------------------------------
    # QUESTION ANSWERING UI
    # -----------------------------------
    st.subheader("💬 Ask Questions From PDF")
    question = st.text_input("Enter your question")

    # -----------------------------------
    # IMPROVED QUESTION ANSWERING
    # -----------------------------------
    def ask_question(question_text):
        # Safety check
        if len(qa_chunks) == 0:
            return "No document loaded."

        # Encode question
        question_embedding = embedding_model.encode([question_text])

        # Calculate similarity
        similarities = cosine_similarity(
            question_embedding,
            chunk_embeddings
        )[0]

        # Get TOP 3 relevant chunks (Note: comments originally said 8, code handles 3)
        top_indices = similarities.argsort()[-3:][::-1]

        # Build better context
        context_parts = []
        for idx in top_indices:
            if idx < len(qa_chunks):
                context_parts.append(qa_chunks[idx])

        context = "\n\n".join(context_parts)

        # STRONGER PROMPT
        prompt = f"""
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {question_text}

        Rules:
        1. Give a short and precise answer.
        2. Do not repeat sentences.
        3. Do not copy the entire document.
        4. Maximum 5 lines.
        5. If answer not found, say:
           "Answer not found in document."

        Answer:
        """

        # Generate answer
        response = qa_generator(
            prompt,
            max_length=512,
            min_length=80,
            do_sample=False
        )

        ans = response[0]['generated_text']

        # Clean generated answer sentences
        sentences = ans.split('.')
        seen_sentences = set()
        unique_sentences = []

        for s in sentences:
            s = s.strip()
            if len(s) < 5:
                continue
            if s not in seen_sentences:
                unique_sentences.append(s)
                seen_sentences.add(s)

        ans = ". ".join(unique_sentences)

        # Better formatting
        ans = ans.replace(". ", ".\n\n")
        return ans

    # -----------------------------------
    # BUTTON
    # -----------------------------------
    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating Answer..."):
                answer = ask_question(question)
                st.success("Answer Generated!")
                st.markdown(answer)