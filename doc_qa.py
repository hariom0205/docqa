"""
DocQA — ask questions about a PDF and get answers grounded in the document,
with citations back to the exact excerpt used.

Why this project exists: this mirrors how insurance/legal teams need answers
sourced directly from dense documents (policies, quotes) rather than an LLM's
best guess. Retrieval (TF-IDF) finds the relevant text; the LLM is only
allowed to answer from what it's given, and must say which excerpt(s) it used.

Usage:
    export GROQ_API_KEY="your-key-here"
    python doc_qa.py path/to/document.pdf
"""

import os
import sys
import textwrap

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

CHUNK_WORDS = 220        # words per chunk
CHUNK_OVERLAP = 40       # overlap so we don't split a sentence's context in half
TOP_K = 4                # how many chunks to hand the LLM per question
MODEL = "llama-3.3-70b-versatile"


def extract_pages(pdf_path):
    """Return a list of (page_number, text) tuples."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_pages(pages):
    """
    Split page text into overlapping word-chunks.
    Each chunk remembers which page it came from, for citation purposes.
    """
    chunks = []  # list of dicts: {id, page, text}
    for page_num, text in pages:
        words = text.split()
        start = 0
        while start < len(words):
            end = start + CHUNK_WORDS
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "id": len(chunks),
                "page": page_num,
                "text": chunk_text,
            })
            if end >= len(words):
                break
            start = end - CHUNK_OVERLAP
    return chunks


def build_retriever(chunks):
    """Fit a TF-IDF vectorizer over all chunks. Returns (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return vectorizer, matrix


def retrieve(question, vectorizer, matrix, chunks, top_k=TOP_K):
    """Return the top_k chunks most relevant to the question."""
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix).flatten()
    ranked_idx = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in ranked_idx]


def build_prompt(question, retrieved_chunks):
    excerpts = "\n\n".join(
        f"[Excerpt {c['id']} — page {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    )
    return f"""You are a careful assistant that answers ONLY using the excerpts below.
If the excerpts don't contain the answer, say so plainly — do not guess.
After your answer, list which excerpt number(s) you used, like: Sources: [2, 4]

EXCERPTS:
{excerpts}

QUESTION: {question}

ANSWER:"""


def ask_llm(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


def main():
    if len(sys.argv) != 2:
        print("Usage: python doc_qa.py path/to/document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY first: export GROQ_API_KEY='your-key'")
        sys.exit(1)

    print(f"Reading {pdf_path} ...")
    pages = extract_pages(pdf_path)
    if not pages:
        print("No extractable text found in this PDF (it may be scanned/image-only).")
        sys.exit(1)

    chunks = chunk_pages(pages)
    print(f"Split into {len(chunks)} chunks across {len(pages)} pages.")

    vectorizer, matrix = build_retriever(chunks)
    client = Groq(api_key=api_key)

    print("\nAsk questions about the document. Type 'quit' to exit.\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        retrieved = retrieve(question, vectorizer, matrix, chunks)
        prompt = build_prompt(question, retrieved)
        answer = ask_llm(client, prompt)

        print("\n" + textwrap.fill(answer, width=88))
        print("\n(Retrieved from pages:", sorted(set(c["page"] for c in retrieved)), ")\n")


if __name__ == "__main__":
    main()
