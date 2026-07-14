"""
citation_eval.py — checks whether the LLM's cited excerpt(s) actually
support its answer, for a set of labeled questions.

Retrieval eval (retrieval_eval.py) only tells you whether the right chunk
was *available* to the LLM. This script checks the next link in the chain:
given the chunks it saw, did the LLM answer correctly AND cite an excerpt
that actually contains that answer? A model can hallucinate a plausible
answer and still cite a real-looking (but wrong) excerpt number, so this
needs a human to actually read the excerpt and confirm it supports the claim.

Requires GROQ_API_KEY to be set — run this locally, not in a sandboxed
environment without network access to Groq.

Usage:
    export GROQ_API_KEY="your-key-here"
    python citation_eval.py path/to/document.pdf path/to/eval_set.json

For each question, this prints the LLM's answer, which excerpt(s) it cited,
and the actual text of those excerpts — then asks you to grade citation
accuracy on the spot (y/n) and computes an overall percentage at the end.
Grades are saved to citation_eval_results.json so you can revisit specific
failures later.
"""

import json
import os
import sys

from doc_qa import (
    extract_pages, chunk_pages, build_retriever, retrieve,
    build_prompt, ask_llm, Groq,
)


def run_eval(pdf_path, eval_set_path):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY first: export GROQ_API_KEY='your-key'")
        sys.exit(1)

    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    vectorizer, matrix = build_retriever(chunks)
    client = Groq(api_key=api_key)

    with open(eval_set_path) as f:
        eval_set = json.load(f)

    graded = []
    for i, item in enumerate(eval_set, start=1):
        question = item["question"]
        retrieved = retrieve(question, vectorizer, matrix, chunks)
        prompt = build_prompt(question, retrieved)
        answer = ask_llm(client, prompt)

        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(eval_set)}] Q: {question}")
        print(f"{'-' * 70}")
        print(f"ANSWER:\n{answer}")
        print(f"\n{'-' * 70}")
        print("RETRIEVED EXCERPTS (for you to check the citation against):")
        for c in retrieved:
            print(f"  [Excerpt {c['id']} — page {c['page']}]: {c['text'][:200]}...")

        grade = input(
            "\nDoes the cited excerpt actually support the answer? (y/n/partial): "
        ).strip().lower()

        graded.append({
            "question": question,
            "answer": answer,
            "retrieved_pages": [c["page"] for c in retrieved],
            "grade": grade,
        })

    return graded


def print_report(graded):
    n = len(graded)
    correct = sum(1 for g in graded if g["grade"] == "y")
    partial = sum(1 for g in graded if g["grade"] == "partial")

    print(f"\n{'=' * 70}")
    print(f"Citation accuracy: {correct}/{n} = {correct / n * 100:.1f}% fully correct")
    print(f"Partial credit:    {partial}/{n} = {partial / n * 100:.1f}% partially correct")

    with open("citation_eval_results.json", "w") as f:
        json.dump(graded, f, indent=2)
    print("Saved full results to citation_eval_results.json")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python citation_eval.py path/to/document.pdf path/to/eval_set.json")
        sys.exit(1)

    graded = run_eval(sys.argv[1], sys.argv[2])
    print_report(graded)
