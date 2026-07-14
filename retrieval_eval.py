"""
retrieval_eval.py — measures retrieval quality for DocQA's TF-IDF retriever.

Answers the question "when I ask something, does the retriever actually
surface the chunk that contains the answer?" — independent of whatever the
LLM does with those chunks afterward. Retrieval failures cap the ceiling on
answer quality no matter how good the generation prompt is, so this is worth
tracking on its own.

Usage:
    python retrieval_eval.py path/to/document.pdf path/to/eval_set.json

eval_set.json format:
    [
      {"question": "How do I sort ascending and descending?", "expected_page": 7},
      ...
    ]

Reports:
    - Top-K hit rate: was the expected page anywhere in the top-K retrieved chunks?
    - Top-1 hit rate: was the expected page the single highest-ranked chunk?
    - Per-question pass/fail with the pages actually retrieved, so misses are
      inspectable rather than just a summary number.
"""

import json
import sys

from doc_qa import extract_pages, chunk_pages, build_retriever, retrieve, TOP_K


def run_eval(pdf_path, eval_set_path, top_k=TOP_K):
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    vectorizer, matrix = build_retriever(chunks)

    with open(eval_set_path) as f:
        eval_set = json.load(f)

    results = []
    for item in eval_set:
        question = item["question"]
        expected_page = item["expected_page"]

        retrieved = retrieve(question, vectorizer, matrix, chunks, top_k=top_k)
        retrieved_pages = [c["page"] for c in retrieved]

        results.append({
            "question": question,
            "expected_page": expected_page,
            "retrieved_pages": retrieved_pages,
            "topk_hit": expected_page in retrieved_pages,
            "top1_hit": retrieved_pages[0] == expected_page if retrieved_pages else False,
        })

    return results


def print_report(results, top_k):
    n = len(results)
    topk_hits = sum(r["topk_hit"] for r in results)
    top1_hits = sum(r["top1_hit"] for r in results)

    print(f"Top-{top_k} retrieval accuracy: {topk_hits}/{n} = {topk_hits / n * 100:.1f}%")
    print(f"Top-1 retrieval accuracy:      {top1_hits}/{n} = {top1_hits / n * 100:.1f}%\n")

    for r in results:
        status = "HIT " if r["topk_hit"] else "MISS"
        print(f"[{status}] expected p{r['expected_page']:>2} | "
              f"retrieved {r['retrieved_pages']} | {r['question']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python retrieval_eval.py path/to/document.pdf path/to/eval_set.json")
        sys.exit(1)

    results = run_eval(sys.argv[1], sys.argv[2])
    print_report(results, TOP_K)
