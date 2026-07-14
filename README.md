# DocQA — Grounded Document Q&A with Citations

Ask questions about a PDF and get answers sourced directly from the document,
with citations back to the exact page/excerpt used — instead of an LLM
answering from memory or guessing.

## Why I built this

Teams working with dense documents (insurance policies, contracts, reports)
need answers they can trust and verify, not just a plausible-sounding
paragraph. This project applies retrieval-augmented generation (RAG) at a
small scale: a simple TF-IDF retriever finds the most relevant chunks of a
PDF for a given question, and the LLM is instructed to answer *only* from
those chunks and to cite which ones it used.

## How it works

1. **Extract** — `pypdf` pulls text out of each page of the PDF.
2. **Chunk** — text is split into ~220-word overlapping chunks, each tagged
   with its source page number.
3. **Retrieve** — `scikit-learn`'s TF-IDF + cosine similarity finds the
   chunks most relevant to the question (no embedding API needed, runs
   instantly and for free).
4. **Answer** — the top chunks are handed to an LLM (via Groq's free API)
   with a prompt that forces it to answer only from what it's given and to
   cite the excerpt numbers it relied on.

## Evaluation

Retrieval and generation were measured separately against a 20-question
labeled eval set (`eval_set.json`), using `retrieval_eval.py` and
`citation_eval.py`.

**Retrieval (TF-IDF, top-4 chunks):**
- Top-4 hit rate: 70% (14/20) — the correct chunk was somewhere in the top 4
- Top-1 hit rate: 40% (8/20) — the correct chunk was ranked highest

Failures clustered around questions phrased in natural language that didn't
lexically overlap with the SQL-syntax source text (e.g. "sort ascending and
descending" vs. the doc's literal `ORDER BY ... DESC`) — a known limitation
of bag-of-words retrieval, and the motivation for the embeddings next step
below.

**Citation accuracy (generation):**
- 20/20 answers were graded correct or appropriately abstained
- On the 3 questions where retrieval missed the right chunk, the LLM
  correctly said the excerpts didn't contain the answer instead of
  hallucinating a plausible-sounding one — validating the grounding prompt
  in `build_prompt()`, which explicitly instructs the model not to guess.

Full per-question results are in `citation_eval_results.json`. Re-run the
eval on a new document with:

```bash
python retrieval_eval.py path/to/document.pdf eval_set.json
python citation_eval.py path/to/document.pdf eval_set.json
```

(`eval_set.json` is written against `SQL Query Interview Questions.pdf` —
swap in your own document and questions to re-run against something else.)

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key at [console.groq.com](https://console.groq.com) (no
credit card required), then:

```bash
export GROQ_API_KEY="your-key-here"
python doc_qa.py path/to/document.pdf
```

Then just ask questions in the terminal. Type `quit` to exit.

## Example

```
Q: What is the deductible for water damage?

The deductible for water damage is $1,000 per occurrence, as stated in the
policy's Property Coverage section.

Sources: [3]
(Retrieved from pages: [4])
```

## Possible next steps

- Swap TF-IDF for real embeddings (e.g. `sentence-transformers`) for
  better semantic matching
- Add a simple web UI (Streamlit/Flask) instead of a CLI
- Support multiple documents at once and show which document an answer
  came from
- Highlight the exact sentence within the cited excerpt, not just the chunk
