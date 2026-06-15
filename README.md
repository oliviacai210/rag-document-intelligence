# RAG Document Intelligence Pipeline
**AI-powered document Q&A system for multi-document mortgage blobs -- from raw PDF to cited answers in under 3 seconds.**

Built during a Data and AI Externship at Outamation (Jan 2026 to Apr 2026). The pipeline automatically segments, classifies, and indexes 200+ page mortgage documents, then answers natural-language queries with source citations and page references.

---

## The Problem

Mortgage loan packages are 100 to 200 page PDFs bundling 10+ document types (contracts, fee sheets, pay slips, deeds, and more) into a single scanned or digital file. Lenders and underwriters spend 30 to 45 days per loan in manual review, creating costly bottlenecks and compliance risk.

| Pain Point | How This Pipeline Solves It |
|---|---|
| Manual review of 200-page document blobs | Automated segmentation and LLM boundary detection across 11 document types |
| Poor OCR quality on scanned pages | Two-stage OCR: docTR (primary) and Tesseract fallback at 200 DPI |
| No structured answers or source attribution | Hybrid retrieval (FAISS and BM25) with Mistral LLM and mandatory citations |
| Queries require knowing which document to search | Auto-routing: LLM classifies query intent and routes to the correct document type |

---

## System Architecture

```
PDF Upload
    |
    v
PyMuPDF Text Extraction + docTR OCR (scanned pages)
    |
    v
LLM Document Classification and Boundary Detection (11 document types)
    |
    v
LlamaIndex SentenceSplitter Chunking (500 tokens, 100 overlap)
    |
    v
Sentence Transformer Embeddings (all-MiniLM-L6-v2, 384-dim)
    |
    v
FAISS Vector Index (per-doc-type sub-indices) + BM25 Sparse Index
    |
    v
Query Expansion (3 LLM variants) -> RRF Fusion (alpha=0.6 semantic weight)
    |
    v
Mistral LLM Answer Generation with Source Attribution
    |
    v
Gradio UI (document upload, chat interface, doc-type filter)
```

---

## Benchmark Results

Evaluated on 15 human-annotated questions across a 57-page multi-document mortgage blob (11 digital and 4 scanned page questions).

### Retrieval Performance (K=4)

| Metric | Overall | Digital Docs | Scanned Docs |
|---|---|---|---|
| Recall@4 | 60.0% | 81.8% | 0.0% |
| Mean Reciprocal Rank (MRR) | 0.555 | 0.757 | 0.000 |
| Precision@4 | 53.3% | 72.7% | 0.0% |
| Hit Rate | 60.0% | 81.8% | 0.0% |

### End-to-End Accuracy

| Metric | Result |
|---|---|
| Answer Accuracy (overall) | 53.3% (n=15) |
| Answer Accuracy (digital docs only) | 72.7% |
| Citation Accuracy | 100.0% (10/10 evaluated queries) |
| Factual Consistency | 93.3% (1 hallucination in 15 queries) |

### System Performance

| Metric | Result |
|---|---|
| Average Response Time | 2.59s overall (2.55s digital, 2.72s scanned) |
| PDF Processing Speed | 4.48 seconds per page (255.4s for 57 pages) |
| Retrieval Latency | ~2 to 4s (includes LLM query expansion) |
| LLM Generation | ~1 to 2s (Mistral API round-trip) |

---

## Pipeline in Action

**Example query (digital document, successful retrieval):**

> "According to the fees worksheet, what is the value of the estimated monthly payments?"

System retrieved 4 relevant chunks from the Fees Worksheet and returned:

> "According to the Fees Worksheet (Pages 48 to 48), the estimated monthly payment is $1,869.37."

Answer Accuracy: Correct | Citation Accuracy: Correct | Response Time: 2.2s

**Known limitation (scanned document):**

> "Who is the mortgage applicant in mortgage document #10009588?"

Retrieved 0/4 relevant chunks. Root cause: scanned page OCR output was insufficient to produce embeddable text. This accounts for the 0% scanned Recall@4. See Limitations section for the improvement roadmap.

---

## Tech Stack

| Component | Technology | Configuration |
|---|---|---|
| OCR | docTR (primary) and Tesseract (fallback) | DB-ResNet50 detection, CRNN recognition, 200 DPI rendering |
| PDF Parsing | PyMuPDF (fitz), PyPDF2 | Text extraction with scanned-page detection |
| Chunking | LlamaIndex SentenceSplitter | 500 tokens, 100 overlap, sentence-boundary-aware |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 384-dim vectors, local inference, no API cost |
| Vector Store | FAISS (IndexFlatL2) | Per-doc-type sub-indices, in-memory |
| Sparse Retrieval | BM25Okapi | Keyword retrieval for mortgage-specific terms |
| Retriever | Hybrid FAISS and BM25 with Query Expansion and RRF | K=4 default, alpha=0.6, 3 LLM query variants, k=60 RRF constant |
| LLM | Mistral (mistral-small-latest) via LlamaIndex | Domain-tuned mortgage prompts, context-grounded answers only |
| UI | Gradio 3.50.2 | PDF upload, chat interface, document-type filter slider |

---

## Design Decisions

All components were selected to optimize for accuracy, explainability, and portability in a Colab environment: no GPU required, no paid vector database, no proprietary embeddings.

**Why hybrid retrieval (FAISS and BM25)?**
BM25 captures keyword-specific mortgage terms (exact fee names, parcel IDs, loan numbers) that dense semantic search misses. The ~1 to 2s additional latency from RRF fusion is justified by the recall improvement on domain-specific queries.

**Why all-MiniLM-L6-v2 over OpenAI embeddings?**
Local inference means no API cost or rate limits during batch evaluation. Trade-off: 384-dim vectors versus 1536-dim may miss nuanced semantic similarity in complex multi-clause mortgage language.

**Why FAISS over ChromaDB or Pinecone?**
Zero-dependency in-memory operation. Sensitive mortgage documents stay local. ChromaDB adds persistence (a production priority) but adds a server dependency.

---

## Known Limitations

**Scanned document OCR quality:** All 4 scanned-page queries returned 0% Recall@4. OCR text was insufficient to produce embeddable chunks. Upgrading docTR and evaluating alternative OCR configurations is the immediate next step.

**Chunk boundary fragmentation:** 500-token chunks may split financial figures from their labels. A hybrid approach with smaller dedicated chunks for numerical fields is planned.

**Session-only state:** The FAISS index is rebuilt on every session restart. Processing a 57-page document takes 255 seconds. Production use requires a persistent vector store and async processing.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/oliviacai210/rag-document-intelligence.git
cd rag-document-intelligence

# Install dependencies
pip install -r requirements.txt

# Set your Mistral API key
export MISTRAL_API_KEY=your_key_here

# Open the notebook
jupyter notebook notebooks/rag_pipeline.ipynb
```

---

## Repo Structure

```
rag-document-intelligence/
├── notebooks/
│   └── rag_pipeline.ipynb       # Full pipeline notebook
├── src/
│   ├── pipeline.py              # Core pipeline: OCR, chunking, indexing, retrieval
│   └── evaluate.py              # Evaluation: Recall@K, MRR, Precision@K, Hit Rate
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Future Work

**Short-term:** Reduce chunk size to 300 tokens for financial fields; add SentenceTransformerRerank as a third retrieval stage to push MRR above 0.8 on digital documents. Benchmark alternative OCR engines including PaddleOCR against the current docTR and Tesseract setup to identify a more reliable solution for scanned document extraction, with the goal of closing the 0% scanned Recall@4 gap.

**Medium-term:** Integrate ChromaDB for persistent document storage; expand test set to 50+ questions across all 11 document types.

**Long-term:** Deploy as a secure web app with per-user document vaults, audit logging, and LOS API integration for mortgage lender workflows.

---

## Author

**Olivia Cai**
M.S. Business Analytics (STEM), USC Marshall School of Business
B.A. International Business, UC San Diego (Cum Laude, Phi Beta Kappa)

[LinkedIn](https://www.linkedin.com/in/oliviaelizabethcai) | [GitHub](https://github.com/oliviacai210) | [Email](mailto:ocai@marshall.usc.edu)
