"""
pipeline.py
-----------
Core RAG pipeline for mortgage document intelligence.

Stages:
    1. PDF ingestion and OCR (PyMuPDF + docTR + Tesseract fallback)
    2. LLM-based document classification and boundary detection
    3. Chunking with LlamaIndex SentenceSplitter
    4. Embedding with sentence-transformers/all-MiniLM-L6-v2
    5. Hybrid indexing: FAISS (dense) + BM25 (sparse)
    6. Query expansion and RRF retrieval fusion
    7. Mistral LLM answer generation with source attribution

Usage:
    See notebooks/rag_pipeline.ipynb for full interactive walkthrough.
    This module exposes the core classes for programmatic use.
"""

# Full implementation available in notebooks/rag_pipeline.ipynb
# Modular extraction in progress
