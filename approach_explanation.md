# Methodology Overview

This solution is designed to act as an **intelligent document analyst**, capable of extracting and prioritizing the most relevant sections from a diverse collection of PDFs based on a specific *persona* and their *job-to-be-done*. The approach is fully generic, robust, and meets all hackathon constraints:

- CPU-only
- Model size <1GB
- No internet required
- Processes 3–5 documents in under 60 seconds

---

## Pipeline Steps

### 1. Input Parsing
The system reads a **JSON configuration** specifying:
- The persona
- The job-to-be-done
- A list of PDF documents to process

### 2. PDF Section Extraction
Each PDF is parsed using **pdfplumber**. Instead of treating each page as a section, the solution uses **heading heuristics** (such as:
- ALL CAPS
- Title Case
- Lines at the top of a page

to split the document into logical sections. Each section includes:
- Title
- Text content
- Starting page number

### 3. Embedding Generation
Both the **persona+job query** and all **extracted sections** are embedded using a **fine-tuned version** of the `all-MiniLM-L6-v2` model from Hugging Face.

- Model size: <1GB
- Optimized for CPU
- Fine-tuned using **triplet loss** on provided collections to boost relevance for specific use cases

### 4. Relevance Ranking
The system computes the **cosine similarity** between the query embedding and each section embedding. Then:
- Sections are ranked by relevance score
- Top N sections (typically 5) are selected

### 5. Subsection Analysis
For each top-ranked section:
- The system **extracts or summarizes** the most relevant sentences
- Provides a **concise and focused** answer for the persona’s task

### 6. Output Formatting
Results are structured in a **standard JSON format**, including:
- Metadata
- Extracted sections (with document, title, page number, and importance rank)
- Subsection-level analysis

---

## Key Features

- **Persona-Driven**: Tailors extraction and ranking to the persona and job-to-be-done.
- **Generic & Domain-Agnostic**: Works for any document type, persona, or task.
- **Efficient & Compliant**: Runs on CPU, model <1GB, processes 3–5 documents in <60 seconds.
- **Offline-Ready**: All models and code are local; no internet required at runtime.

---

## Why This Approach?

This pipeline leverages **state-of-the-art semantic search techniques** while staying lightweight and efficient.

By combining:
- Logical section detection
- Fine-tuned semantic ranking

…it achieves high accuracy and generalizability across diverse document types and tasks.

The **modular design** also enables easy extension to new collections or personas in the future.
