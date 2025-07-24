# Challenge 1b: Multi-Collection PDF Analysis

## Overview
This solution processes multiple PDF document collections and extracts the most relevant sections based on a specific persona and job-to-be-done. It is generic, robust, and meets all hackathon constraints (CPU-only, model <1GB, no internet, <60s for 3-5 docs).

## Project Structure
```
Challenge_1b/
├── Collection 1/                    # Travel Planning
│   ├── PDFs/                       # South of France guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results (gold)
├── Collection 2/                    # Adobe Acrobat Learning
│   ├── PDFs/                       # Acrobat tutorials
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results (gold)
├── Collection 3/                    # Recipe Collection
│   ├── PDFs/                       # Cooking guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results (gold)
├── all-MiniLM-L6-v2-finetuned/      # Fine-tuned model (offline)
├── pdf_analyzer.py                  # Main pipeline script
├── requirements.txt
├── Dockerfile
├── approach_explanation.md
└── README.md
```

## Setup & Execution

### **1. Requirements**
- Python 3.8+
- Docker (for containerized runs)
- No internet required at runtime (model is local)

### **2. Run with Docker (Recommended)**
```sh
docker build -t pdf-analyzer .
docker run --rm -v $(pwd):/app pdf-analyzer --collection 1
```
- Use `--collection 1`, `--collection 2`, or `--collection 3` to select which collection to process.
- Output will be written to the respective collection folder as `challenge1b_output_agent.json`.

### **3. Run Locally (Python)**
```sh
pip install -r requirements.txt
python pdf_analyzer.py --collection 1
```

## **Approach Summary**
- Extracts text and logical sections from each PDF using heading heuristics.
- Embeds persona+job and all sections using a fine-tuned MiniLM model (<1GB).
- Ranks sections by semantic similarity to the persona/task.
- Outputs top N sections and their refined summaries in the required JSON format.
- Fully offline, CPU-only, and meets all hackathon constraints.

## **Hackathon Compliance**
- **CPU-only:** No GPU required.
- **Model size:** <1GB (MiniLM, fine-tuned).
- **No internet:** Model and code are local.
- **Speed:** <60s for 3-5 documents (validated).

## **Deliverables**
- `pdf_analyzer.py` (main script)
- `requirements.txt`
- `Dockerfile`
- `approach_explanation.md`
- This `README.md`

---
**For any issues, see the comments in `pdf_analyzer.py` or contact the author.** 