import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
import pdfplumber
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import argparse

# IMPORTANT: Place the 'all-MiniLM-L6-v2-finetuned' model folder in the project root for offline use
MODEL_DIR = 'all-MiniLM-L6-v2-finetuned'

# --- Input loading and PDF extraction ---
def load_input_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_pdf_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                'page_number': i + 1,
                'text': text.strip()
            })
    return pages

def detect_sections(pdf_path):
    """Extract logical sections from a PDF using heading heuristics."""
    sections = []
    current_section = None
    current_text = []
    current_title = None
    current_page = 1
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            lines = text.split('\n')
            for idx, line in enumerate(lines):
                # Heuristic: heading if ALL CAPS, Title Case, or at top of page
                is_heading = False
                if idx == 0 or re.match(r'^[A-Z][A-Za-z0-9\-\s:,.]+$', line.strip()):
                    # Title Case or ALL CAPS
                    if line.strip() and (line.strip().isupper() or line.istitle() or len(line.strip()) < 60):
                        is_heading = True
                if is_heading:
                    # Save previous section
                    if current_section:
                        sections.append({
                            'document': os.path.basename(pdf_path),
                            'section_title': current_title,
                            'text': '\n'.join(current_text).strip(),
                            'page_number': current_page
                        })
                    # Start new section
                    current_title = line.strip()[:80]
                    current_text = []
                    current_page = i + 1
                current_text.append(line)
            # End of page
        # Save last section
        if current_title and current_text:
            sections.append({
                'document': os.path.basename(pdf_path),
                'section_title': current_title,
                'text': '\n'.join(current_text).strip(),
                'page_number': current_page
            })
    return sections

def detect_dishes(pdf_path):
    """Extract recipes/dishes as sections from a PDF using dish name heuristics."""
    import pdfplumber
    import re
    sections = []
    current_title = None
    current_text = []
    current_page = 1
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            lines = text.split('\n')
            for idx, line in enumerate(lines):
                # Heuristic: dish name is Title Case, not generic, and not too short/long
                is_dish = False
                if (
                    line.strip() and
                    not line.strip().endswith((':', '.')) and
                    2 <= len(line.strip().split()) <= 6 and
                    line.istitle() and
                    line.strip().lower() not in ['ingredients', 'instructions', 'method', 'preparation']
                ):
                    is_dish = True
                if is_dish:
                    # Save previous dish
                    if current_title and current_text:
                        sections.append({
                            'document': os.path.basename(pdf_path),
                            'section_title': current_title,
                            'text': '\n'.join(current_text).strip(),
                            'page_number': current_page
                        })
                    # Start new dish
                    current_title = line.strip()[:80]
                    current_text = []
                    current_page = i + 1
                current_text.append(line)
            # End of page
        # Save last dish
        if current_title and current_text:
            sections.append({
                'document': os.path.basename(pdf_path),
                'section_title': current_title,
                'text': '\n'.join(current_text).strip(),
                'page_number': current_page
            })
    return sections

def extract_all_sections(documents, pdf_dir):
    extracted = []
    for doc in documents:
        filename = doc['filename']
        full_path = os.path.join(pdf_dir, filename)
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} not found.")
            continue
        try:
            doc_sections = detect_sections(full_path)
            extracted.extend(doc_sections)
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
    return extracted

def extract_all_dishes(documents, pdf_dir):
    extracted = []
    for doc in documents:
        filename = doc['filename']
        full_path = os.path.join(pdf_dir, filename)
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} not found.")
            continue
        try:
            doc_sections = detect_dishes(full_path)
            extracted.extend(doc_sections)
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
    return extracted

# --- Embedding and ranking ---
def build_query(persona, job_to_be_done):
    return f"Persona: {persona}. Task: {job_to_be_done}"

def get_section_title(text):
    # Heuristic: use the first line as section title
    lines = text.split('\n')
    for line in lines:
        if line.strip():
            return line.strip()[:80]  # Truncate for brevity
    return "Untitled Section"

def analyze_sections(model, query, sections, top_k=5):
    # Prepare texts
    section_texts = [s['text'] for s in sections]
    # Generate embeddings
    query_emb = model.encode([query])
    section_embs = model.encode(section_texts)
    # Compute similarities
    sims = cosine_similarity(query_emb, section_embs)[0]
    # Rank sections
    ranked_indices = np.argsort(sims)[::-1][:top_k]
    extracted_sections = []
    subsection_analysis = []
    for rank, idx in enumerate(ranked_indices, 1):
        sec = sections[idx]
        extracted_sections.append({
            'document': sec['document'],
            'section_title': get_section_title(sec['text']),
            'importance_rank': rank,
            'page_number': sec['page_number']
        })
        # Subsection analysis: extract top 2-3 most relevant sentences
        sentences = [s for s in sec['text'].split('. ') if len(s.strip()) > 10]
        if sentences:
            sent_embs = model.encode(sentences)
            sent_sims = cosine_similarity(query_emb, sent_embs)[0]
            top_sent_idx = np.argsort(sent_sims)[::-1][:3]
            refined_text = '. '.join([sentences[i] for i in top_sent_idx])
        else:
            refined_text = sec['text'][:300]
        subsection_analysis.append({
            'document': sec['document'],
            'refined_text': refined_text,
            'page_number': sec['page_number']
        })
    return extracted_sections, subsection_analysis

def get_collection_paths(collection_num):
    base = f"Challenge_1b/Collection {collection_num}"
    return {
        'input_path': f'{base}/challenge1b_input.json',
        'pdf_dir': f'{base}/PDFs',
        'output_path': f'{base}/challenge1b_output.json'
    }

# --- Main pipeline ---
def main():
    parser = argparse.ArgumentParser(description="PDF Collection Analyzer")
    parser.add_argument('--collection', type=int, choices=[1,2,3], default=1, help='Collection number to process (1, 2, or 3)')
    args = parser.parse_args()
    paths = get_collection_paths(args.collection)
    print(f"PDF Analysis Pipeline Starting for Collection {args.collection}...")
    input_path = paths['input_path']
    pdf_dir = paths['pdf_dir']
    output_path = paths['output_path']
    if not os.path.exists(input_path):
        print(f"Input JSON {input_path} not found.")
        return
    if not os.path.exists(pdf_dir):
        print(f"PDF directory {pdf_dir} not found.")
        return
    input_data = load_input_json(input_path)
    documents = input_data.get('documents', [])
    persona = input_data.get('persona', {}).get('role', '')
    job_to_be_done = input_data.get('job_to_be_done', {}).get('task', '')
    # 1. Extract logical sections from PDFs
    extracted_sections = extract_all_sections(documents, pdf_dir)
    print(f"Extracted {len(extracted_sections)} sections from {len(documents)} documents.")
    # 2. Load the local model
    print("Loading local model from:", MODEL_DIR)
    model = SentenceTransformer(MODEL_DIR)
    print("Model loaded successfully.")
    # 3. Build query
    query = build_query(persona, job_to_be_done)
    # 4. Analyze and rank sections
    top_sections, subsection_analysis = analyze_sections(model, query, extracted_sections, top_k=5)
    # 5. Prepare output JSON
    output = {
        'metadata': {
            'input_documents': [doc['filename'] for doc in documents],
            'persona': persona,
            'job_to_be_done': job_to_be_done,
            'processing_timestamp': datetime.now().isoformat()
        },
        'extracted_sections': top_sections,
        'subsection_analysis': subsection_analysis
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Output written to {output_path}")

if __name__ == "__main__":
    main() 