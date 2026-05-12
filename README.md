# Drawing Spec Compliance

AI-powered PDF drawing analysis pipeline for technical specification validation.

The system processes construction and engineering PDFs, separates individual drawings from multi-page albums, extracts text and visual information, retrieves relevant specification chunks with RAG, and generates structured compliance reports using local LLMs and deterministic validation rules.

Designed for offline/local workflows where auditability and controlled validation are more important than generic AI responses.

---

## Features

* PDF drawing extraction and album splitting
* OCR with Tesseract + EasyOCR fallback
* Multimodal drawing understanding with LLaVA
* FAISS RAG retrieval over technical specifications
* Structured JSON compliance reports
* Deterministic rule-based validation
* Local Ollama-powered inference
* Multi-drawing PDF support
* Heuristic fallback when LLM is unavailable

---

## Workflow

```text
PDF / Album
      ↓
Drawing Detection
      ↓
Text + OCR Extraction
      ↓
Vision Analysis (LLaVA)
      ↓
RAG Retrieval (FAISS)
      ↓
LLM Compliance Check
      ↓
Rule Validation
      ↓
Structured JSON Report
```

---

## Tech Stack

* Python
* PyMuPDF
* Tesseract OCR
* EasyOCR
* Ollama
* LLaVA
* Mistral
* FAISS
* sentence-transformers
* Pydantic

---

## Example Use Cases

* Construction drawing validation
* Technical specification compliance
* Engineering document QC
* Supplier drawing review
* Automated document auditing

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/drawing-spec-compliance.git

cd drawing-spec-compliance

pip install -r requirements.txt
```

---

## Build Specification Index

```bash
python -m scripts.index_rag --build-tz
```

---

## Usage

Analyze single PDF:

```bash
python main.py --pdf drawing.pdf --drawing-id A-100
```

Analyze multi-page album:

```bash
python main.py --pipeline-album --pdf album.pdf
```

Check Ollama connection:

```bash
python main.py --check-ollama
```

---

## Example Output

```json
{
  "drawing_id": "A-101",
  "is_compliant": false,
  "issues": [
    {
      "field": "wall_thickness",
      "expected": "200mm",
      "actual": "160mm"
    }
  ]
}
```

---

## Architecture

![Architecture](assets/architecture.png)

---

## Project Structure

```text
scripts/        # OCR, RAG, validation, LLM pipeline
config/         # Rules and validation settings
docs/           # Technical specifications
tests/          # Unit tests
data/           # Generated indexes and outputs
```

---

## Notes

* Fully local/offline-capable pipeline
* No cloud APIs required
* Designed for auditable engineering workflows
* Deterministic validation layer independent from the LLM

---

## License

MIT
