# AI Product Ingredients Scanner and RAG Assistant

A Flask-based AI web app that:
- captures or uploads a product image with OpenCV/browser camera input,
- identifies the product using Gemini vision,
- retrieves relevant product-label chunks from a FAISS index,
- answers ingredient and nutrition questions using RAG,
- supports a PDF knowledge base built from product documents.

## Main stack
- Flask
- OpenCV
- Gemini API (`google-genai`)
- FAISS
- PyPDF
- HTML/CSS/JavaScript

## Folder structure

- `data_pdfs/` — put your product PDFs here
- `uploads/` — temporary images
- `vectorstore/` — FAISS index and metadata
- `templates/` — HTML pages
- `static/` — CSS and JavaScript
- `services/` — ingestion, vector search, and Gemini helpers

## Setup

1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your Gemini API key.
4. Put product PDFs into `data_pdfs/`.
5. Build the FAISS index:
   ```bash
   python build_index.py
   ```
6. Run the app:
   ```bash
   python app.py
   ```

## Notes

- The embedding model uses the current Gemini embedding flow with `gemini-embedding-001`.
- Image understanding is handled through Gemini vision.
- The assistant is instructed to answer only from retrieved context.

## Suggested improvements
- Barcode scanning
- EasyOCR fallback
- Multilingual UI
- User accounts
- Cloud deployment
- Mobile app wrapper
