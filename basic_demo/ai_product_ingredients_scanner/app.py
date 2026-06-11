from __future__ import annotations

from flask import Flask, flash, redirect, render_template, request, url_for
from dotenv import load_dotenv

from config import Config
from services.gemini_service import GeminiService
from services.pdf_ingest import load_all_documents
from services.utils import ensure_dir, is_allowed_image, save_base64_image, save_upload, safe_json_loads
from services.vector_store import load_store, search, save_store

load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
app.secret_key = Config.SECRET_KEY

for p in [Config.DATA_DIR, Config.UPLOAD_DIR, Config.VECTOR_DIR]:
    ensure_dir(p)

gemini = GeminiService(
    api_key=Config.GEMINI_API_KEY,
    embed_model=Config.EMBED_MODEL,
    chat_model=Config.CHAT_MODEL,
    vision_model=Config.VISION_MODEL,
    output_dimensionality=Config.OUTPUT_DIMENSIONALITY,
)

index, metadata = load_store(Config.INDEX_PATH, Config.META_PATH)


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        gemini_ready=gemini.ready,
        index_ready=bool(metadata),
        product=None,
        contexts=[],
        answer=None,
        error=None,
    )


@app.route("/identify", methods=["POST"])
def identify():
    global index, metadata

    try:
        file_obj = request.files.get("image_file")
        webcam_data = request.form.get("webcam_image", "").strip()

        if file_obj and file_obj.filename:
            if not is_allowed_image(file_obj.filename):
                flash("Please upload a PNG, JPG, JPEG, WEBP, or BMP image.")
                return redirect(url_for("home"))
            upload_path = save_upload(file_obj, Config.UPLOAD_DIR)
        elif webcam_data:
            upload_path = save_base64_image(webcam_data, Config.UPLOAD_DIR)
        else:
            flash("Please upload an image or capture one from the webcam.")
            return redirect(url_for("home"))

        if gemini.ready:
            product = gemini.identify_product(str(upload_path))
        else:
            product = gemini.fallback_identify()

        query_text = f"{product.product_name} {product.brand} {' '.join(product.visible_text)} {product.packaging_notes}"
        contexts = []
        if index is not None and metadata and gemini.ready:
            query_vector = gemini.embed_text(query_text)
            contexts = search(index, metadata, query_vector, top_k=Config.TOP_K)

        return render_template(
            "index.html",
            gemini_ready=gemini.ready,
            index_ready=bool(metadata),
            product=product.model_dump(),
            contexts=contexts,
            answer=None,
            error=None,
        )

    except Exception as exc:
        return render_template(
            "index.html",
            gemini_ready=gemini.ready,
            index_ready=bool(metadata),
            product=None,
            contexts=[],
            answer=None,
            error=str(exc),
        )


@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = (request.form.get("question") or "").strip()
        product_json = request.form.get("product_json") or "{}"
        contexts_json = request.form.get("contexts_json") or "[]"

        if not question:
            flash("Please type a question.")
            return redirect(url_for("home"))

        product = safe_json_loads(product_json, default={}) or {}
        contexts = safe_json_loads(contexts_json, default=[]) or []

        if not contexts:
            flash("No retrieved context was available. Upload a product image first.")
            return redirect(url_for("home"))

        answer = gemini.answer_question(question=question, product=product, contexts=contexts) if gemini.ready else None

        return render_template(
            "index.html",
            gemini_ready=gemini.ready,
            index_ready=bool(metadata),
            product=product,
            contexts=contexts,
            answer=answer.model_dump() if answer else None,
            error=None,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            gemini_ready=gemini.ready,
            index_ready=bool(metadata),
            product=None,
            contexts=[],
            answer=None,
            error=str(exc),
        )


@app.route("/rebuild-index", methods=["POST"])
def rebuild_index():
    if not gemini.ready:
        flash("Add GEMINI_API_KEY before rebuilding the index.")
        return redirect(url_for("home"))

    docs = load_all_documents(Config.DATA_DIR, chunk_size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP)
    if not docs:
        flash("No PDFs were found in the data_pdfs folder.")
        return redirect(url_for("home"))

    texts = [d.text for d in docs]
    vectors = gemini.embed_texts(texts)
    chunks = [
        {
            "text": d.text,
            "source_pdf": d.source_pdf,
            "page_number": d.page_number,
            "chunk_index": d.chunk_index,
        }
        for d in docs
    ]

    save_store(Config.INDEX_PATH, Config.META_PATH, vectors, chunks)
    global index, metadata
    index, metadata = load_store(Config.INDEX_PATH, Config.META_PATH)
    flash(f"Rebuilt index with {len(chunks)} chunks.")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
