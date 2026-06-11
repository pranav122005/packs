from __future__ import annotations

import json
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class ProductVisionResult(BaseModel):
    product_name: str = Field(description="Best guess product name from the packaging.")
    brand: str = Field(description="Brand or manufacturer name if visible.")
    visible_text: list[str] = Field(default_factory=list, description="Short labels and text fragments visible on the package.")
    packaging_notes: str = Field(default="", description="Visual clues that helped identify the product.")
    confidence: float = Field(default=0.0, description="Confidence from 0 to 1.")


class RagAnswerResult(BaseModel):
    answer: str
    extracted_ingredients: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    preservatives: list[str] = Field(default_factory=list)
    additives: list[str] = Field(default_factory=list)
    dietary_flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_summary: str = Field(default="")
    confidence: float = Field(default=0.0)


class GeminiService:
    def __init__(self, api_key: str, embed_model: str, chat_model: str, vision_model: str, output_dimensionality: int = 768):
        self.api_key = api_key
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.vision_model = vision_model
        self.output_dimensionality = output_dimensionality
        self.client = genai.Client(api_key=api_key) if api_key else None

    @property
    def ready(self) -> bool:
        return self.client is not None

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        if not texts:
            return np.zeros((0, self.output_dimensionality), dtype="float32")

        response = self.client.models.embed_content(
            model=self.embed_model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self.output_dimensionality),
        )
        vectors = np.array([emb.values for emb in response.embeddings], dtype="float32")
        return vectors

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def identify_product(self, image_path: str) -> ProductVisionResult:
        if not self.ready:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = (
            "You are identifying a packaged food product from an image. "
            "Return the best possible product name and brand. Use only what is visible on the package. "
            "Also extract visible text fragments that may help with retrieval. "
            "If uncertain, make your best guess and lower the confidence."
        )

        response = self.client.models.generate_content(
            model=self.vision_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config={
                "response_format": {
                    "text": {
                        "mime_type": "application/json",
                        "schema": ProductVisionResult.model_json_schema(),
                    }
                }
            },
        )
        raw = response.text or "{}"
        return ProductVisionResult.model_validate_json(raw)

    def answer_question(self, question: str, product: dict[str, Any], contexts: list[dict[str, Any]]) -> RagAnswerResult:
        if not self.ready:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        context_text = []
        for i, item in enumerate(contexts, start=1):
            context_text.append(
                f"[Chunk {i}] Source: {item.get('source_pdf','unknown')} | Page: {item.get('page_number','?')}\n"
                f"{item.get('text','')}"
            )
        joined_context = "

".join(context_text)

        prompt = f"""
You are a strict retrieval-augmented assistant for packaged food products.

Rules:
- Use only the provided context.
- If the answer is not in the context, say that the document does not contain enough information.
- Do not invent nutrition facts, ingredients, allergens, or health claims.
- Give practical, concise, consumer-friendly language.

Product identified:
{json.dumps(product, ensure_ascii=False, indent=2)}

Retrieved context:
{joined_context}

User question:
{question}

Return JSON with:
- answer
- extracted_ingredients
- allergens
- preservatives
- additives
- dietary_flags
- warnings
- source_summary
- confidence
""".strip()

        response = self.client.models.generate_content(
            model=self.chat_model,
            contents=prompt,
            config={
                "response_format": {
                    "text": {
                        "mime_type": "application/json",
                        "schema": RagAnswerResult.model_json_schema(),
                    }
                }
            },
        )
        raw = response.text or "{}"
        return RagAnswerResult.model_validate_json(raw)

    def fallback_identify(self) -> ProductVisionResult:
        return ProductVisionResult(
            product_name="Unknown product",
            brand="Unknown brand",
            visible_text=[],
            packaging_notes="Gemini API key is not configured, so the system is running in demo mode.",
            confidence=0.0,
        )
