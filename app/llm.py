"""Gemini LLM Module: Grounded Answer Generation using official google-genai SDK."""

import logging
from typing import Optional, List
from app.config import settings

logger = logging.getLogger(__name__)

GROUNDED_REFUSAL_MESSAGE = (
    "I couldn't find enough information about this in the provided Agentic AI knowledge base."
)

SYSTEM_GROUNDING_PROMPT = """You are a strictly grounded question-answering assistant for the Agentic AI eBook knowledge base.
You must answer the user's question ONLY using the provided context extracted from the Agentic AI PDF.

STRICT GROUNDING RULES:
1. Rely ONLY on the clear facts directly mentioned in the context below.
2. Do NOT use any outside knowledge, general training data, or assumptions.
3. Do NOT invent facts or hallucinate details.
4. If the context does NOT contain enough information to answer the question, state:
   "I couldn't find enough information about this in the provided Agentic AI knowledge base."
5. FORMATTING REQUIREMENT: Format your response using clean Markdown:
   - Put every item/bullet point on its own new line starting with `* `.
   - Use bold headers (`**Category Name:**`) for major headings.
   - Separate distinct categories or paragraphs with blank lines.
6. When possible, cite the specific page number(s) from context headers (e.g., "[Page X]").

CONTEXT FROM AGENTIC AI PDF:
{context}

USER QUESTION:
{question}
"""


class GeminiLLM:
    """Handles Google Gemini LLM API calls enforcing strict context grounding."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        # Model fallback candidate order if a specific model encounters 404/503
        self.fallback_models = [self.model_name, "gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3.6-flash"]

    def generate(self, question: str, context: str) -> str:
        """Generate answer strictly grounded in the provided context."""
        if not context or not context.strip():
            return GROUNDED_REFUSAL_MESSAGE

        prompt = SYSTEM_GROUNDING_PROMPT.format(context=context, question=question)

        # Check if a real Gemini API Key is configured
        if not self.api_key or self.api_key.startswith("mock") or self.api_key.startswith("your_"):
            logger.info("Using local grounded synthesis (No valid GEMINI_API_KEY provided in .env).")
            return self._generate_fallback(context)

        # Attempt 1: Try official google.genai SDK across model candidates
        try:
            from google import genai

            client = genai.Client(api_key=self.api_key)

            for target_model in self.fallback_models:
                # Strip models/ prefix if present
                clean_model = target_model.replace("models/", "")
                try:
                    logger.info(f"Invoking Gemini model '{clean_model}' via google.genai SDK...")
                    response = client.models.generate_content(
                        model=clean_model,
                        contents=prompt,
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as model_err:
                    logger.warning(f"Model '{clean_model}' returned error: {model_err}. Trying next candidate...")

        except Exception as e1:
            logger.warning(f"google.genai SDK failed ({e1}). Trying langchain_google_genai...")

        # Attempt 2: Try langchain_google_genai as backup
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            for target_model in self.fallback_models:
                clean_model = target_model.replace("models/", "")
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=clean_model,
                        google_api_key=self.api_key,
                        temperature=0.0,
                    )
                    response = llm.invoke(prompt)
                    if response and response.content:
                        return response.content.strip()
                except Exception as model_err:
                    logger.warning(f"langchain_google_genai model '{clean_model}' failed: {model_err}")

        except Exception as e2:
            logger.error(f"langchain_google_genai failed ({e2}). Falling back to local grounded extract.")

        return self._generate_fallback(context)

    def _generate_fallback(self, context: str) -> str:
        """Local deterministic fallback for offline execution and testing."""
        lines = [line.strip() for line in context.split("\n") if line.strip() and not line.startswith("---") and not line.startswith("[Page")]
        snippet = " ".join(lines[:3]) if lines else "Information retrieved from the Agentic AI eBook."
        if len(snippet) > 350:
            snippet = snippet[:350] + "..."
        return f"Based on the Agentic AI knowledge base: {snippet}"
