import logging
from .prompt import SYSTEM_PROMPT, build_context

log = logging.getLogger(__name__)


def generate_answer(question: str, evidence: list) -> str:
    if not evidence:
        return "I couldn't find sufficient evidence in the provided documents to answer that reliably."
    from ..config import settings
    if not settings.gemini_api_key:
        return "Gemini is not configured. Add GEMINI_API_KEY to .env to generate a grounded answer."
    try:
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(model=settings.gemini_model, contents=f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nRetrieved evidence:\n{build_context(evidence)}")
        return response.text or "I couldn't generate an answer from the supplied evidence."
    except Exception as exc:
        log.exception("Gemini generation failed")
        return f"Gemini generation failed: {exc}"
