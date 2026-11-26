import logging

from ollama import ChatResponse
from ollama import chat

logger = logging.getLogger(__name__)

LLM_MODEL = 'qwen3.5:4b'

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided context. "
    "Use only the information from the context below to answer the user's question. "
    "If the answer is not contained in the context, respond with: "
    "'I could not find this information in the provided documents.' "
    "Do not make up or infer information that is not present in the context. "
    # "When possible, cite the relevant part of the context to support your answer.\n\n"
    "### Context:\n"
    "<chunks_text>"
    "\n### End of Context"
)


def create_response(query_question, chunks_text=None, chunks_metadata=None):
    chunks_text = chunks_text or []

    if not chunks_text:
        logger.info("No chunks provided, skipping LLM call.")
        return "I could not find this information in the provided documents."

    joined_chunks = "\n\n".join(
        chunk.strip() for chunk in chunks_text if chunk and chunk.strip()
    )

    system_prompt = SYSTEM_PROMPT.replace('<chunks_text>', joined_chunks)

    response: ChatResponse = chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query_question},
        ],
        options={
            "temperature": 0.2,
            "num_ctx": 8192,
        },
    )

    return response['message']['content']
