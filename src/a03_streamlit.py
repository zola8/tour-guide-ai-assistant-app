import logging

import numpy as np
import streamlit as st
from PIL import Image as PILImage

from src.commons import client_db
from src.commons import configure_logging
from src.commons import document_embeddings
from src.commons import image_embeddings
from src.llm import create_response

configure_logging()

logger = logging.getLogger(__name__)


def classify_image_array(img_rgb: np.ndarray, max_distance: float):
    collection = client_db.get_collection(
        name="images",
        embedding_function=image_embeddings,
    )
    embeddings = image_embeddings([img_rgb])
    results = collection.query(query_embeddings=embeddings, n_results=1)

    if not results["ids"] or not results["ids"][0]:
        return None, None, None

    distance = results["distances"][0][0]
    match_id = results["ids"][0][0]
    logger.info("image match: id=%s distance=%.4f", match_id, distance)

    if distance > max_distance:
        return None, distance, match_id

    category = results["metadatas"][0][0].get("category")
    return category, distance, match_id


def get_similar_documents(query_question, img_category, max_distance):
    collection = client_db.get_collection(
        name=f"documents__{img_category}",
        embedding_function=document_embeddings,
    )
    results = collection.query(query_texts=[query_question], n_results=5)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    filtered = [(d, m, dist) for d, m, dist in zip(docs, metas, dists) if dist < max_distance]
    logger.info("category=%s retrieved=%d kept=%d", img_category, len(docs), len(filtered))
    return filtered


st.set_page_config(page_title="Image RAG Q&A", layout="wide")
st.title("Image RAG Q&A")
st.caption("Upload an image, then ask questions about the place it depicts.")

if "category" not in st.session_state:
    st.session_state.category = None
if "image_distance" not in st.session_state:
    st.session_state.image_distance = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None


def reset_state():
    st.session_state.category = None
    st.session_state.image_distance = None
    st.session_state.uploaded_name = None


col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "gif"])

    image_threshold = st.slider(
        "Image match threshold (lower = stricter)",
        min_value=0.10, max_value=0.80, value=0.35, step=0.01,
    )

    if uploaded is not None and uploaded.name != st.session_state.uploaded_name:
        reset_state()
        st.session_state.uploaded_name = uploaded.name

        try:
            pil_img = PILImage.open(uploaded).convert("RGB")
            img_rgb = np.array(pil_img)
            st.image(pil_img, caption=uploaded.name, width='stretch')

            with st.spinner("Classifying image..."):
                category, distance, match_id = classify_image_array(img_rgb, image_threshold)

            if category is None:
                st.session_state.category = None
                st.session_state.image_distance = distance
                st.warning(
                    f"Image not recognized. Closest match is {match_id} "
                    f"at distance {distance:.3f}, which is above the acceptance "
                    f"threshold ({image_threshold:.2f}). "
                    f"Ask is disabled until an image is accepted. "
                    f"Raise the threshold or upload a different image."
                )
            else:
                st.session_state.category = category
                st.session_state.image_distance = distance
                st.success(f"Recognized as {category} (distance {distance:.3f}).")
                st.caption(f"Nearest indexed image: {match_id}")
        except Exception as e:
            st.exception(e)

    if st.session_state.uploaded_name is not None:
        if st.button("Reset"):
            reset_state()
            st.rerun()

with col_right:
    st.subheader("Ask a question")

    ready = st.session_state.category is not None

    question = st.text_input(
        "Question",
        placeholder="e.g. When is the Academic department open?",
        disabled=not ready,
        key="question_input",
    )

    doc_threshold = st.slider(
        "Document retrieval threshold (lower = stricter)",
        min_value=0.10, max_value=1.50, value=0.80, step=0.05,
        disabled=not ready,
    )

    ask = st.button("Ask", type="primary", disabled=not (ready and question.strip()))

    if ask:
        with st.spinner("Retrieving context..."):
            retrieved = get_similar_documents(
                query_question=question,
                img_category=st.session_state.category,
                max_distance=doc_threshold,
            )

        if not retrieved:
            st.warning("No relevant context found for this question.")
        else:
            with st.expander(f"Retrieved {len(retrieved)} chunk(s)", expanded=False):
                for i, (doc, meta, dist) in enumerate(retrieved, 1):
                    st.markdown(f"Chunk {i} - distance {dist:.3f} - source {meta.get('source', '?')}")
                    st.write(doc)
                    st.divider()

            chunks_text = [d for d, _, _ in retrieved]
            chunks_meta = [m for _, m, _ in retrieved]

            with st.spinner("Asking the model..."):
                try:
                    answer = create_response(question, chunks_text, chunks_meta)
                except Exception as e:
                    st.exception(e)
                    answer = None

            if answer:
                st.subheader("Answer")
                st.write(answer)
