import logging
import os
from hashlib import sha1

import numpy as np
from PIL import Image as PILImage
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.commons import DATA_PATH
from src.commons import client_db
from src.commons import configure_logging
from src.commons import document_embeddings
from src.commons import image_embeddings

configure_logging()

logger = logging.getLogger(__name__)

os.environ["UNSTRUCTURED_DISABLE_NOTICE"] = "1"


def ingest_images():
    extensions = (".jpg", ".png", ".jpeg", ".gif")

    image_collection = client_db.get_or_create_collection(
        name="images",
        embedding_function=image_embeddings,
    )

    for _dir in os.listdir(DATA_PATH):
        dir_path = os.path.join(DATA_PATH, _dir)
        if not os.path.isdir(dir_path):
            continue
        names = [n for n in os.listdir(dir_path) if n.lower().endswith(extensions)]
        if not names:
            continue

        ids, imgs, metas = [], [], []
        for name in names:
            p = os.path.join(dir_path, name)
            try:
                arr = np.array(PILImage.open(p).convert("RGB"))
            except Exception:
                logger.exception("Skipping unreadable image %s", p)
                continue
            ids.append(f"{_dir}--{name}")
            imgs.append(arr)
            metas.append({"category": _dir, "filename": name})

        if not imgs:
            continue

        image_collection.add(ids=ids, images=imgs, metadatas=metas)
        logger.info("'images': added %d from %s", len(ids), _dir)

    return image_collection


def ingest_documents():
    for _dir in os.listdir(DATA_PATH):
        dir_path = os.path.join(DATA_PATH, _dir)
        if not os.path.isdir(dir_path):
            continue

        document_collection = client_db.get_or_create_collection(
            name=f"documents__{_dir}",
            embedding_function=document_embeddings,
        )

        documents = DirectoryLoader(path=dir_path, glob="*.txt").load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            length_function=len,
            add_start_index=True,
        )

        chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]
        if not chunks:
            logger.warning("No documents found in %s, skipping", _dir)
            continue

        ids = [
            sha1(f"{chunk.metadata['source']}:{chunk.metadata['start_index']}".encode()).hexdigest()
            for chunk in chunks
        ]
        metadatas = [
            {k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))}
            for c in chunks
        ]

        document_collection.add(
            ids=ids,
            documents=[c.page_content.strip() for c in chunks],
            metadatas=metadatas,
        )

        logger.info("'documents__%s': %d docs -> %d chunks", _dir, len(documents), len(chunks))


def cleanup_db():
    for collection in client_db.list_collections():
        client_db.delete_collection(name=collection.name)
    client_db.clear_system_cache()


if __name__ == '__main__':
    logger.info("--- START ---")

    logger.info("--- Cleanup database ---")
    # cleanup_db()

    logger.info("--- Ingest images ---")
    ingest_images()

    logger.info("--- Ingest documents ---")
    ingest_documents()

    logger.info("--- END ---")
