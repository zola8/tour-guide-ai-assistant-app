import logging
import os

from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.commons import DATA_PATH
from src.commons import ImageEmbeddings
from src.commons import client_db
from src.commons import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

os.environ["UNSTRUCTURED_DISABLE_NOTICE"] = "1"


def ingest_images():
    extensions = (".jpg", ".png", ".jpeg", ".gif")

    image_collection = client_db.get_or_create_collection(
        name="images",
        embedding_function=ImageEmbeddings(),
        data_loader=ImageLoader(),
    )

    for _dir in os.listdir(DATA_PATH):
        dir_path = os.path.join(DATA_PATH, _dir)
        images_path = [filename for filename in os.listdir(dir_path)
                       if filename.lower().endswith(extensions)]

        logger.debug(f"Processing / dir path: {dir_path}, images: {images_path}")

        image_collection.add(
            ids=[f"{_dir}-{img_name}" for img_name in images_path],
            uris=[os.path.join(dir_path, img_path) for img_path in images_path],
        )
        logger.info(f"{len(images_path)} images added to 'images' collection from {_dir}")

    return image_collection


def ingest_documents():
    for _dir in os.listdir(DATA_PATH):
        dir_path = os.path.join(DATA_PATH, _dir)

        document_collection = client_db.get_or_create_collection(
            name=f"documents_{_dir}",
            embedding_function=OllamaEmbeddingFunction(url="http://localhost:11434", model_name="qwen3-embedding:0.6b")
        )

        documents = DirectoryLoader(path=dir_path, glob="*.txt").load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=lambda x: len(x),
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)

        document_collection.add(
            ids=[str(j) for j in range(len(chunks))],
            documents=[chunk.page_content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

        logger.info(f"Processed documents in 'documents_{_dir}': {len(documents)}, chunks: {len(chunks)}")


def cleanup_db():
    for collection in client_db.list_collections():
        client_db.delete_collection(name=collection.name)
    client_db.clear_system_cache()


if __name__ == '__main__':
    logger.info("--- START ---")

    logger.info("--- Cleanup database ---")
    cleanup_db()

    logger.info("--- Ingest images ---")
    ingest_images()

    logger.info("--- Ingest documents ---")
    ingest_documents()

    logger.info("--- END ---")
