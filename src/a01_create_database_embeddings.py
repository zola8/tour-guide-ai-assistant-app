import logging
import os
from typing import Any
from typing import Dict

import chromadb
from PIL import Image
from chromadb import EmbeddingFunction
from chromadb import Embeddings
from chromadb.api.types import Images
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from chromadb.utils.embedding_functions import register_embedding_function
from img2vec_pytorch import Img2Vec
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

os.environ["UNSTRUCTURED_DISABLE_NOTICE"] = "1"

DATA_PATH = "./data"
DB_PATH = "./db"


# ----- Embedding function

@register_embedding_function
class ImageEmbeddings(EmbeddingFunction):

    def __init__(self):
        self.model = Img2Vec()

    def __call__(self, input: Images) -> Embeddings:
        embeddings = self._get_imgs_embeddings(input)
        return embeddings

    def _get_imgs_embeddings(self, image_list: Images) -> Embeddings:
        return [self.model.get_vec(Image.fromarray(img)) for img in image_list]

    @staticmethod
    def name() -> str:
        return "img2vec"

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return ImageEmbeddings(config['model'])


# ----- Database

client_db = chromadb.PersistentClient(path=DB_PATH)


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


if __name__ == '__main__':
    logger.info("--- START ---")

    logger.info("--- Ingest images ---")
    # ingest_images()

    logger.info("--- Ingest documents ---")
    # ingest_documents()

    logger.info("--- END ---")
