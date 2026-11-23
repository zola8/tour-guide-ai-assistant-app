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
from chromadb.utils.embedding_functions import register_embedding_function
from img2vec_pytorch import Img2Vec
from langchain_community.document_loaders import DirectoryLoader

logging.basicConfig(
    level="DEBUG",
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

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
        logger.debug(f"{len(images_path)} files added to DB collection")

    return image_collection


def ingest_documents():
    for _dir in os.listdir(DATA_PATH):
        dir_path = os.path.join(DATA_PATH, _dir)

        documents = DirectoryLoader(path=dir_path, glob="*.txt").load()
        print(documents)

        # document collection
        # doc_collection = client_db.get_or_create_collection(
        #     name=f"documents_{_dir}",
        #     embedding_function=OllamaEmbeddingFunction(url="http://localhost:11434", model_name="qwen3-embedding:0.6b"),
        # )


if __name__ == '__main__':
    logger.info("--- START ---")
    # ingest_images()
    ingest_documents()
