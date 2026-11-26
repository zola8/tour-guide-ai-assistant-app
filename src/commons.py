import logging
from typing import Any
from typing import Dict

import chromadb
from PIL import Image
from chromadb import EmbeddingFunction
from chromadb import Embeddings
from chromadb.api.types import Images
from chromadb.utils.embedding_functions import register_embedding_function
from img2vec_pytorch import Img2Vec

DATA_PATH = "./data"
DB_PATH = "./db"


# ----- Logging

def configure_logging():
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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
