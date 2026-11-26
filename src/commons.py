import logging
from typing import Any
from typing import Dict

import chromadb
import numpy as np
from PIL import Image
from chromadb import EmbeddingFunction
from chromadb import Embeddings
from chromadb.api.types import Images
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from chromadb.utils.embedding_functions import register_embedding_function
from img2vec_pytorch import Img2Vec

DATA_PATH = "./data"
DB_PATH = "./db"
EMBEDDING_MODEL = "qwen3-embedding:4b"


# ----- Logging

def configure_logging():
    logging.basicConfig(
        level="DEBUG",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for name in ("httpcore", "httpx", "urllib3", "chromadb", "chromadb.telemetry", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ----- Embedding function


@register_embedding_function
class ImageEmbeddings(EmbeddingFunction):

    def __init__(self):
        self.model = Img2Vec()

    def __call__(self, input: Images) -> Embeddings:
        embeddings = self._get_imgs_embeddings(input)
        return embeddings

    def _get_imgs_embeddings(self, image_list: Images) -> Embeddings:
        out = []
        for img in image_list:
            vec = self.model.get_vec(Image.fromarray(img)).tolist()
            arr = np.asarray(vec, dtype=np.float32)
            arr = arr / (np.linalg.norm(arr) + 1e-8)  # L2 normalize
            out.append(arr.tolist())
        return out

    @staticmethod
    def name() -> str:
        return "img2vec"

    def get_config(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return ImageEmbeddings()


document_embeddings = OllamaEmbeddingFunction(url="http://localhost:11434", model_name=EMBEDDING_MODEL)
image_embeddings = ImageEmbeddings()

# ----- Database

client_db = chromadb.PersistentClient(path=DB_PATH)
