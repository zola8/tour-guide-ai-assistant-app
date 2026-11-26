import logging

import cv2

from src.commons import DATA_PATH
from src.commons import ImageEmbeddings
from src.commons import client_db
from src.commons import configure_logging
from src.llm import create_response

configure_logging()

logger = logging.getLogger('a02_query_data')

image_embedding = ImageEmbeddings()


def classify_image(client_db, query_img):
    """ Query the most similar image to the query image """
    if query_img is None:
        logger.error("No query image provided")
        return None

    collection = client_db.get_collection(name='images')
    logger.debug(f"Get image collection: {collection}")

    img = cv2.imread(query_img)
    _embeddings = image_embedding([img])
    logger.debug(f"Image loaded and converted to embedding: {query_img}")

    results = collection.query(
        query_embeddings=_embeddings,
        n_results=1,
    )

    # flat = {k: v[0] if isinstance(v, list) and v else v for k, v in result.items()}
    # logger.debug(f"Result: {json.dumps(flat, indent=2)}")
    logger.debug(
        f"query_img: {query_img} ; similar image found: {results['ids'][0][0]} ; distance: {results['distances'][0][0]}")

    return results['ids'][0][0].split('-')[0]


def get_similar_documents(client_db, query_question, img_category):
    collection = client_db.get_collection(name=f"documents_{img_category}")
    results = collection.query(
        query_texts=[query_question],
        n_results=3,
    )
    return results['documents'][0], results['metadatas'][0]


if __name__ == '__main__':
    logger.info("--- START ---")

    test_list = [
        ('eth_zh.jpg',
         [
             'When is the Academic department open?',
             'Recommend me practical tips'
         ]),
        ('fraumunster.jpg', []),
        ('grossmunster.jpg', []),
        ('operahouse_zh.jpg', []),
    ]

    for test_img, test_questions in test_list:
        query_img = f"{DATA_PATH}/test/{test_img}"
        classification = classify_image(client_db=client_db, query_img=query_img)
        logger.info(f"Image name: {test_img}, classification: {classification}")

        for question in test_questions:
            chunks_text, chunks_metadata = get_similar_documents(client_db=client_db, query_question=question,
                                                                 img_category=classification)
            answer = create_response(question, chunks_text, chunks_metadata)

            logger.info(f"\tQ: {question}")
            logger.info(f"\tA: {answer}")

    logger.info("--- END ---")
