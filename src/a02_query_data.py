import logging

import cv2

from src.commons import client_db
from src.commons import configure_logging
from src.commons import document_embeddings
from src.commons import image_embeddings
from src.llm import create_response

configure_logging()

logger = logging.getLogger('a02_query_data')


def classify_image(client_db, query_img):
    if query_img is None:
        logger.error("No query image provided")
        return None

    collection = client_db.get_collection(
        name="images",
        embedding_function=image_embeddings,
    )

    img = cv2.imread(query_img)
    if img is None:
        logger.error("Could not read image: %s", query_img)
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    embeddings = image_embeddings([img_rgb])

    results = collection.query(query_embeddings=embeddings, n_results=1)
    logger.debug(
        "query_img=%s -> id=%s distance=%s",
        query_img, results['ids'][0][0], results['distances'][0][0],
    )

    meta = results['metadatas'][0][0] or {}
    return meta.get("category") or results['ids'][0][0].split('-')[0]


def get_similar_documents(client_db, query_question, img_category, max_distance=1.0):
    collection = client_db.get_collection(
        name=f"documents__{img_category}",
        embedding_function=document_embeddings,
    )
    results = collection.query(query_texts=[query_question], n_results=5)
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    dists = results['distances'][0]
    filtered = [(d, m) for d, m, dist in zip(docs, metas, dists) if dist < max_distance]

    logger.info("category=%s, retrieved=%d, kept=%d", img_category, len(docs), len(filtered))
    if not filtered:
        return [], []
    return [d for d, _ in filtered], [m for _, m in filtered]


if __name__ == '__main__':
    logger.info("--- START ---")

    test_list = [
        ('eth_zh.jpg',
         [
             'When is the Academic department open?',
             'Recommend me practical tips',
         ]),
        ('fraumunster.jpg', []),
        ('grossmunster.jpg', []),
        ('operahouse_zh.jpg', []),
    ]

    for test_img, test_questions in test_list:
        query_img = f"./test_data/{test_img}"
        classification = classify_image(client_db=client_db, query_img=query_img)
        logger.info(f"Image name: {test_img}, classification: {classification}")

        for question in test_questions:
            chunks_text, chunks_metadata = get_similar_documents(client_db=client_db, query_question=question,
                                                                 img_category=classification)
            answer = create_response(question, chunks_text, chunks_metadata)

            logger.info(f"\tQ: {question}")
            logger.info(f"\tA: {answer}")

    logger.info("--- END ---")
