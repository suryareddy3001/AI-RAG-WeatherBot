import argparse
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from qdrant_client.models import PointStruct

from ..llm.providers import EmbeddingsProvider
from ..vectordb.qdrant_store import QdrantStore
from ..logging import get_logger

"""PDF ingestion module for RAGChain-WeatherBot.

This module provides functionality to load, chunk, embed, and store PDF documents
in a Qdrant vector database. It supports command-line ingestion and ensures compatibility
with older Qdrant servers by using integer IDs and pure Python floats.
"""

logger = get_logger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

def load_and_chunk_pdf(pdf_path: Path) -> List[Document]:
    """Load and chunk a PDF document into smaller segments.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of Document objects, each representing a chunk of the PDF.
    """
    docs = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def _to_py_floats(vec):
    """Convert a vector to a list of pure Python floats.

    Args:
        vec: Input vector, potentially containing numpy.float32 values.

    Returns:
        A list of Python float values.
    """
    return [float(v) for v in (vec.tolist() if hasattr(vec, "tolist") else vec)]

def embed_and_upsert(docs: List[Document], embeddings, vectordb: QdrantStore) -> None:
    """Embed document chunks and upload them to Qdrant.

    Args:
        docs: List of Document objects to embed and upload.
        embeddings: Embeddings provider instance.
        vectordb: QdrantStore instance for vector storage.
    """
    points: list[PointStruct] = []
    pid = 0
    for i, doc in enumerate(docs):
        vec = _to_py_floats(embeddings.embed_query(doc.page_content))

        payload = {
            "doc_id": doc.metadata.get("source", "unknown"),
            "page": int(doc.metadata.get("page", -1)) if str(doc.metadata.get("page", -1)).isdigit() else -1,
            "chunk": i,
            "text": doc.page_content,
        }

        points.append(
            PointStruct(
                id=pid,
                vector=vec,
                payload=payload,
            )
        )
        pid += 1

    vectordb.upload(points)
    logger.info("Ingested %d chunks", len(points))

def main():
    """Command-line interface for ingesting a PDF into Qdrant.

    Parses command-line arguments, loads and chunks the PDF, embeds the chunks,
    and uploads them to the Qdrant vector database.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = load_and_chunk_pdf(pdf_path)

    embeddings = EmbeddingsProvider().get()
    probe = embeddings.embed_query("probe")
    probe = probe.tolist() if hasattr(probe, "tolist") else probe
    vec_size = len(probe)

    vectordb = QdrantStore()
    vectordb.ensure_collection(vec_size, recreate_if_mismatch=True)

    embed_and_upsert(docs, embeddings, vectordb)
    logger.info("PDF ingestion complete.")

if __name__ == "__main__":
    main()