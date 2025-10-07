from __future__ import annotations

from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

"""Qdrant vector database wrapper for RAGChain-WeatherBot.

This module provides a QdrantStore class to manage vector collections, including creation,
upserts, uploads, and similarity searches. It ensures compatibility with older Qdrant servers
and handles vector size validation.
"""

COLLECTION_NAME = "ai_rag_weather_docs"

class QdrantStore:
    """Wrapper for Qdrant vector database operations.

    Manages vector collections with support for creation, upserting, uploading, and searching.
    Ensures backwards compatibility with older Qdrant servers (e.g., 1.7.x).
    """
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        check_compatibility: bool = False,
    ):
        """Initialize the Qdrant client.

        Args:
            url: Qdrant server URL (defaults to 'http://127.0.0.1:6333').
            api_key: Optional Qdrant API key.
            prefer_grpc: Whether to prefer gRPC over HTTP (default: False).
            check_compatibility: Whether to check server compatibility (default: False).
        """
        self.client = QdrantClient(
            url or "http://127.0.0.1:6333",
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            check_compatibility=check_compatibility,
        )

    def _extract_vector_size(self, vectors_cfg) -> Optional[int]:
        """Extract vector size from the collection's vector configuration.

        Args:
            vectors_cfg: Vector configuration from the Qdrant collection.

        Returns:
            The vector size if found, otherwise None.
        """
        if isinstance(vectors_cfg, VectorParams):
            return vectors_cfg.size
        if isinstance(vectors_cfg, dict):
            key = next(iter(vectors_cfg))
            vp = vectors_cfg[key]
            if isinstance(vp, VectorParams):
                return vp.size
        return None

    def _get_vectors_config(self):
        """Retrieve the vector configuration for the collection.

        Returns:
            The vector configuration object or None if not found.
        """
        info = self.client.get_collection(COLLECTION_NAME)
        try:
            return info.config.params.vectors
        except Exception:
            return getattr(info, "vectors", None)

    def _get_size_if_exists(self) -> Optional[int]:
        """Get the vector size of an existing collection.

        Returns:
            The vector size if the collection exists, otherwise None.
        """
        try:
            vectors_cfg = self._get_vectors_config()
        except UnexpectedResponse as e:
            if getattr(e, "status_code", None) == 404:
                return None
            raise
        except Exception:
            return None
        if vectors_cfg is None:
            return None
        return self._extract_vector_size(vectors_cfg)

    def ensure_collection(
        self,
        vector_size: int,
        *,
        recreate_if_mismatch: bool = True,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Ensure a collection exists with the specified vector size.

        Creates a new collection if it doesn't exist or recreates it if the vector size
        mismatches and recreate_if_mismatch is True.

        Args:
            vector_size: The size of the vectors for the collection.
            recreate_if_mismatch: Whether to recreate the collection if vector sizes mismatch (default: True).
            distance: The distance metric for the collection (default: COSINE).
        """
        current_size = self._get_size_if_exists()

        if current_size is None:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            return

        if current_size != vector_size:
            if not recreate_if_mismatch:
                raise RuntimeError(
                    f"Collection vector size mismatch: existing={current_size}, new={vector_size}"
                )
            self.client.delete_collection(COLLECTION_NAME)
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )

    def upsert(self, points: List[PointStruct]):
        """Upsert a list of points into the Qdrant collection.

        Args:
            points: List of PointStruct objects to upsert.

        Returns:
            The result of the upsert operation.
        """
        return self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def upload(self, points: List[PointStruct], wait: bool = True):
        """Upload a list of points to the Qdrant collection.

        Args:
            points: List of PointStruct objects to upload.
            wait: Whether to wait for the upload to complete (default: True).

        Returns:
            The result of the upload operation.
        """
        return self.client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=wait,
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform a vector similarity search in the Qdrant collection.

        Args:
            query_vector: The query vector for similarity search.
            top_k: Number of top results to return (default: 5).
            score_threshold: Minimum score threshold for results (default: None).

        Returns:
            A list of dictionaries containing search results with id, score, payload, and text.
        """
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
                "text": hit.payload.get("text"),
            }
            for hit in results
        ]
