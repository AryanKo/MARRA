import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.models.schema import DocumentChunk
from src.utils.observability import trace_span as span

class VectorStore:
    def __init__(self, host: str = None, port: int = None, collection_name: str = "marra_multimodal_768"):
        import os
        if host is None:
            host = os.environ.get("QDRANT_HOST", "localhost")
        if port is None:
            port = int(os.environ.get("QDRANT_PORT", "6333"))
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self, vector_size: int = 768):
        collections_response = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections_response.collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=vector_size,
                    distance=rest.Distance.COSINE
                )
            )

    def clear_collection(self, vector_size: int = 768):
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass
        self._ensure_collection_exists(vector_size=vector_size)

    def upsert_chunks(self, chunks: list[DocumentChunk]):
        points = []
        for chunk in chunks:
            point_id = str(uuid.uuid4())
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=chunk.dense_vector,
                    payload={
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        **chunk.metadata
                    }
                )
            )
        
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    @span(name="qdrant_dense_search")
    def dense_search(self, query_vector: list[float], k: int = 3, media_filter: str = "all"):
        from qdrant_client import models
        qdrant_filter = None
        if media_filter and media_filter != "all":
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.media_type",
                        match=models.MatchValue(value=media_filter)
                    )
                ]
            )
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=k
        ).points
        return search_result
