import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from services.vector_store import VectorStore
from services.embedder import Embedder
from services.chunker import Chunker


class TestMetadataFiltering:
    """Test Phase 2: Metadata filtering functionality"""

    def test_build_filter_with_document_name(self):
        """Test filter building with document_name"""
        mock_client = Mock()
        vector_store = VectorStore(mock_client, "test_collection")

        filter_obj = vector_store._build_filter(document_name="test.pdf")

        assert filter_obj is not None
        assert len(filter_obj.must) == 1

    def test_build_filter_with_document_id(self):
        """Test filter building with document_id"""
        mock_client = Mock()
        vector_store = VectorStore(mock_client, "test_collection")

        filter_obj = vector_store._build_filter(document_id="123-456")

        assert filter_obj is not None
        assert len(filter_obj.must) == 1

    def test_build_filter_with_both(self):
        """Test filter building with both document_name and document_id"""
        mock_client = Mock()
        vector_store = VectorStore(mock_client, "test_collection")

        filter_obj = vector_store._build_filter(
            document_name="test.pdf", document_id="123-456"
        )

        assert filter_obj is not None
        assert len(filter_obj.must) == 2

    def test_build_filter_empty(self):
        """Test filter building with no filters"""
        mock_client = Mock()
        vector_store = VectorStore(mock_client, "test_collection")

        filter_obj = vector_store._build_filter()

        assert filter_obj is None


class TestHybridSearch:
    """Test Phase 2: Hybrid search functionality"""

    def test_tokenize(self):
        """Test text tokenization"""
        mock_client = Mock()
        vector_store = VectorStore(mock_client, "test_collection")

        tokens = vector_store._tokenize("Hello World Test 123")

        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_compute_bm25_score_with_match(self):
        """Test BM25 scoring with matching terms"""
        mock_client = Mock()
        vector_store = VectorStore(mock_client, "test_collection")

        score = vector_store._compute_bm25_score(
            query_terms=["hello", "world"], text="Hello world this is a test document"
        )

        assert score > 0

    def test_compute_bm25_score_no_match(self):
        """Test BM25 scoring with no matching terms"""
        mock_client = Mock()
        vector_store = VectorStore(mock_client, "test_collection")

        score = vector_store._compute_bm25_score(
            query_terms=["xyz", "abc"], text="Hello world this is a test document"
        )

        assert score == 0

    def test_hybrid_search_combines_scores(self):
        """Test hybrid search combines vector and keyword scores"""
        mock_client = Mock()
        mock_client.search.return_value = [
            Mock(
                id="1",
                score=0.9,
                payload={
                    "text": "test document about AWS",
                    "document_id": "doc1",
                    "document_name": "test.pdf",
                    "chunk_index": 0,
                },
            ),
            Mock(
                id="2",
                score=0.8,
                payload={
                    "text": "another test",
                    "document_id": "doc2",
                    "document_name": "test2.pdf",
                    "chunk_index": 0,
                },
            ),
        ]

        vector_store = VectorStore(mock_client, "test_collection")

        results = vector_store.hybrid_search(
            query_text="AWS cloud", query_vector=[0.1] * 384, top_k=5, alpha=0.7
        )

        assert len(results) > 0
        assert all("vector_score" in r for r in results)
        assert all("keyword_score" in r for r in results)


class TestEmbedderModels:
    """Test Phase 2: Multiple embedding model support"""

    @patch.dict(os.environ, {"EMBED_MODEL": "all-MiniLM-L6-v2"})
    def test_embedder_uses_env_model(self):
        """Test embedder uses model from environment variable"""
        with patch("services.embedder.SentenceTransformer") as mock_st:
            mock_st.return_value.get_sentence_embedding_dimension.return_value = 384

            embedder = Embedder()

            mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    @patch.dict(os.environ, {"EMBED_MODEL": " paraphrase-mpnet-base-v2"})
    def test_embedder_different_model(self):
        """Test embedder can use different models"""
        with patch("services.embedder.SentenceTransformer") as mock_st:
            mock_st.return_value.get_sentence_embedding_dimension.return_value = 768

            embedder = Embedder()

            mock_st.assert_called_once_with(" paraphrase-mpnet-base-v2")


class TestChunker:
    """Test text chunking functionality"""

    def test_chunk_text(self):
        """Test basic text chunking"""
        chunker = Chunker(chunk_size=50, chunk_overlap=10)

        text = "This is a long text that needs to be split into chunks. " * 10
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_small(self):
        """Test chunking small text"""
        chunker = Chunker()

        text = "Short text"
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1

    def test_chunk_documents(self):
        """Test chunking multiple documents"""
        chunker = Chunker()

        docs = ["Document one here", "Document two here"]
        chunks = chunker.chunk_documents(docs)

        assert len(chunks) >= 2


class TestVectorStore:
    """Test basic vector store functionality"""

    def test_add_vectors(self):
        """Test adding vectors to store"""
        mock_client = Mock()
        vector_store = VectorStore(mock_client, "test_collection")

        vectors = [[0.1] * 384, [0.2] * 384]
        payloads = [{"text": "doc1"}, {"text": "doc2"}]

        vector_store.add_vectors(vectors, payloads)

        mock_client.upsert.assert_called_once()

    def test_search_returns_results(self):
        """Test search returns properly formatted results"""
        mock_client = Mock()
        mock_client.search.return_value = [
            Mock(
                id="1",
                score=0.95,
                payload={
                    "text": "test text",
                    "document_id": "doc1",
                    "document_name": "test.pdf",
                    "chunk_index": 0,
                },
            )
        ]

        vector_store = VectorStore(mock_client, "test_collection")

        results = vector_store.search(query_vector=[0.1] * 384, top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["score"] == 0.95
        assert results[0]["text"] == "test text"

    def test_delete_collection(self):
        """Test collection deletion"""
        mock_client = Mock()
        vector_store = VectorStore(mock_client, "test_collection")

        vector_store.delete_collection()

        mock_client.delete_collection.assert_called_once_with(
            collection_name="test_collection"
        )
