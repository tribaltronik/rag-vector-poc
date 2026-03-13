"""
Integration tests for Phase 2 API endpoints
Run these with: pytest tests/test_api_integration.py -v
"""

import pytest
import requests
import time

BASE_URL = "http://localhost:8000"


class TestAPIIntegration:
    """Integration tests for Phase 2 API endpoints"""

    @pytest.fixture(autouse=True)
    def wait_for_api(self):
        """Wait for API to be ready"""
        max_retries = 30
        for _ in range(max_retries):
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=2)
                if r.status_code == 200:
                    return
            except:
                pass
            time.sleep(1)
        pytest.skip("API not available")

    def test_health_check(self):
        """Test health endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_search_endpoint_basic(self):
        """Test basic search without filters"""
        response = requests.get(
            f"{BASE_URL}/query/search", params={"q": "certificate", "top_k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_search_endpoint_with_document_filter(self):
        """Test search with document_name filter"""
        response = requests.get(
            f"{BASE_URL}/query/search",
            params={
                "q": "certificate",
                "top_k": 3,
                "document_name": "Well-Architected",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        if data["results"]:
            assert "Well-Architected" in data["results"][0].get("document_name", "")

    def test_search_endpoint_hybrid(self):
        """Test hybrid search"""
        response = requests.get(
            f"{BASE_URL}/query/search",
            params={"q": "AWS certificate", "top_k": 3, "use_hybrid": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_query_endpoint_basic(self):
        """Test basic query endpoint"""
        response = requests.post(
            f"{BASE_URL}/query/",
            json={"question": "What course was completed?", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_endpoint_with_rerank(self):
        """Test query with re-ranking"""
        response = requests.post(
            f"{BASE_URL}/query/",
            json={
                "question": "What course was completed?",
                "top_k": 3,
                "use_rerank": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_endpoint_with_filter(self):
        """Test query with metadata filter"""
        response = requests.post(
            f"{BASE_URL}/query/",
            json={
                "question": "What is this about?",
                "top_k": 3,
                "filter": {"document_name": "Well-Architected"},
            },
        )
        assert response.status_code == 200

    def test_query_empty_question(self):
        """Test query with empty question"""
        response = requests.post(
            f"{BASE_URL}/query/", json={"question": "", "top_k": 3}
        )
        assert response.status_code == 400

    def test_search_empty_query(self):
        """Test search with empty query"""
        response = requests.get(
            f"{BASE_URL}/query/search", params={"q": "", "top_k": 3}
        )
        assert response.status_code == 400


class TestQdrantConnection:
    """Test Qdrant connectivity"""

    def test_qdrant_health(self):
        """Test Qdrant is accessible"""
        response = requests.get("http://localhost:6333/")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data

    def test_qdrant_collection(self):
        """Test Qdrant collection exists"""
        response = requests.get("http://localhost:6333/collections/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["status"] == "green"
