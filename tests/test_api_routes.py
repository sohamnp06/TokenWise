import pytest
from fastapi.testclient import TestClient
from api.app import app, get_pipeline

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "is_indexed" in data


def test_list_documents_endpoint():
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_documents" in data


def test_upload_invalid_file_type():
    files = [("files", ("test.invalid", b"some data", "application/octet-stream"))]
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert len(data["errors"]) > 0
    assert "Unsupported file type" in data["errors"][0]


def test_query_empty():
    response = client.post("/api/query", json={"query": "  "})
    assert response.status_code == 400
