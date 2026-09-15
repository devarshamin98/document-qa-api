"""Phase A smoke test: the endpoint answers a question end to end, offline."""

import json

from httpx import AsyncClient


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_qa_answers_a_json_document(client: AsyncClient) -> None:
    document = json.dumps(
        {"vendor": {"name": "Acme", "cloud providers": ["AWS", "Google Cloud Platform"]}}
    ).encode()
    questions = json.dumps(["Which cloud providers does the vendor rely on?"]).encode()

    response = await client.post(
        "/api/v1/qa",
        files={
            "questions": ("questions.json", questions, "application/json"),
            "document": ("vendor.json", document, "application/json"),
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["found"] is True
    assert result["error"] is None
    assert result["citations"], "a grounded answer must carry at least one citation"

    meta = body["meta"]
    assert meta["document_type"] == "json"
    assert meta["chunks"] >= 1
    assert meta["tokens"]["embedding"] > 0
