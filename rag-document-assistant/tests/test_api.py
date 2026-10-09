"""Tests for app/main.py: the real FastAPI endpoints, with a fake embedder and fake LLM."""

import pytest
from fastapi.testclient import TestClient

from app.llm_client import LLMError
from app.main import create_app
from helpers import FakeLLM, make_pdf, make_services

PAGE = "Tabu search keeps a tabu list of recent moves\nso the search does not cycle back to old solutions."


def make_client(tmp_path, llm: FakeLLM) -> TestClient:
    services = make_services(tmp_path / "data", llm)
    return TestClient(create_app(lambda: services))


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path, FakeLLM("Tabu search keeps a list of recent moves [1].")) as client:
        yield client


def upload(client: TestClient, *files: tuple[str, bytes]) -> list[dict]:
    response = client.post("/documents", files=[("files", (name, data, "application/pdf")) for name, data in files])
    assert response.status_code == 200
    return response.json()


def test_health_describes_the_service_without_the_api_key(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["chunks"] == 0
    assert "sk-secret-test" not in response.text


def test_uploaded_pdf_is_indexed_and_a_copy_kept(client, tmp_path):
    [result] = upload(client, ("tabu.pdf", make_pdf(PAGE)))

    assert result["status"] == "indexed"
    assert client.get("/health").json()["chunks"] == result["chunks"]
    assert [d["filename"] for d in client.get("/documents").json()] == ["tabu.pdf"]
    assert [p.name.endswith("_tabu.pdf") for p in (tmp_path / "data" / "uploads").iterdir()] == [True]


def test_uploading_the_same_pdf_again_is_skipped(client):
    data = make_pdf(PAGE)
    upload(client, ("tabu.pdf", data))

    [again] = upload(client, ("tabu.pdf", data))

    assert again["status"] == "skipped"
    assert len(client.get("/documents").json()) == 1


def test_each_bad_file_gets_its_own_reason(client):
    results = upload(
        client,
        ("notes.txt", b"plain text"),
        ("fake.pdf", b"this is not a pdf"),
        ("huge.pdf", b"%PDF-1.7\n" + b"0" * (1024 * 1024 + 1)),  # over the 1 MB test limit
        ("good.pdf", make_pdf(PAGE)),
    )

    assert [r["status"] for r in results] == ["failed", "failed", "failed", "indexed"]
    assert "Only PDF" in results[0]["message"]
    assert "not a PDF" in results[1]["message"]
    assert "larger than" in results[2]["message"]


def test_ask_returns_the_answer_with_its_sources(client):
    upload(client, ("tabu.pdf", make_pdf(PAGE)))

    response = client.post("/ask", json={"question": "tabu search recent moves", "top_k": 2})

    assert response.status_code == 200
    answer = response.json()
    assert answer["found"] is True
    assert answer["cited"] == [1]
    assert answer["sources"][0]["filename"] == "tabu.pdf"
    assert answer["sources"][0]["page_start"] == 1


def test_ask_with_an_empty_index_does_not_call_the_llm(tmp_path):
    llm = FakeLLM()
    with make_client(tmp_path, llm) as client:
        answer = client.post("/ask", json={"question": "anything"}).json()

    assert answer["found"] is False
    assert llm.calls == []


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "x" * 2001}, {"question": "ok", "top_k": 0}, {"top_k": 3}])
def test_invalid_questions_are_rejected(client, body):
    assert client.post("/ask", json=body).status_code == 422


def test_llm_failure_becomes_a_502_with_a_readable_message(tmp_path):
    with make_client(tmp_path, FakeLLM(error=LLMError("Groq is down"))) as client:
        upload(client, ("tabu.pdf", make_pdf(PAGE)))
        response = client.post("/ask", json={"question": "tabu search"})

    assert response.status_code == 502
    assert "Groq is down" in response.json()["detail"]
