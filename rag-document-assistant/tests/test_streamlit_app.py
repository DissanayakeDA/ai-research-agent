"""Drive the real Streamlit page, without a browser, against a real API server with fake models.

Streamlit's AppTest runs the script and lets us type and click; the API runs in a background
thread on a free port. Together they exercise UI -> HTTP -> FastAPI -> pipeline.
"""

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from app.ingestion import ingest_pdf
from app.main import create_app
from helpers import FakeLLM, make_pdf, make_services

APP = str(Path(__file__).resolve().parent.parent / "frontend" / "streamlit_app.py")
PAGE = "Tabu search keeps a tabu list of recent moves\nso the search does not cycle back to old solutions."
ANSWER = "Tabu search keeps a list of recent moves [1]."


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def api_url(tmp_path):
    services = make_services(tmp_path / "data", FakeLLM(ANSWER))
    ingest_pdf(make_pdf(PAGE), "tabu.pdf", services.settings, services.embedder, services.store)

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(lambda: services), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        assert time.monotonic() < deadline, "API server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_asking_a_question_shows_the_answer_and_its_passages(api_url, monkeypatch):
    monkeypatch.setenv("API_URL", api_url)
    page = AppTest.from_file(APP, default_timeout=60).run()
    assert not page.exception

    page.text_input[0].input("tabu search recent moves")
    page.button[0].click().run()

    assert not page.exception
    assert ANSWER in [element.value for element in page.markdown]
    assert any("tabu.pdf" in element.value and "[1]" in element.value for element in page.markdown)  # sources list
    assert any(expander.label.startswith("[1] CITED") for expander in page.expander)


def test_missing_api_shows_how_to_start_it(monkeypatch):
    monkeypatch.setenv("API_URL", f"http://127.0.0.1:{free_port()}")  # nothing listens there

    page = AppTest.from_file(APP, default_timeout=60).run()

    assert any("Cannot reach the API" in element.value for element in page.error)
