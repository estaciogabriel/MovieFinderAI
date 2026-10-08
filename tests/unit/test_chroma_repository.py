"""
Unit tests for ChromaRepository credential validation
"""

import pytest

from src.infrastructure.chroma_repository import ChromaRepository


@pytest.fixture(autouse=True)
def reset_singleton():
    """Each test gets a fresh singleton."""
    ChromaRepository._instance = None
    yield
    ChromaRepository._instance = None


def _patch_credentials(monkeypatch, **values):
    defaults = {"CHROMA_API_KEY": "key", "CHROMA_TENANT": "tenant",
                "CHROMA_DATABASE": "db"}
    defaults.update(values)
    for name, value in defaults.items():
        monkeypatch.setattr(
            f"src.infrastructure.chroma_repository.{name}", value
        )


def test_all_credentials_missing(monkeypatch):
    _patch_credentials(
        monkeypatch,
        CHROMA_API_KEY="", CHROMA_TENANT="", CHROMA_DATABASE="",
    )
    with pytest.raises(RuntimeError) as exc:
        ChromaRepository()
    msg = str(exc.value)
    for name in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"):
        assert name in msg
    assert ".env" in msg


def test_partial_credentials_missing(monkeypatch):
    _patch_credentials(monkeypatch, CHROMA_DATABASE="")
    with pytest.raises(RuntimeError) as exc:
        ChromaRepository()
    msg = str(exc.value)
    assert "CHROMA_DATABASE" in msg
    assert "CHROMA_API_KEY" not in msg
    assert "CHROMA_TENANT" not in msg


def test_failed_init_does_not_leave_zombie(monkeypatch):
    _patch_credentials(monkeypatch, CHROMA_API_KEY="")
    with pytest.raises(RuntimeError):
        ChromaRepository()
    assert ChromaRepository._instance is None
