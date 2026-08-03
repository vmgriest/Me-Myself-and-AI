# Unit tests for the ingestion pipeline
#
# No live models/network here: chunking is pure string-splitting logic,
# and the source loaders are tested through their "not configured, skip
# cleanly" paths — the paths that actually hit GitHub/Tavily/HF are covered
# by manual smoke testing (see README), not this offline suite.
import json

from langchain_core.documents import Document

from src.ingestion import chunking
from src.web import github_connector, website_scraper


def _doc(text: str, source: str = "bio.md") -> Document:
    return Document(page_content=text, metadata={"source": source, "source_type": "bio"})


def test_parent_child_chunk_children_link_back_to_saved_parents(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "PARENT_STORE_DIR", tmp_path / "parent_store")
    monkeypatch.setattr(config, "PARENT_STORE_PATH", tmp_path / "parent_store" / "parents.json")
    monkeypatch.setattr(chunking, "_parent_store_cache", None)  # clear any cache from a previous test

    long_text = "Paragraph text about the project. " * 100  # long enough to span multiple parents
    children = chunking.parent_child_chunk([_doc(long_text)])

    assert len(children) > 0
    assert config.PARENT_STORE_PATH.exists()  # save_parents() actually wrote the file

    for child in children:
        parent_id = child.metadata["parent_id"]
        parent_text = chunking.load_parent(parent_id)  # round-trips through the file just written
        assert child.page_content in parent_text  # every child's text is a substring of its own parent


def test_load_bio_reads_configured_file(tmp_path, monkeypatch):
    from src import config
    from src.ingestion import document_loader

    bio_path = tmp_path / "bio.md"
    bio_path.write_text("Test bio content.", encoding="utf-8")
    monkeypatch.setattr(config, "BIO_PATH", bio_path)

    docs = document_loader.load_bio()
    assert len(docs) == 1
    assert docs[0].page_content == "Test bio content."
    assert docs[0].metadata["source"] == "bio.md"


def test_github_connector_skips_cleanly_without_username(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "GITHUB_USERNAME", None)
    monkeypatch.setattr(config, "PARENT_STORE_DIR", tmp_path / "parent_store")
    monkeypatch.setattr(config, "GITHUB_REPOS_MANIFEST_PATH", tmp_path / "parent_store" / "github_repos.json")

    assert github_connector.load_github_repos() == []  # no network call attempted
    assert config.GITHUB_REPOS_MANIFEST_PATH.exists()  # still writes an (empty) manifest, clearing any stale one
    assert json.loads(config.GITHUB_REPOS_MANIFEST_PATH.read_text(encoding="utf-8")) == []


def test_save_repo_manifest_writes_name_and_description(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "PARENT_STORE_DIR", tmp_path / "parent_store")
    monkeypatch.setattr(config, "GITHUB_REPOS_MANIFEST_PATH", tmp_path / "parent_store" / "github_repos.json")

    github_connector.save_repo_manifest([{"name": "MyRepo", "description": "Does a thing."}])

    written = json.loads(config.GITHUB_REPOS_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert written == [{"name": "MyRepo", "description": "Does a thing."}]


def test_website_scraper_skips_cleanly_without_config(monkeypatch):
    from src import config

    monkeypatch.setattr(config, "WEBSITE_URLS", [])
    monkeypatch.setattr(config, "TAVILY_API_KEY", None)
    assert website_scraper.load_websites() == []  # no network call attempted
