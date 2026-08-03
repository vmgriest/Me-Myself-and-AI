# Loads data from multiple sources (PDF, MD, Web)
#
# load_all_sources() is the single seam every source plugs into. Each
# loader degrades gracefully (returns an empty list) when its source isn't
# configured, so ingestion still works with just bio.md if that's all
# that's set up.
from langchain_core.documents import Document  # LangChain's standard text + metadata container
from pypdf import PdfReader  # reads text out of PDF files

from src import config  # for config.BIO_PATH / RESUME_PATH
from src.web import github_connector, website_scraper


def load_bio() -> list[Document]:
    text = config.BIO_PATH.read_text(encoding="utf-8")  # read the whole bio file as one string
    # wrap it as a single Document; metadata lets downstream code attribute chunks back to "bio.md"
    return [Document(page_content=text, metadata={"source": "bio.md", "source_type": "bio"})]


def load_resume() -> list[Document]:
    """Reads data/raw/resume.pdf (gitignored, personal document) if present."""
    if not config.RESUME_PATH.exists():  # optional source — skip cleanly if not provided
        return []

    reader = PdfReader(config.RESUME_PATH)  # opens the PDF for text extraction
    text = "\n".join(page.extract_text() or "" for page in reader.pages)  # join all pages' text into one string
    return [Document(page_content=text, metadata={"source": "resume.pdf", "source_type": "resume"})]


def load_all_sources() -> list[Document]:
    return [
        *load_bio(),
        *load_resume(),
        *github_connector.load_github_repos(),
        *website_scraper.load_websites(),
    ]
