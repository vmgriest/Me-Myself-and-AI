# GitHub API client for pulling repos
#
# Uses PyGithub unauthenticated (or with GITHUB_TOKEN if set) to pull each
# public repo's README for the configured user. Unauthenticated calls are
# rate-limited (60 req/hour) but that's plenty for a handful of repos.
import json  # to write the repo manifest

from github import Github, GithubException  # PyGithub's client + its base exception type
from langchain_core.documents import Document

from src import config


def load_github_repos() -> list[Document]:
    """Fetches the README of every public, non-fork repo for
    config.GITHUB_USERNAME. Returns an empty list (rather than raising) if
    no username is configured, so ingestion still works without it.

    Also writes a complete name+description manifest of every repo found
    (see save_repo_manifest) — a "list/how many repos do you have" question
    doesn't map well to vector search (there's no ranking/completeness
    signal in a top-k similarity search over README chunks), and without
    this, such questions under-retrieved and fell back to a live web
    search that dragged in unrelated noise instead of just answering from
    what's already known."""
    if not config.GITHUB_USERNAME:  # nothing configured — skip this source cleanly
        save_repo_manifest([])  # clears out any manifest left over from a previous, differently-configured run
        return []

    client = Github(config.GITHUB_TOKEN) if config.GITHUB_TOKEN else Github()  # authenticated if a token is set
    documents: list[Document] = []
    manifest: list[dict] = []  # every non-fork repo's name + description, regardless of README presence

    try:
        user = client.get_user(config.GITHUB_USERNAME)  # the configured GitHub account
        for repo in user.get_repos():  # every public repo the account owns
            if repo.fork:  # skip forks — not original content worth indexing
                continue
            manifest.append({"name": repo.name, "description": repo.description or ""})

            try:
                readme = repo.get_readme()  # raises if the repo has no README
                text = readme.decoded_content.decode("utf-8", errors="ignore")  # README bytes -> text
            except GithubException:  # no README on this repo — skip it, not a fatal error
                continue

            documents.append(
                Document(
                    page_content=f"GitHub repo: {repo.name}\n{repo.description or ''}\n\n{text}",
                    metadata={"source": f"github:{repo.name}", "source_type": "github"},
                )
            )
    except GithubException as e:  # bad username, rate limit, etc. — degrade gracefully rather than crash ingestion
        print(f"github_connector: could not fetch repos for {config.GITHUB_USERNAME}: {e}")

    save_repo_manifest(manifest)
    return documents


def save_repo_manifest(manifest: list[dict]) -> None:
    """Overwrites the repo manifest with this ingestion run's repo list."""
    config.PARENT_STORE_DIR.mkdir(parents=True, exist_ok=True)  # make sure the folder exists
    config.GITHUB_REPOS_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
