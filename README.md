# Agentic Equipment Service Assistant (AEM)

An AI assistant that helps technicians troubleshoot equipment. A Flask frontend sends questions to a FastAPI backend, which runs a LangGraph workflow of agents (request, planning, agentic RAG, diagnostic, safety). The agents answer only from the AEM equipment manuals and call OpenAI.

```
Browser -> frontend (Flask, :5000) -> backend (FastAPI, :8000) -> injection guard -> PII redaction -> LangGraph agents -> ChromaDB manual index + OpenAI -> safety check
```

## Prerequisites (everyone)

1. **Python 3.11 or newer**
2. **An OpenAI API key**, set as `OPENAI_API_KEY` in `.env`. The agents call OpenAI, so questions that pass the input guard (with emails, card numbers and secrets redacted) are sent to it.
3. **The manual index.** The sponsor PDFs live in `aem_documents/` (never committed). Build the ChromaDB index once, and again whenever the PDFs change:
   ```
   # PowerShell:  $env:PYTHONPATH = ".;src"      macOS / Linux:  export PYTHONPATH=.:src
   python scripts/ingest.py
   ```
   This writes `chroma_store/` (gitignored) and downloads the embedding model on first run.
4. **Git**. Docker is only needed for option B.

Ollama is no longer needed. The old Ollama client is still in `app/services/llm_client.py` but the chat path does not use it.

First-time setup for all options:

```
git clone https://github.com/Sandhya-SG/Agentic-Equipment-Service-Assistant.git
cd Agentic-Equipment-Service-Assistant
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

`.env` is for your own settings and is never committed. When you add a new environment variable in code, add it to `.env.example` in the same PR.

## Option A: run locally with Python (no Docker)

Best for day-to-day development.

```
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate

pip install -r requirements.txt
```

Set the import path so the `app` and `asa` packages are found:

```
# Windows PowerShell:  $env:PYTHONPATH = ".;src"
# macOS / Linux:       export PYTHONPATH=.:src
```

Start the backend in one terminal:

```
uvicorn app.main:app --reload --port 8000
```

Start the frontend in a second terminal (activate the venv and set `PYTHONPATH` there too):

```
flask --app frontend.app run --port 5000
```

Open http://localhost:5000. The backend health check is http://localhost:8000/health.

## Option B: run with Docker Compose

Best for checking that the containers work. Compose builds the images from the Dockerfiles on your machine, so nothing needs to be downloaded from a registry.

```
docker compose up --build
```

Open http://localhost:5000. Stop with `Ctrl+C`, then `docker compose down`.

The image builds its own manual index from `aem_documents/` and includes the embedding model, so it needs no host folders. Rebuild the image whenever the manuals change. Compose mounts `./logs` (the audit trail) and passes `OPENAI_API_KEY` from `.env`.

## Option C: run the published images

CI publishes both images to GitHub Container Registry when code is merged to `main`. This gives everyone the same build, for demos and deployment.

```
docker pull ghcr.io/sandhya-sg/asa-backend:latest
docker pull ghcr.io/sandhya-sg/asa-frontend:latest
```

Every image is also tagged `sha-<first 7 characters of the commit>`. **To roll back, redeploy the earlier `sha-` tag.** If the pull is denied, run `docker login ghcr.io` with a GitHub token that has `read:packages`, or ask the maintainer to make the packages visible to the team.

## Rollback

If a release misbehaves (errors, failing health check, unsafe or wrong answers), go back to the last good version. Rollback means running an earlier image, not editing code on the server.

1. **Find the last good version.** Open the repo's Actions tab, find the last green run on `main` before the problem, and note its commit SHA. The image tag is `sha-` plus the first 7 characters. The Packages page lists the tags that exist.
2. **Run that version.** In Docker Compose, set the image tags and restart:
   ```
   docker pull ghcr.io/sandhya-sg/asa-backend:sha-<good-sha>
   docker pull ghcr.io/sandhya-sg/asa-frontend:sha-<good-sha>
   ```
   then point the `backend` and `frontend` services at those tags (`image:` instead of `build:`) and run `docker compose up -d`.
3. **Verify.** Check `http://localhost:8000/health` returns `ok`, send one test question in the UI, and look at the backend logs for errors (`docker compose logs backend`).
4. **Fix forward in git.** Revert the bad change with `git revert <commit>` and open a PR. Do not force-push `main`. Merging the revert publishes a new image with a new `sha-` tag.

Never roll back to `latest`, because it always points at the newest build. Use the `sha-` tags, which never change.

## Run the tests

```
pip install -r requirements.txt
pytest tests
```

## CI and images

On every pull request, CI runs lint, unit and security tests, a dependency scan, then builds both images and scans them with Trivy. The build fails on fixable CRITICAL or HIGH vulnerabilities. On merge to `main` it also pushes the images.

## Rules for contributors

- **Backend dependencies:** the backend image installs only `requirements-backend.txt`, and the frontend image only `requirements-frontend.txt`. If your code imports a new package at runtime, add it to the right file (as well as `requirements.txt`), or the container will fail at startup even though it works on your machine.
- **No secrets in git.** Put real keys in `.env`, which is ignored. `.env.example` holds placeholders only.
- Run `ruff check` and `ruff format` before pushing. CI checks the files you changed.
