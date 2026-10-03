# Agentic Equipment Service Assistant (AEM)

An AI assistant that helps technicians troubleshoot equipment. A Flask frontend sends questions to a FastAPI backend, which calls a local Ollama model.

```
Browser -> frontend (Flask, :5000) -> backend (FastAPI, :8000) -> Ollama (:11434)
```

## Prerequisites (everyone)

1. **Python 3.11 or newer**
2. **Ollama**, installed from https://ollama.com, with the model pulled:
   ```
   ollama pull llama3.1
   ```
   Ollama must be running (the desktop app or `ollama serve`). Check with `ollama list`.
3. **Git**. Docker is only needed for option B.

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

The backend container reaches Ollama on your computer through `host.docker.internal`, so Ollama must be running on the host. To point at a different Ollama, set `DOCKER_OLLAMA_BASE_URL` in `.env`.

## Option C: run the published images

CI publishes both images to GitHub Container Registry when code is merged to `main`. This gives everyone the same build, for demos and deployment.

```
docker pull ghcr.io/sandhya-sg/asa-backend:latest
docker pull ghcr.io/sandhya-sg/asa-frontend:latest
```

Every image is also tagged `sha-<first 7 characters of the commit>`. **To roll back, redeploy the earlier `sha-` tag.** If the pull is denied, run `docker login ghcr.io` with a GitHub token that has `read:packages`, or ask the maintainer to make the packages visible to the team.

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
