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

Best for checking that the containers work. Compose builds the images on your machine from the Dockerfiles, so no private registry or login is needed. The build does download the Python base image, the Python packages and the embedding model from the internet, so the **first build takes several minutes** (roughly 3 to 5 minutes on a fast connection, longer on a slow one) and the backend image is about 2.9 GB. Later builds take seconds if the dependencies did not change.

You need Docker Desktop running and an `OPENAI_API_KEY` in your `.env` (`cp .env.example .env`, then set the key).

```
docker compose up --build
```

Open http://localhost:5000. Stop with `Ctrl+C`, then `docker compose down`. To run it in the background, use `docker compose up --build -d` and look at the logs with `docker compose logs -f backend`.

The image builds its own manual index from `aem_documents/` and includes the embedding model, so it needs no host folders. Rebuild the image whenever the manuals change. Compose mounts `./logs` (the audit trail) and passes `OPENAI_API_KEY` from `.env`. The backend uses about 0.5 GB of memory when idle, so a Docker Desktop memory limit of 4 GB or more is comfortable.

**Troubleshooting**
- **`docker : The term 'docker' is not recognized` (Windows):** the Docker folder is not on your PATH. Start Docker Desktop first, then use one of these fixes.

  *Quick fix (this terminal only):*

  ```powershell
  $env:Path = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;" + $env:Path
  docker --version
  ```

  Then run `docker compose up --build` in the same window. You need to repeat this line in each new terminal.

  *Permanent fix (all future terminals):*

  ```powershell
  $p = [Environment]::GetEnvironmentVariable("Path", "User")
  [Environment]::SetEnvironmentVariable("Path", "$p;$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin", "User")
  ```

  Then close VS Code completely and reopen it. A new window of the same VS Code is not enough, because it only reads PATH when it starts. After that, `docker --version` should work anywhere.

  If it is still not found, check that the folder exists: `Test-Path "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe"`. If it says `False`, Docker Desktop may be installed elsewhere (for example `C:\Program Files\Docker\Docker\resources\bin`): use that folder in the commands above.
- **Frontend log shows `Control server error: [Errno 13] Permission denied: '/home/app'`:** this is harmless (the frontend still works) and comes from an older frontend image. Gunicorn 26 tries to open a control socket in the user's home folder, and the non-root container user has none. The current `Dockerfile.frontend` switches it off with `--no-control-socket`. Pull the latest code and rebuild with `docker compose up --build` (add `--no-cache` if the old image is reused). The log should then show only `INFO` lines.
- **Cannot connect to the Docker daemon:** start Docker Desktop and wait until it says it is running.
- **Port 8000 or 5000 already in use:** stop the other program (for example a local `uvicorn` or `flask`), or run `docker compose down`.
- **Every answer says "temporarily unavailable":** check that `OPENAI_API_KEY` is set in `.env`, then `docker compose logs backend`.
- **Linux only, audit log not written:** the container runs as user 10001; make sure `./logs` is writable (`mkdir -p logs && chmod 777 logs`) before the first run.

## Option C: run the published images

CI publishes both images to GitHub Container Registry when code is merged to `main`. This gives everyone the same build, for demos and deployment.

```
docker pull ghcr.io/sandhya-sg/asa-backend:latest
docker pull ghcr.io/sandhya-sg/asa-frontend:latest
```

Every image is also tagged `sha-<first 7 characters of the commit>`. **To roll back, redeploy the earlier `sha-` tag.** If the pull is denied, run `docker login ghcr.io` with a GitHub token that has `read:packages`, or ask the maintainer to make the packages visible to the team.

## Option D: run on Kubernetes (local cluster)

Best for showing scaling and self-healing. The manifests are in `k8s/` and run the backend and frontend as two replicas each on a local cluster (Docker Desktop with the kind provisioner, or minikube). Build the images, create the secret from your `.env` with `python scripts/k8s_secret.py`, then `kubectl apply -k k8s`. See `k8s/README.md` for the full steps, how to scale, and the design notes (probes, one audit file per pod, non-root containers).

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

## Tracing with Langfuse (optional)

Langfuse shows every request as a trace: the agent steps, each OpenAI call with its model, latency and token counts, and the outcome (status, escalated, hazard count, sources). It is optional: tracing is off unless both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are in `.env`, and the app behaves the same without it.

**Everyone uses their own local Langfuse and their own keys.** Do not share keys or a server: keys belong to a project, the secret key is shown only once, and traces from different people would mix.

1. Generate your own key pair and paste both lines into `.env` (keep `LANGFUSE_HOST=http://localhost:3000`):
   ```
   python -c "import uuid; print('LANGFUSE_PUBLIC_KEY=pk-lf-' + str(uuid.uuid4())); print('LANGFUSE_SECRET_KEY=sk-lf-' + str(uuid.uuid4()))"
   ```
2. Start your own server (first start pulls several images and needs about 2 to 3 GB of RAM):
   ```
   docker compose -f docker-compose.langfuse.yml up -d      # UI at http://localhost:3000
   docker compose -f docker-compose.langfuse.yml down       # stop (add -v to delete all trace data)
   ```
   On its first start it creates a project that uses exactly the keys in your `.env`. Sign in at http://localhost:3000 with `admin@aem.local` and `change-me-locally` (override with `LANGFUSE_ADMIN_EMAIL` and `LANGFUSE_ADMIN_PASSWORD`). The keys are only read when the database is first created: if you change them later, run `down -v` and start again.
3. Start the app as usual and ask a question. The trace appears within seconds, tagged `run:<run_id>`, the same ID as the audit log and the API response.

**What is traced:** the node-by-node LangGraph run, every OpenAI call with its token counts and latency, and scores for the outcome. The tracing code is `src/asa/components/tracing.py`; it is wired in at the API layer, so the agents themselves are not changed.

**Privacy:** by default traces contain no question, answer or manual text, only structure, timing, tokens and scores. Set `LANGFUSE_CAPTURE_CONTENT=true` to also keep text; it is PII-redacted and truncated first. Tests never send traces, even if real keys are in `.env`.

**Security:** the passwords in `docker-compose.langfuse.yml` are local defaults only, and every port is bound to `127.0.0.1`. Run it only on your own machine and never expose its ports. Never commit `.env`.

## Run the tests

```
pip install -r requirements.txt
pytest tests
```

### Live OpenAI tests (on demand)

`pytest tests` never calls OpenAI. Two live tests run real requests through the agents and cost a few cents, so they are skipped unless you ask:

```
pytest tests/live --run-openai              # or: RUN_OPENAI_TESTS=1 pytest tests/live
```

They need `OPENAI_API_KEY` in `.env` and the index built with `python scripts/ingest.py`; without them they skip themselves.

## CI and images

On every pull request, CI runs lint, unit and security tests, a dependency scan, then builds both images and scans them with Trivy. The build fails on fixable CRITICAL or HIGH vulnerabilities. On merge to `main` it also pushes the images.

## Rules for contributors

- **Backend dependencies:** the backend image installs only `requirements-backend.txt`, and the frontend image only `requirements-frontend.txt`. If your code imports a new package at runtime, add it to the right file (as well as `requirements.txt`), or the container will fail at startup even though it works on your machine.
- **No secrets in git.** Put real keys in `.env`, which is ignored. `.env.example` holds placeholders only.
- Run `ruff check` and `ruff format` before pushing. CI checks the files you changed.
