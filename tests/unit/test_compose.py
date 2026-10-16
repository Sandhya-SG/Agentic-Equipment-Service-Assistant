"""The Docker Compose deployment: the two services start in order, talk to each other by name, keep their logs, take
secrets only from the environment, and every setting they pass through is documented in .env.example."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
SERVICES = COMPOSE["services"]
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def test_the_two_services_are_built_from_their_own_dockerfiles():
    assert set(SERVICES) == {"backend", "frontend"}
    assert SERVICES["backend"]["build"]["dockerfile"] == "Dockerfile.backend"
    assert SERVICES["frontend"]["build"]["dockerfile"] == "Dockerfile.frontend"
    for service in SERVICES.values():
        assert (ROOT / service["build"]["dockerfile"]).is_file() and service["build"]["context"] == "."


def test_the_frontend_waits_until_the_backend_is_healthy():
    assert SERVICES["frontend"]["depends_on"] == {"backend": {"condition": "service_healthy"}}
    check = " ".join(SERVICES["backend"]["healthcheck"]["test"])
    assert "localhost:8000/health" in check  # liveness, not /ready: the agents may still be loading
    assert SERVICES["backend"]["healthcheck"]["start_period"]  # room for the slow first start


def test_the_services_find_each_other_by_name_on_the_ports_the_images_listen_on():
    assert SERVICES["frontend"]["environment"]["BACKEND_URL"] == "http://backend:8000"
    assert "8000:8000" in SERVICES["backend"]["ports"]
    assert any(port.endswith(":5000") for port in SERVICES["frontend"]["ports"])


def test_both_services_restart_unless_stopped_and_the_backend_keeps_its_logs_on_the_host():
    assert all(service["restart"] == "unless-stopped" for service in SERVICES.values())
    assert "./logs:/app/logs" in SERVICES["backend"]["volumes"]  # the audit trail survives a container restart


def test_no_secret_is_written_into_the_compose_file():
    for name, service in SERVICES.items():
        for variable, value in service["environment"].items():
            if any(word in variable for word in SECRET_WORDS):
                assert str(value).startswith("${") and str(value).endswith(":-}"), f"{name}.{variable} = {value!r}"


def test_compose_runs_in_development_mode_by_default_which_leaves_only_monitoring_open():
    """In development the monitoring endpoint is open and the review API still needs its token (no open mode)."""
    assert SERVICES["backend"]["environment"]["ENVIRONMENT"] == "${ENVIRONMENT:-development}"


def test_every_variable_compose_reads_from_the_environment_is_documented_in_the_env_example():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", example, flags=re.MULTILINE))
    used = set(re.findall(r"\$\{([A-Z][A-Z0-9_]+)", (ROOT / "docker-compose.yml").read_text(encoding="utf-8")))
    # Compose-only spellings of a documented setting (the container address differs from the local one).
    renamed = {"DOCKER_OLLAMA_BASE_URL", "DOCKER_LANGFUSE_HOST"}
    missing = used - documented - renamed
    assert not missing, f"used by compose but not in .env.example: {sorted(missing)}"


def test_the_tokens_the_cluster_needs_are_also_passed_through_by_compose():
    environment = SERVICES["backend"]["environment"]
    for name in (
        "OPENAI_API_KEY",
        "REVIEWER_TOKEN",
        "MONITOR_TOKEN",
        "CLARIFICATION_SECRET",
        "OPENAI_SPEND_CAP_USD",
        "RATE_LIMIT_PER_MINUTE",
    ):
        assert name in environment, name


def test_the_ledger_is_on_the_mounted_logs_folder_so_the_cap_survives_a_restart():
    assert SERVICES["backend"]["environment"]["OPENAI_SPEND_LEDGER"].startswith("/app/logs/")


def test_the_optional_langfuse_stack_is_valid_yaml_and_holds_no_real_key():
    text = (ROOT / "docker-compose.langfuse.yml").read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    assert document["services"]
    assert not re.search(r"sk-lf-[A-Za-z0-9]{8,}|pk-lf-[A-Za-z0-9]{8,}", text)  # real Langfuse keys have these prefixes


@pytest.mark.parametrize("name", ["backend", "frontend"])
def test_each_service_has_no_privileged_mode_or_host_network(name):
    service = SERVICES[name]
    assert not service.get("privileged") and service.get("network_mode") != "host"
    assert not any(str(v).startswith(("/var/run/docker.sock", "/:")) for v in service.get("volumes", []))
