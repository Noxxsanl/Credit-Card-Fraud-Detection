"""Đóng gói Docker — T-58, docs/03 §6.3, docs/10 §4.1.

Không build ảnh, không cần Docker: canh những giả định mà ảnh dựa vào và dễ bị phá âm thầm.

* Phiên bản thư viện trong ``api/requirements*.txt`` phải trùng lúc xuất hiện vật. Huấn luyện lại
  bằng bản mới mà quên sửa thì container vẫn build được, chỉ chấm ra điểm khác.
* ``api/entrypoint.py`` thử lại migration khi PostgreSQL chưa nhận kết nối, và dừng hẳn khi hết lượt.
* Cấu trúc ``docker-compose.yml`` và ``.dockerignore``.
"""

import json
import re

import pytest
from sqlalchemy.exc import OperationalError

from api import entrypoint
from api.loader import SERVING_PACKAGES
from src.config import METRICS_PATH, PROJECT_ROOT

API_REQUIREMENTS = ("api/requirements.txt", "api/requirements-nodeps.txt")


def pins(*paths: str) -> dict[str, str]:
    """``{tên chuẩn hóa: phiên bản}`` của mọi dòng ``tên[extra]==phiên bản``."""
    found = {}
    for path in paths:
        for line in (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([^\s;#]+)", line)
            if match:
                found[normalize(match.group(1))] = match.group(3)
    return found


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_api_requirements_are_exact_pins():
    for path in API_REQUIREMENTS:
        for line in (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                assert "==" in line, f"{path}: '{line}' phải ghim bằng == (tái lập, NFR-07)"


def exported_packages() -> dict[str, str]:
    if not METRICS_PATH.exists():
        pytest.skip("chưa có models/metrics.json")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))["environment"]["packages"]


def test_api_pins_match_exported_artifacts():
    """Chỉ những gói API thật sự nạp (SERVING_PACKAGES) — shap có trong metrics.json nhưng ảnh không cài."""
    expected = {name: v for name, v in exported_packages().items() if name in SERVING_PACKAGES}
    assert set(expected) == set(SERVING_PACKAGES)
    pinned = pins(*API_REQUIREMENTS)
    mismatched = {name: (version, pinned.get(normalize(name)))
                  for name, version in expected.items() if pinned.get(normalize(name)) != version}
    assert not mismatched, f"api/requirements*.txt lệch phiên bản lúc xuất hiện vật (cần, đang ghim): {mismatched}"


def test_api_image_does_not_install_shap():
    """SHAP tính bằng pred_contribs của XGBoost (api/serving.py); shap kéo theo numba + llvmlite ~210 MB."""
    assert "shap" not in pins(*API_REQUIREMENTS)


def test_root_requirements_are_exact_pins():
    """Cách B của README tạo lại hiện vật từ notebook: cài bản khác là notebook 08 dừng vì số lệch."""
    for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            assert "==" in line, f"requirements.txt: '{line}' phải ghim bằng == (tái lập, NFR-07)"


def test_root_pins_match_exported_artifacts():
    pinned = pins("requirements.txt")
    mismatched = {name: (version, pinned.get(normalize(name)))
                  for name, version in exported_packages().items() if pinned.get(normalize(name)) != version}
    assert not mismatched, f"requirements.txt lệch phiên bản lúc xuất hiện vật (cần, đang ghim): {mismatched}"


def test_xgboost_is_installed_without_dependencies():
    assert "xgboost" in pins("api/requirements-nodeps.txt")
    assert "xgboost" not in pins("api/requirements.txt"), "xgboost kéo theo nvidia-nccl-cu12 nếu cài kèm phụ thuộc"


def test_api_packages_exist_in_development_environment():
    root = {normalize(re.split(r"[\[<>=~!;\s]", line.strip(), maxsplit=1)[0])
            for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")}
    extra = set(pins(*API_REQUIREMENTS)) - root
    assert not extra, f"container dùng gói không có trong requirements.txt gốc: {sorted(extra)}"


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

def refused() -> OperationalError:
    return OperationalError("SELECT 1", {}, Exception("connection refused\nchi tiết"))


def test_migrate_retries_until_database_accepts(monkeypatch):
    calls = []

    def upgrade(url):
        calls.append(url)
        if len(calls) < 3:
            raise refused()

    monkeypatch.setattr(entrypoint, "upgrade", upgrade)
    monkeypatch.setattr(entrypoint.time, "sleep", lambda s: None)
    entrypoint.migrate("postgresql+psycopg://x")
    assert len(calls) == 3


def test_migrate_gives_up_after_last_attempt(monkeypatch):
    calls = []

    def upgrade(url):
        calls.append(url)
        raise refused()

    monkeypatch.setattr(entrypoint, "upgrade", upgrade)
    monkeypatch.setattr(entrypoint.time, "sleep", lambda s: None)
    with pytest.raises(OperationalError):
        entrypoint.migrate("postgresql+psycopg://x")
    assert len(calls) == entrypoint.MIGRATE_ATTEMPTS


def test_migrate_does_not_retry_a_broken_migration(monkeypatch):
    calls = []

    def upgrade(url):
        calls.append(url)
        raise RuntimeError("lỗi trong tệp migration")

    monkeypatch.setattr(entrypoint, "upgrade", upgrade)
    with pytest.raises(RuntimeError):
        entrypoint.migrate("postgresql+psycopg://x")
    assert len(calls) == 1


# --------------------------------------------------------------------------
# Compose và ngữ cảnh build
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def compose():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_has_three_services(compose):
    assert set(compose["services"]) == {"db", "api", "web"}


def test_api_waits_for_healthy_database(compose):
    api = compose["services"]["api"]
    assert api["depends_on"]["db"]["condition"] == "service_healthy"
    assert "@db:5432/" in api["environment"]["DATABASE_URL"]


def test_database_healthcheck_uses_tcp(compose):
    # Qua socket Unix thì báo khỏe cả lúc máy chủ tạm của initdb đang chạy (docker-compose.yml)
    assert "-h 127.0.0.1" in " ".join(compose["services"]["db"]["healthcheck"]["test"])


def test_ports_are_published_on_localhost_by_default(compose):
    """API không có đăng nhập: mặc định không mở cổng nào ra mạng LAN (BIND_ADDRESS trong .env để mở)."""
    for name, service in compose["services"].items():
        for port in service.get("ports", []):
            assert port.startswith("${BIND_ADDRESS:-127.0.0.1}:"), f"{name}: cổng '{port}' mở ra mọi địa chỉ"


def test_database_password_is_not_hard_coded(compose):
    db, api = compose["services"]["db"], compose["services"]["api"]
    assert db["environment"]["POSTGRES_PASSWORD"].startswith("${POSTGRES_PASSWORD")
    assert ":${POSTGRES_PASSWORD" in api["environment"]["DATABASE_URL"]


def test_artifacts_are_mounted_read_only_not_baked(compose):
    volumes = compose["services"]["api"]["volumes"]
    assert "./models:/app/models:ro" in volumes
    assert "./data:/app/data:ro" in volumes


def test_build_context_never_contains_data_or_secrets():
    rules = [line.strip() for line in (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    assert rules[0] == "*", ".dockerignore phải là danh sách trắng"
    allowed = {rule[1:].rstrip("/") for rule in rules if rule.startswith("!")}
    assert not allowed & {"data", "models", ".env", ".venv", "notebooks", "reports"}
    assert "frontend/node_modules" in rules
