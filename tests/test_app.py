"""
Testes unitários do targeting-service.

Estratégia: mockamos psycopg2 e requests para isolar completamente
a lógica de negócio sem necessidade de banco de dados ou auth-service.
"""
import json
import pytest
from unittest.mock import patch, MagicMock


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake:fake@localhost/fake")
    monkeypatch.setenv("AUTH_SERVICE_URL", "http://fake-auth")


@pytest.fixture
def app_client(mock_env):
    with patch("psycopg2.pool.SimpleConnectionPool") as mock_pool_cls, \
         patch("requests.get") as mock_requests_get:

        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool

        import sys
        if "app" in sys.modules:
            del sys.modules["app"]

        import app as flask_app_module
        flask_app_module.pool = mock_pool

        flask_app_module.app.config["TESTING"] = True
        client = flask_app_module.app.test_client()

        yield client, mock_pool, mock_requests_get, flask_app_module


# ── Helpers ─────────────────────────────────────────────────────────────────

def auth_ok(mock_requests_get):
    resp = MagicMock()
    resp.status_code = 200
    mock_requests_get.return_value = resp


def auth_fail(mock_requests_get):
    resp = MagicMock()
    resp.status_code = 401
    mock_requests_get.return_value = resp


SAMPLE_RULE = {"type": "PERCENTAGE", "value": 50}


# ── /health ─────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, app_client):
        client, *_ = app_client
        res = client.get("/health")
        assert res.status_code == 200
        assert res.get_json() == {"status": "ok"}


# ── Middleware de autenticação ───────────────────────────────────────────────

class TestAuthMiddleware:
    def test_missing_auth_header_returns_401(self, app_client):
        client, *_ = app_client
        res = client.get("/rules/some-flag")
        assert res.status_code == 401

    def test_invalid_key_returns_401(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_fail(mock_requests_get)
        res = client.get(
            "/rules/some-flag",
            headers={"Authorization": "Bearer bad_key"}
        )
        assert res.status_code == 401

    def test_auth_service_timeout_returns_504(self, app_client):
        import requests as req_lib
        client, mock_pool, mock_requests_get, _ = app_client
        mock_requests_get.side_effect = req_lib.exceptions.Timeout
        res = client.get(
            "/rules/some-flag",
            headers={"Authorization": "Bearer key"}
        )
        assert res.status_code == 504

    def test_auth_service_unavailable_returns_503(self, app_client):
        import requests as req_lib
        client, mock_pool, mock_requests_get, _ = app_client
        mock_requests_get.side_effect = req_lib.exceptions.ConnectionError
        res = client.get(
            "/rules/some-flag",
            headers={"Authorization": "Bearer key"}
        )
        assert res.status_code == 503


# ── POST /rules ──────────────────────────────────────────────────────────────

class TestCreateRule:
    def test_create_rule_success(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.return_value = {
            "id": 1, "flag_name": "my-flag",
            "is_enabled": True, "rules": SAMPLE_RULE
        }

        res = client.post(
            "/rules",
            data=json.dumps({"flag_name": "my-flag", "rules": SAMPLE_RULE}),
            content_type="application/json",
            headers={"Authorization": "Bearer valid_key"},
        )
        assert res.status_code == 201
        assert res.get_json()["flag_name"] == "my-flag"

    def test_create_rule_missing_fields_returns_400(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        res = client.post(
            "/rules",
            data=json.dumps({"flag_name": "only-flag-name"}),
            content_type="application/json",
            headers={"Authorization": "Bearer valid_key"},
        )
        assert res.status_code == 400

    def test_create_rule_duplicate_returns_409(self, app_client):
        import psycopg2
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.execute.side_effect = psycopg2.IntegrityError

        res = client.post(
            "/rules",
            data=json.dumps({"flag_name": "dup", "rules": SAMPLE_RULE}),
            content_type="application/json",
            headers={"Authorization": "Bearer valid_key"},
        )
        assert res.status_code == 409

    def test_create_rule_db_error_returns_500(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.execute.side_effect = Exception("db error")

        res = client.post(
            "/rules",
            data=json.dumps({"flag_name": "my-flag", "rules": SAMPLE_RULE}),
            content_type="application/json",
            headers={"Authorization": "Bearer valid_key"},
        )
        assert res.status_code == 500


# ── GET /rules/<flag_name> ───────────────────────────────────────────────────

class TestGetRule:
    def test_get_existing_rule_returns_200(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.return_value = {
            "id": 1, "flag_name": "my-flag",
            "is_enabled": True, "rules": SAMPLE_RULE
        }

        res = client.get(
            "/rules/my-flag",
            headers={"Authorization": "Bearer valid_key"}
        )
        assert res.status_code == 200
        assert res.get_json()["flag_name"] == "my-flag"

    def test_get_nonexistent_rule_returns_404(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.return_value = None

        res = client.get(
            "/rules/ghost",
            headers={"Authorization": "Bearer valid_key"}
        )
        assert res.status_code == 404


# ── PUT /rules/<flag_name> ───────────────────────────────────────────────────

class TestUpdateRule:
    def test_update_rule_success(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.rowcount = 1
        mock_cur.fetchone.return_value = {
            "id": 1, "flag_name": "my-flag",
            "is_enabled": False, "rules": SAMPLE_RULE
        }

        res = client.put(
            "/rules/my-flag",
            data=json.dumps({"is_enabled": False}),
            content_type="application/json",
            headers={"Authorization": "Bearer valid_key"},
        )
        assert res.status_code == 200

    def test_update_rule_no_body_returns_400(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        res = client.put(
            "/rules/my-flag",
            content_type="application/json",
            headers={"Authorization": "Bearer valid_key"},
        )
        assert res.status_code == 400

    def test_update_rule_no_valid_fields_returns_400(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        res = client.put(
            "/rules/my-flag",
            data=json.dumps({"bogus": "field"}),
            content_type="application/json",
            headers={"Authorization": "Bearer valid_key"},
        )
        assert res.status_code == 400

    def test_update_nonexistent_rule_returns_404(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.rowcount = 0

        res = client.put(
            "/rules/ghost",
            data=json.dumps({"is_enabled": True}),
            content_type="application/json",
            headers={"Authorization": "Bearer valid_key"},
        )
        assert res.status_code == 404


# ── DELETE /rules/<flag_name> ────────────────────────────────────────────────

class TestDeleteRule:
    def test_delete_existing_rule_returns_204(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.rowcount = 1

        res = client.delete(
            "/rules/my-flag",
            headers={"Authorization": "Bearer valid_key"}
        )
        assert res.status_code == 204

    def test_delete_nonexistent_rule_returns_404(self, app_client):
        client, mock_pool, mock_requests_get, _ = app_client
        auth_ok(mock_requests_get)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.rowcount = 0

        res = client.delete(
            "/rules/ghost",
            headers={"Authorization": "Bearer valid_key"}
        )
        assert res.status_code == 404
