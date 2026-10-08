# tests/test_web_jobs.py —— 进度矩阵与手动补跑表单

import pytest
from starlette.testclient import TestClient

from tests.test_web_index import _settings


@pytest.fixture
def jobs_client(tmp_path, monkeypatch):
    import camdigest.pipeline.orchestrator as orch
    from camdigest import db
    from camdigest.web.app import create_app

    monkeypatch.setattr(orch, "run_day",
                        lambda date, s, until=None: None)   # 补跑不真跑
    settings = _settings(tmp_path)
    app = create_app(settings)
    seed_url = f"sqlite:///{tmp_path}/t.db"
    app.state.db_url = seed_url
    db.init_db(seed_url)
    with db.session_scope(seed_url) as s:
        s.add(db.Job(date="2026-09-29", stage="s1_index", status="done"))
        s.add(db.Job(date="2026-09-29", stage="s3_person", status="failed",
                     error="boom"))
        s.add(db.Job(date="2026-09-28", stage="s1_index", status="running"))
    return TestClient(app), app


def test_jobs_matrix(jobs_client):
    c, _ = jobs_client
    r = c.get("/jobs")
    assert r.status_code == 200
    for token in ("2026-09-29", "2026-09-28", "s1_index", "s10_clip", "boom"):
        assert token in r.text


def test_run_form_and_post(jobs_client):
    c, _ = jobs_client
    assert c.get("/run").status_code == 200
    r = c.post("/run", data={"date": "2026-09-27"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/jobs"


def test_run_busy_redirects_with_msg(jobs_client):
    c, app = jobs_client
    app.state.runs._busy.set()
    r = c.post("/run", data={"date": "2026-09-27"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/jobs?msg=busy")
    app.state.runs._busy.clear()


def test_run_rejects_bad_input(jobs_client):
    c, _ = jobs_client
    assert c.post("/run", data={"date": "not-a-date"}).status_code == 422
    assert c.post("/run", data={"date": "2026-09-27", "until": "s99"}).status_code == 422
