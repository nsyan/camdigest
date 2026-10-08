# tests/test_cli.py —— argparse 解析与子命令分派（薄封装，mock 内层函数）
import pytest

from camdigest import cli
from camdigest.config import Settings


def _settings():
    return Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/a"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}})


def test_run_passes_date_and_config(monkeypatch, tmp_path):
    seen = {}

    def fake_run_day(date, settings, until=None):
        seen["date"] = date
        seen["until"] = until
        return {"s1_index": 0}

    monkeypatch.setattr(cli, "_run_day_for_cli", fake_run_day)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("""
schedule: { daily_at: "03:00", timezone: Asia/Shanghai }
cameras:
  - { id: gate, name: 大门, device: gate, lens: single, dir: /a }
models:
  recognition: { protocol: openai, base_url: "http://x/v1", model: m1, api_key: k1 }
  report: { protocol: openai, base_url: "http://x/v1", model: m2, api_key: k2 }
storage: { data_dir: /data }
""", encoding="utf-8")
    rc = cli.main(["run", "--date", "2026-09-29", "--config", str(cfg), "--until", "s6_recognize"])
    assert rc == 0 and seen["date"] == "2026-09-29" and seen["until"] == "s6_recognize"


def test_run_without_config_uses_env_or_default(monkeypatch):
    seen = {}

    def fake_load(path):
        seen["path"] = str(path)
        return _settings()

    def fake_run_day(date, settings, until=None):
        seen["ran"] = True
        return {}

    monkeypatch.setattr(cli, "_load_settings", fake_load)
    monkeypatch.setattr(cli, "_run_day_for_cli", fake_run_day)
    monkeypatch.setenv("CAMDIGEST_CONFIG", "/env/path.yaml")
    cli.main(["run", "--date", "2026-09-29"])
    assert seen["path"] == "/env/path.yaml" and seen.get("ran")


def test_unknown_subcommand_exit_2():
    with pytest.raises(SystemExit) as e:
        cli.main(["nonsense"])
    assert e.value.code == 2


def test_backfill_iterates_days(monkeypatch, tmp_path):
    days = []

    def fake_run_day(date, settings, until=None):
        days.append(date)
        return {}

    s = _settings()
    s.storage.data_dir = tmp_path
    monkeypatch.setattr(cli, "_run_day_for_cli", fake_run_day)
    monkeypatch.setattr(cli, "_load_settings", lambda path: s)
    rc = cli.main(["backfill", "--from", "2026-09-01", "--to", "2026-09-03"])
    assert rc == 0 and days == ["2026-09-01", "2026-09-02", "2026-09-03"]


def test_backfill_force_reruns_reported_days(monkeypatch, tmp_path):
    from camdigest import db

    days = []
    monkeypatch.setattr(cli, "_run_day_for_cli",
                        lambda date, settings, until=None: days.append(date) or {})
    s = _settings()
    s.storage.data_dir = tmp_path
    monkeypatch.setattr(cli, "_load_settings", lambda path: s)
    url = f"sqlite:///{tmp_path}/camdigest.db"   # CLI 读 data_dir/camdigest.db
    db.init_db(url)
    with db.session_scope(url) as sess:
        sess.merge(db.Report(date="2026-09-02", md_path="/x.md"))
    cli.main(["backfill", "--from", "2026-09-01", "--to", "2026-09-03"])
    assert days == ["2026-09-01", "2026-09-03"]          # 有日报的 09-02 跳过
    days.clear()
    cli.main(["backfill", "--from", "2026-09-02", "--to", "2026-09-02", "--force"])
    assert days == ["2026-09-02"]                        # --force 重跑


def test_enroll_faces_dispatches(monkeypatch, tmp_path):
    seen = {}

    def fake_enroll(data_dir, rest_url, session=None):
        seen["dir"] = str(data_dir)
        seen["session"] = session
        return 3

    monkeypatch.setattr(cli, "_enroll_for_cli", fake_enroll)
    s = _settings()
    s.storage.data_dir = tmp_path
    monkeypatch.setattr(cli, "_load_settings", lambda path: s)
    rc = cli.main(["enroll-faces"])
    assert rc == 0 and "faces" in seen["dir"]
    assert seen["session"] is not None   # spec §5：identities 行同步需要 session


def test_web_subcommand_starts_uvicorn(monkeypatch, tmp_path):
    import uvicorn

    seen = {}

    def fake_run(app, host=None, port=None, **kw):
        seen["app"] = app
        seen["host"] = host
        seen["port"] = port

    monkeypatch.setattr(uvicorn, "run", fake_run)
    s = _settings()
    s.storage.data_dir = tmp_path
    monkeypatch.setattr(cli, "_load_settings", lambda path: s)
    rc = cli.main(["web"])
    assert rc == 0
    assert seen["host"] == "0.0.0.0" and seen["port"] == 8080
    assert seen["app"].state.settings is s
