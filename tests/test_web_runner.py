# tests/test_web_runner.py —— 后台补跑互斥与 post_run 钩子
import threading
import time

from tests.test_web_index import _settings


def _wait_busy_clear(m, timeout=100):
    for _ in range(timeout):
        if not m.busy:
            return
        time.sleep(0.05)


def test_submit_runs_in_thread_and_reports_busy(tmp_path, monkeypatch):
    import camdigest.pipeline.orchestrator as orch
    from camdigest.web.runner import RunManager

    settings = _settings(tmp_path)
    seen = []
    release = threading.Event()

    def fake_run_day(date, settings, until=None):
        seen.append(date)
        release.wait(timeout=5)          # 模拟长任务

    monkeypatch.setattr(orch, "run_day", fake_run_day)
    m = RunManager(settings)
    assert m.submit("2026-09-29") is True
    assert m.busy is True
    assert m.submit("2026-09-28") is False       # 忙 → 拒绝不排队
    release.set()
    _wait_busy_clear(m)
    assert seen == ["2026-09-29"] and m.busy is False


def test_post_run_hook_called_on_success(tmp_path, monkeypatch):
    import camdigest.pipeline.orchestrator as orch
    from camdigest.web.runner import RunManager

    settings = _settings(tmp_path)
    monkeypatch.setattr(orch, "run_day", lambda date, s, until=None: None)
    m = RunManager(settings)
    hooks = []
    m.post_run.append(hooks.append)
    m.submit("2026-09-29")
    _wait_busy_clear(m)
    assert hooks == ["2026-09-29"]


def test_post_run_hook_error_swallowed(tmp_path, monkeypatch):
    import camdigest.pipeline.orchestrator as orch
    from camdigest.web.runner import RunManager

    settings = _settings(tmp_path)
    monkeypatch.setattr(orch, "run_day", lambda date, s, until=None: None)
    m = RunManager(settings)

    def boom(date):
        raise RuntimeError("hook failed")

    m.post_run.append(boom)
    m.submit("2026-09-29")
    _wait_busy_clear(m)
    assert m.busy is False                       # 钩子异常不影响收尾
