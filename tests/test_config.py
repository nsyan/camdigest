# tests/test_config.py
import textwrap
from pathlib import Path

import pytest

from camdigest.config import Settings


def _yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_load_minimal(tmp_path):
    p = _yaml(tmp_path, """\
        schedule: { daily_at: "03:00", timezone: Asia/Shanghai }
        cameras:
          - { id: gate, name: 大门, device: gate, lens: single, dir: /media/nvr/gate }
        models:
          recognition: { protocol: openai, base_url: http://x/v1, model: m1, api_key: k1 }
          report: { protocol: openai, base_url: http://x/v1, model: m2, api_key: k2 }
        storage: { data_dir: /data }
        """)
    s = Settings.load(p)
    assert s.cameras[0].id == "gate"
    assert s.highlight.tiers == [5, 10, 30, 60]      # 架构 §9 默认值
    assert s.prefilter.scene_threshold == 0.03
    assert s.anomaly.stranger_while_family_absent.enabled is True


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    p = _yaml(tmp_path, """\
        schedule: { daily_at: "03:00", timezone: Asia/Shanghai }
        cameras:
          - { id: gate, name: 大门, device: gate, dir: /media/nvr/gate }
        models:
          recognition: { protocol: openai, base_url: http://x/v1, model: m1, api_key: "${TEST_KEY}" }
          report: { protocol: anthropic, base_url: http://y, model: m2, api_key: "${TEST_KEY}" }
        storage: { data_dir: /data }
        """)
    s = Settings.load(p)
    assert s.models.recognition.api_key == "sk-xyz"
    assert s.models.report.protocol == "anthropic"


def test_dual_lens_validation(tmp_path):
    # 双摄设备必须 fixed+ptz 成对；ptz 无覆盖时自动给 0.08
    p = _yaml(tmp_path, """\
        schedule: { daily_at: "03:00", timezone: Asia/Shanghai }
        cameras:
          - { id: living,    name: 客厅固定, device: living, lens: fixed, dir: /a }
          - { id: living_pt, name: 客厅云台, device: living, lens: ptz,   dir: /b }
        models:
          recognition: { protocol: openai, base_url: http://x/v1, model: m1, api_key: k1 }
          report: { protocol: openai, base_url: http://x/v1, model: m2, api_key: k2 }
        storage: { data_dir: /data }
        """)
    s = Settings.load(p)
    assert s.cameras[1].effective_scene_threshold() == 0.08


def test_broken_pair_rejected(tmp_path):
    p = _yaml(tmp_path, """\
        schedule: { daily_at: "03:00", timezone: Asia/Shanghai }
        cameras:
          - { id: living, name: 客厅固定, device: living, lens: fixed, dir: /a }
        models:
          recognition: { protocol: openai, base_url: http://x/v1, model: m1, api_key: k1 }
          report: { protocol: openai, base_url: http://x/v1, model: m2, api_key: k2 }
        storage: { data_dir: /data }
        """)
    with pytest.raises(ValueError, match="fixed.*ptz"):
        Settings.load(p)
