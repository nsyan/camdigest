# tests/test_s1.py
from pathlib import Path

from camdigest import db
from camdigest.config import CameraCfg
from camdigest.pipeline.s1_index import index_camera


def test_index_idempotent(tmp_path, video_file):
    url = f"sqlite:///{tmp_path}/t.db"  # 内存库多连接各开新库，必须用文件库
    cam_dir = tmp_path / "gate"
    cam_dir.mkdir()
    (cam_dir / "20260929100000.mp4").write_bytes(Path(video_file).read_bytes())
    cam = CameraCfg(id="gate", name="大门", device="gate", dir=cam_dir, pattern="20260929*.mp4")
    db.init_db(url)
    with db.session_scope(url) as s:
        assert index_camera(cam, s) == 1
    with db.session_scope(url) as s:
        assert index_camera(cam, s) == 0            # 幂等：已入库跳过
        assert s.query(db.MediaFile).count() == 1
        mf = s.query(db.MediaFile).one()
        assert mf.start_ts.hour == 10               # 文件名解析成功
