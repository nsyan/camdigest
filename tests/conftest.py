# tests/conftest.py
import subprocess

import pytest


def make_video(path, seconds=6, with_audio=True) -> str:
    cmd = ["ffmpeg", "-y", "-f", "lavfi",
           "-i", f"testsrc=duration={seconds}:size=320x240:rate=10"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-g", "10"]  # 1s GOP@10fps：整数秒切点有关键帧，S10 切片对齐
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)
    return str(path)


@pytest.fixture(scope="session")
def video_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("media") / "20260929100000.mp4"
    return make_video(p)


@pytest.fixture(scope="session")
def silent_video_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("media") / "20260929110000.mp4"
    return make_video(p, with_audio=False)
