"""Thin launcher around `lerobot.record` that fixes the left-arrow re-record crash.

Upstream bug: LeRobotDataset.clear_episode_buffer() (called when you press ← to
re-record) does shutil.rmtree() on the episode's image dir WITHOUT first draining
the async image-writer threads. Those threads (4/camera) are still flushing PNGs
into the dir, so the delete races and dies with:

    OSError: [Errno 66] Directory not empty: '.../images/observation.images.front/episode_00000X'

Every other cleanup path in lerobot calls self._wait_image_writer() first; this one
forgets to. We can't edit lerobot (reference-only clone), so we monkeypatch the one
method to wait, then hand off to lerobot's normal CLI. All CLI args pass straight
through — record_episode.sh calls this instead of `python -m lerobot.record`.
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

_orig_clear_episode_buffer = LeRobotDataset.clear_episode_buffer


def _clear_episode_buffer_wait_first(self) -> None:
    # Drain the async image writer before rmtree, otherwise re-record races the
    # writer threads and crashes with "Directory not empty".
    self._wait_image_writer()
    _orig_clear_episode_buffer(self)


LeRobotDataset.clear_episode_buffer = _clear_episode_buffer_wait_first


if __name__ == "__main__":
    from lerobot.record import record

    record()
