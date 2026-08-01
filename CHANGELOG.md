# Changelog

All notable changes to Calibra are documented here.

## [0.7.0] — Dataset Integrity

### Added

- **`calibra integrity`** — new front-door command answering "can I trust this dataset?" before quality, coverage, or optimization matter. Findings are grouped into Critical / Warnings / Passed rather than led with a single score; an `Integrity Score` is still computed but demoted to a summary line.
  - Timestamp consistency and sensor sync (jitter, dropout, camera lag, action/observation alignment)
  - Episode completeness (statistical short-episode detection)
  - Duplicate frame detection
  - Camera freeze detection
  - Blur detection
  - Image integrity checks (duplicate/freeze/blur) now available for **LeRobot v1** datasets via the new opt-in `--decode-images` flag
- `LeRobotReader(decode_images=True)` — decodes HuggingFace `Image`-feature columns for LeRobot v1 datasets. Off by default (increases load time/memory); no effect on the existing v2/v3 fast path.
- Homepage and documentation reorganized around the Integrity → Quality → Coverage → Optimize workflow (new `docs/integrity.md`, updated README, `mkdocs.yml` nav, demo assets under `docs/demo.tape`/`docs/demo_fixture.py`).
- Hugging Face Space (`spaces/app.py`) reorganized to check Dataset Integrity first, ahead of the Quality/Coverage score.

### Notes

- Duplicate-frame, camera-freeze, and blur checks work out of the box on HDF5/Isaac Lab/robomimic data, and on LeRobot v1 via `--decode-images`. LeRobot v2/v3 (video-encoded) datasets are intentionally out of scope for this release — `--decode-images` prints a warning and has no effect there.

## Planned next

- **Vision Integrity for video-backed LeRobot datasets (v2/v3)** — decode a sampled subset of frames from LeRobot's mp4-encoded v2/v3 datasets so duplicate-frame/camera-freeze/blur detection work there too. Needs a new video-decoding dependency and parsing per-episode chunk/timestamp offsets from `meta/episodes/*.parquet` (v3's multi-episode-per-file layout) — deferred until it can be validated against representative real datasets rather than shipped speculatively.
