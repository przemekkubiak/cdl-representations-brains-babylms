#!/usr/bin/env python
"""Download the stimulus audio for a dataset (git-annex symlinks -> real files).

The BOLD downloader already knows how to turn an annex symlink into an OpenNeuro
URL; this reuses that machinery for the `stimuli/` tree, which nothing else
fetches. Needed for the positive control: ds003604 is an AUDITORY design (every
stimulus is a .wav), so the low-level acoustic control that TODO.md section 0
asks for requires the audio itself.

Small: ds003604's stimuli are 352 files of ~300 KB, ~120 MB total, against a
350 GB disk floor. Nothing here needs the streaming discipline that BOLD does.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.batch_download_bold import download_file, get_candidate_urls  # noqa: E402
from src.datasets import get_dataset  # noqa: E402


def fetch(path: Path, data_dir: Path, spec) -> tuple[Path, bool, str]:
    """Resolve one symlink to real bytes. Already-present files are left alone."""
    if path.exists() and not path.is_symlink():
        return path, True, "present"
    if path.is_symlink() and path.resolve().exists():
        return path, True, "present (annex object)"
    urls = get_candidate_urls(path, data_dir, spec)
    tmp = path.with_suffix(path.suffix + ".part")
    last = "no candidate urls"
    for url in urls:
        try:
            download_file(url, tmp)
            if path.is_symlink():
                path.unlink()
            tmp.replace(path)
            return path, True, "downloaded"
        except Exception as e:  # noqa: BLE001 - report, try the next mirror
            last = f"{type(e).__name__}: {e}"
    if tmp.exists():
        tmp.unlink()
    return path, False, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ds003604")
    ap.add_argument("--data-root", default="data/brain")
    ap.add_argument("--pattern", default="",
                    help="glob to match; default = every stimulus media type, "
                         "case-insensitively")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    a = ap.parse_args()

    spec = get_dataset(a.dataset)
    data_dir = Path(a.data_root) / spec.require_downloadable()
    # spec.stimulus_dir() points at the *characteristics* subfolder for
    # ds003604, not the audio, so search the BIDS stimuli/ tree and fall back to
    # the registry path only if that finds nothing.
    # Case matters and extensions vary: ds003604 ships .wav, ds002236 ships .WAV,
    # ds006239 is entirely .bmp, ds001894 has both. A single lowercase glob finds
    # nothing for two of the four and exits "no files matching *.wav".
    MEDIA = {".wav", ".bmp", ".jpg", ".jpeg", ".png", ".mp3", ".aiff", ".aif"}

    def _match(root):
        if a.pattern:
            return sorted(root.rglob(a.pattern))
        return sorted(f for f in root.rglob("*")
                      if f.suffix.lower() in MEDIA)

    stim_dir = data_dir / "stimuli"
    files = _match(stim_dir) if stim_dir.exists() else []
    if not files:
        alt = spec.stimulus_dir(a.data_root)
        if alt and alt.exists():
            stim_dir, files = alt, _match(alt)
    if not files:
        print(f"no stimulus media found under {stim_dir}")
        sys.exit(1)
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} stimulus files under {stim_dir}")

    done = failed = skipped = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(fetch, f, data_dir, spec): f for f in files}
        for fut in as_completed(futs):
            path, ok, msg = fut.result()
            if not ok:
                failed += 1
                print(f"  FAIL {path.name}: {msg}")
            elif msg.startswith("present"):
                skipped += 1
            else:
                done += 1
                if done % 50 == 0:
                    print(f"  {done} downloaded")

    total_mb = sum(f.stat().st_size for f in files if f.exists()) / 1e6
    print(f"downloaded {done}, already present {skipped}, failed {failed} "
          f"-- {total_mb:.0f} MB on disk")
    sys.exit(1 if failed and not done else 0)


if __name__ == "__main__":
    main()
