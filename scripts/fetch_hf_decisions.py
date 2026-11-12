"""Cache the HF decision dataset as a compact local JSONL.

`huggingface.co` is unreachable from this machine, but the dataset viewer API is, so this pages
through `/rows` (100 at a time) and keeps only the fields the corpus cross-check needs. Image URLs
are dropped: they are large, and the PNGs are already in `assets/`.

Output one record per decision:
    {"puzzle_id", "step_index", "row", "column", "label", "past", "rows", "columns",
     "difficulty", "source_date", "split"}

Co-ordinates are 1-based and top-down ("Rows top to bottom; columns left to right"), so subtract 1
to reach our 0-based space.

Usage:
    uv run python scripts/fetch_hf_decisions.py                 # train split
    uv run python scripts/fetch_hf_decisions.py --split validation --split test_small
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = "zuhairsan/train-tracks-decisions"
API = "https://datasets-server.huggingface.co/rows"


def fetch_page(split: str, offset: int, length: int, attempts: int = 8) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": split, "offset": offset, "length": length}
    )
    url = f"{API}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "train-track-grpo/0.1"})
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 15 * (attempt + 1)
                print(f"  rate limited at {split} offset {offset}; sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            time.sleep(min(30, 2**attempt))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"{split} offset {offset}: giving up after {attempts} attempts ({last})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="append", default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "hf")
    ap.add_argument("--page", type=int, default=100)
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between pages (the viewer is rate limited)")
    args = ap.parse_args()

    splits = args.split or ["train"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        out = args.out_dir / f"{split}_decisions.jsonl"
        done = 0
        if out.exists():
            lines = out.read_text("utf-8").splitlines()
            valid = []
            for line in lines:
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    break
                valid.append(line)
            if len(valid) != len(lines):
                print(f"{split}: dropping {len(lines) - len(valid)} partial line(s)", flush=True)
                out.write_text("\n".join(valid) + ("\n" if valid else ""), encoding="utf-8")
            done = len(valid)
            if done:
                print(f"{split}: resuming from {done} cached decisions", flush=True)
        mode = "a" if done else "w"
        total = None
        with out.open(mode, encoding="utf-8", newline="\n") as handle:
            offset = done
            while True:
                if offset > done:
                    time.sleep(args.delay)
                page = fetch_page(split, offset, args.page)
                total = page["num_rows_total"]
                if not page["rows"]:
                    break
                for item in page["rows"]:
                    row = item["row"]
                    state = row["state"]
                    handle.write(
                        json.dumps(
                            {
                                "puzzle_id": row["puzzle_id"],
                                "step_index": row["step_index"],
                                "row": state["target_cell"]["row"],
                                "column": state["target_cell"]["column"],
                                "label": row["label"],
                                "past": len(state["past_actions"]),
                                "rows": row["rows"],
                                "columns": row["columns"],
                                "difficulty": row["difficulty"],
                                "source_date": row["source_date"],
                                "split": row["split"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                offset += len(page["rows"])
                if offset % 1000 < args.page:
                    print(f"{split}: {offset}/{total}", flush=True)
                if offset >= total:
                    break
        print(f"{split}: {offset}/{total} decisions -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
