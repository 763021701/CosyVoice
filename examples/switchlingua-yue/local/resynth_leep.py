#!/usr/bin/env python3
"""Re-speak existing A/B LEEP clips.

The wav is synthesized from the same prompt, with the word LEEP pronounced
as "leap". The manifest label and the corpus text stay "LEEP". "sleep" is
left alone.

    python local/resynth_leep.py
"""

from __future__ import annotations

import base64
import json
import re
import wave
from pathlib import Path

from synth_supp_omni import BATCH, encode_audio, post_batch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_synth"
WORD = re.compile(r"\b[Ll][Ee][Ee][Pp]\b")
ARMS = ("A_pathref", "A_slref", "B_pathref", "B_slref")
TEXT_FILES = [
    DATA / "corpus/A/text",
    DATA / "corpus/B/text",
    DATA / "cells/A_pathref/text",
    DATA / "cells/A_slref/text",
    DATA / "cells/B_pathref/text",
    DATA / "cells/B_slref/text",
]


def seconds(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return round(handle.getnframes() / handle.getframerate(), 3)


def label(text: str) -> str:
    return WORD.sub("LEEP", text)


def spoken(text: str) -> str:
    return WORD.sub("leap", text)


def main() -> None:
    jobs = []
    for arm in ARMS:
        manifest = json.loads((DATA / "synth" / f"{arm}.manifest.json").read_text())
        pairs = json.loads((DATA / "cells" / f"{arm}_prompt_pairs.json").read_text())
        hit = [row for row in manifest if WORD.search(row["text"])]
        for row in hit:
            text = row["text"]
            say = spoken(text)
            if WORD.search(say) or not re.search(r"\bleap\b", say):
                raise SystemExit(f"{arm} {row['utt']} pronunciation unchanged: {text}")
            prompt = pairs[row["utt"]]
            jobs.append({
                "arm": arm,
                "utt": row["utt"],
                "label": label(text),
                "tts": say,
                "prompt_wav": prompt["prompt_wav"],
                "prompt_text": prompt["prompt_text"],
                "dest": DATA / "synth" / arm / f"{row['utt']}.wav",
            })
    print(f"{len(jobs)} clips", flush=True)
    cache: dict[str, str] = {}
    failed = []
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        items = [
            {"input": job["tts"], "ref_audio": encode_audio(job["prompt_wav"], cache), "ref_text": job["prompt_text"]}
            for job in chunk
        ]
        try:
            result = post_batch("http://127.0.0.1:8091", "cosyvoice3", items, 300.0)
        except Exception as exc:
            print(f"batch {start} failed: {exc}", flush=True)
            failed.extend({"utt": job["utt"], "arm": job["arm"], "error": str(exc)} for job in chunk)
            continue
        by_index = {item["index"]: item for item in result.get("results", [])}
        for i, job in enumerate(chunk):
            item = by_index.get(i)
            if not item or item.get("status") != "success" or not item.get("audio_data"):
                failed.append({"utt": job["utt"], "arm": job["arm"], "error": None if not item else item.get("error")})
                continue
            audio = base64.b64decode(item["audio_data"])
            if not audio.startswith(b"RIFF"):
                failed.append({"utt": job["utt"], "arm": job["arm"], "error": "not a wav"})
                continue
            job["dest"].write_bytes(audio)
        print(f"{start + len(chunk)}/{len(jobs)}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} failed: {failed[:3]}")

    done = {(job["arm"], job["utt"]): job for job in jobs}
    for arm in ARMS:
        path = DATA / "synth" / f"{arm}.manifest.json"
        manifest = json.loads(path.read_text())
        for row in manifest:
            job = done.get((arm, row["utt"]))
            if job is None:
                continue
            row["text"] = job["label"]
            row["synth_sec"] = seconds(job["dest"])
        path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    utts = {job["utt"] for job in jobs}
    for path in TEXT_FILES:
        lines = []
        changed = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt, text = line.split(maxsplit=1)
            if utt in utts and WORD.search(text):
                text = label(text)
                changed += 1
            lines.append(f"{utt} {text}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{path.relative_to(ROOT)} label updates {changed}", flush=True)
    print("ok", flush=True)


if __name__ == "__main__":
    main()
