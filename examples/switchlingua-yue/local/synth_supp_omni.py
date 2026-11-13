#!/usr/bin/env python3
"""Synthesize data_synth/corpus/supp through the vLLM-Omni CosyVoice3 server.

Two arms, same 1,020 texts:
  supp_pathref  prompts already stored on the manifest (pathology, <=12 s)
  supp_slref    SwitchLingua clips <=12 s, dealt round-robin

The server must already be up. Each POST /v1/audio/speech/batch carries 8
items, each with its own ref_audio and ref_text. Existing non-empty wavs are
skipped, so a rerun resumes.

    python local/synth_supp_omni.py --arm pathref
    python local/synth_supp_omni.py --arm slref
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = ROOT / "data_synth/corpus/supp/manifest.jsonl"
SLREF = ROOT / "data_synth/libraries/slref.library.jsonl"
BATCH = 8
MAX_PROMPT_SEC = 12.0
MIN_PROMPT_SEC = 1.0


def encode_audio(path: str, cache: dict[str, str]) -> str:
    hit = cache.get(path)
    if hit is not None:
        return hit
    raw = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    mime = {".wav": "audio/wav", ".flac": "audio/flac"}.get(suffix, "audio/wav")
    hit = "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")
    cache[path] = hit
    return hit


def deal(rows: list[dict], speaker: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[speaker], []).append(r)
    for v in groups.values():
        v.sort(key=lambda r: r["utt"])
    order = []
    width = max(len(v) for v in groups.values())
    for i in range(width):
        for g in sorted(groups):
            if len(groups[g]) > i:
                order.append(groups[g][i])
    return order


def slref_prompts() -> list[dict]:
    rows = []
    for line in SLREF.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if MIN_PROMPT_SEC <= r["dur"] <= MAX_PROMPT_SEC and r["text"].strip():
            rows.append(r)
    return deal(rows, "item")


def jobs(arm: str) -> list[dict]:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: r["utt"])
    if arm == "pathref":
        return rows
    prompts = slref_prompts()
    if len(prompts) < len(rows):
        raise SystemExit(f"slref prompts {len(prompts)} < texts {len(rows)}")
    paired = []
    for row, prompt in zip(rows, prompts):
        item = dict(row)
        item["prompt_utt"] = prompt["utt"]
        item["prompt_wav"] = prompt["wav"]
        item["prompt_case"] = prompt["item"]
        item["prompt_dur"] = prompt["dur"]
        item["prompt_text"] = prompt["text"]
        paired.append(item)
    return paired


def post_batch(api_base: str, model: str, items: list[dict], timeout: float) -> dict:
    body = {
        "model": model,
        "response_format": "wav",
        "items": items,
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        api_base.rstrip("/") + "/v1/audio/speech/batch",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("pathref", "slref"), required=True)
    ap.add_argument("--api-base", default="http://127.0.0.1:8091")
    ap.add_argument("--model", default="cosyvoice3")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    pending = jobs(args.arm)
    out_dir = ROOT / "data_synth/synth" / f"supp_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_path = ROOT / "data_synth/cells" / f"supp_{args.arm}_prompt_pairs.json"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    pair_path.write_text(
        json.dumps(
            {
                r["utt"]: {
                    "prompt_utt": r["prompt_utt"],
                    "prompt_wav": r["prompt_wav"],
                    "prompt_text": r["prompt_text"],
                    "prompt_case": r["prompt_case"],
                    "prompt_dur": r["prompt_dur"],
                    "library": args.arm,
                }
                for r in pending
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    todo = [r for r in pending if not (out_dir / f"{r['utt']}.wav").is_file() or (out_dir / f"{r['utt']}.wav").stat().st_size == 0]
    print(f"{args.arm}: {len(pending)} texts, {len(pending) - len(todo)} already on disk, {len(todo)} to synthesize", flush=True)
    cache: dict[str, str] = {}
    failed: list[dict] = []
    t0 = time.perf_counter()
    done = 0
    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        items = [
            {
                "input": r["tts_text"] if r.get("tts_text") else r["text"],
                "ref_audio": encode_audio(r["prompt_wav"], cache),
                "ref_text": r["prompt_text"],
            }
            for r in chunk
        ]
        try:
            result = post_batch(args.api_base, args.model, items, args.timeout)
        except Exception as exc:
            print(f"batch {start} request failed: {exc}", flush=True)
            for r in chunk:
                failed.append({"utt": r["utt"], "error": str(exc)})
            continue
        by_index = {item["index"]: item for item in result.get("results", [])}
        for i, r in enumerate(chunk):
            item = by_index.get(i)
            if not item or item.get("status") != "success" or not item.get("audio_data"):
                err = None if not item else item.get("error")
                failed.append({"utt": r["utt"], "error": err or "missing result"})
                continue
            audio = base64.b64decode(item["audio_data"])
            if not audio.startswith(b"RIFF"):
                failed.append({"utt": r["utt"], "error": f"not a wav ({audio[:12]!r})"})
                continue
            (out_dir / f"{r['utt']}.wav").write_bytes(audio)
            done += 1
        elapsed = time.perf_counter() - t0
        print(
            f"{args.arm} {start + len(chunk)}/{len(todo)} ok={done} fail={len(failed)} {elapsed:.0f}s",
            flush=True,
        )

    fail_path = out_dir.parent / f"supp_{args.arm}.failed.json"
    fail_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {done} wavs, {len(failed)} failed -> {fail_path}", flush=True)
    if failed:
        raise SystemExit(1)
    write_arm_manifest(args.arm, pending, out_dir)


def write_arm_manifest(arm: str, rows: list[dict], out_dir: Path) -> None:
    manifest = []
    for r in sorted(rows, key=lambda item: item["utt"]):
        wav = out_dir / f"{r['utt']}.wav"
        with wave.open(str(wav)) as handle:
            sec = round(handle.getnframes() / handle.getframerate(), 3)
        manifest.append({
            "utt": r["utt"],
            "synth_wav": f"data_synth/synth/supp_{arm}/{r['utt']}.wav",
            "synth_sec": sec,
            "real_sec": None,
            "fallback": False,
            "same_item": False,
            "text": r["text"],
        })
    path = out_dir.parent / f"supp_{arm}.manifest.json"
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(manifest)})", flush=True)


if __name__ == "__main__":
    main()
