#!/usr/bin/env python3
"""Quality control for the generated corpus: is every clip real speech, in the requested voice, and diverse?

A text-only generation corpus has no ground-truth audio, so `asr_score.py`'s duration ratio and its real-audio
floor do not exist here. What can still be measured without references, and what the gates enforce (thresholds
and their empirical anchors are in the plan, section 4):

  completeness     every expected utt produced a wav, or is listed in <cell>.failed.json
  readable         the wav opens and its header matches its content
  duration         1.0 s <= sec <= 45 s
  peak             peak >= 0.01 on every clip -- R2's digital silence sat at peak 0.000 (RESULTS.md 5)
  dominant_token   share of the single most frequent speech token <= 0.35, the attractor detector that caught
                   R2 (training audio maxed at 0.716 before cleaning, 0.208 after -- RESULTS.md 6.2)
  content_ratio    voiced time per character, RELATIVE to the cell's own median, inside [0.75, 1.60] on
                   >=98% of clips. The only gate that sees missing words: an early-EOS clip is short of
                   speech yet perfectly audible (peak 0.99), so peak / duration / dominant_token all pass it.
                   ASR on flagged clips says deletion, not mispronunciation: 60.9% of reference units missing
                   below 0.5x, 34.0% between 0.5-0.72x, insertions only 1.7-2.6%
  voice_follow     campplus cosine(synth, its own prompt): median >= 0.70, >= 0.60 on 90% of clips, p05 >= 0.45,
                   anchored on measured same-speaker real pairs (SL 0.831 median / 0.687 p05; pathology 0.853)
  voice_contrast   cos(synth, its own prompt) minus cos(synth, the prompt dealt half a rotation away): the
                   difference must be >= 0.15 at the median and >= 0.05 on 90% of clips. This replaces a
                   nearest-neighbour gate, which cannot work here -- with 7,272 candidates and only ~24 true
                   same-speaker clips, build_prompt_pairs.py:9-10 measured a 97-100% chance of at least one
                   spurious match at any threshold that keeps coverage. NN agreement is still REPORTED per
                   cell (nn_same_speaker) as a description of the library, never as a gate.
  diversity        distinct prompts used == number of targets, distinct speakers == min(library, targets)
  no_voice_collapse mean pairwise cosine across synth embeddings < 0.90

ASR round-trip is deliberately NOT a gate: on this domain the ASR floor is 86-96%, so only cross-cell deltas
mean anything; those come from asr_score.py / fw_hyp.py / punct_metric.py.

Synth embeddings are cached in `qc/<cell>.emb.pt` and library embeddings in `qc/library.emb.pt`, because the
full corpus is 31.5k clips and both ONNX models run on CPU.

    python -I local/qc_synth.py --pilot
    python -I local/qc_synth.py --tokens sample
    python -I local/qc_synth.py --repair-list
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clean_pathology import speech_runs  # noqa: E402

MIN_DUR, MAX_DUR = 1.0, 45.0
PEAK_MIN = 0.01
DOM_MAX = 0.35
COS_MED_MIN, COS_OK, COS_OK_MIN, COS_P05_MIN = 0.70, 0.60, 0.90, 0.45
CTRL_MARGIN, CTRL_FRAC, CTRL_MED, CROSS_MAX = 0.05, 0.90, 0.15, 0.90
# voiced speech per second of text, measured on this corpus: 800 random clips across the four cells give a
# median of exactly 1.00 with this value
VOICED_CPS = 16.5
# The band is RELATIVE to each cell's own median, because the two arms sit at different absolute values:
# vratio median 1.07/1.08 with SwitchLingua prompts (their clip length median 5.0 s) vs 0.90/0.92 with
# pathology prompts (10.2 s). An absolute [0.6, 1.8] band therefore measures the arm's speaking density, not
# its defects -- and the defect rate is what matters: measured against prompt duration, clips whose prompt
# ran >=20 s are missing content 43.3% (B) / 39.0% (A) of the time, vs 2.6-3.5% under 12 s. ASR on the
# flagged clips confirms the mechanism is deletion, not speed: 60.9% of ref units missing (vratio<0.5) and
# 34.0% (0.5-0.72), with insertions only 1.7-2.6%. Absolute values stay in the report.
VR_REL_LO, VR_REL_HI, VR_IN_FRAC = 0.75, 1.60, 0.98


def sample(utt: str, pct: float) -> bool:
    """Deterministic selection: md5, not hash(), whose salt changes every process."""
    return int(hashlib.md5(utt.encode()).hexdigest()[:8], 16) % 1000 < int(pct * 1000)


def embed_paths(paths: list[str], onnx: str, threads: int) -> dict:
    """campplus 192-d vectors, the same recipe as tools/extract_embedding.py:26-37."""
    import onnxruntime
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    sess = onnxruntime.InferenceSession(onnx, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name

    def one(path):
        audio, sr = torchaudio.load(path, backend='soundfile')
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        return sess.run(None, {name: feat.unsqueeze(0).numpy()})[0].flatten().tolist()

    with ThreadPoolExecutor(threads) as ex:
        return {p: v for p, v in zip(paths, ex.map(one, paths))}


def token_top1(paths: list[str], onnx: str, threads: int) -> dict:
    """(dominant id, share) per clip, replicating tools/extract_speech_token.py:26-38. >30 s inputs are
    unsupported by that tokenizer and come back as None rather than a misleading 0."""
    import onnxruntime
    import torchaudio
    import whisper
    sess = onnxruntime.InferenceSession(onnx, providers=['CPUExecutionProvider'])
    ins = [i.name for i in sess.get_inputs()]

    def one(path):
        audio, sr = torchaudio.load(path, backend='soundfile')
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        if audio.shape[1] / 16000 > 30:
            return None
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        tok = sess.run(None, {ins[0]: feat.numpy(),
                             ins[1]: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        if not tok:
            return None
        dom, n = Counter(tok).most_common(1)[0]
        return int(dom), round(n / len(tok), 4)

    with ThreadPoolExecutor(threads) as ex:
        return dict(zip(paths, ex.map(one, paths)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='*', default=['A_slref', 'A_pathref', 'B_slref', 'B_pathref'])
    ap.add_argument('--root', type=Path, default=Path('data_synth'))
    ap.add_argument('--pilot', action='store_true', help='score only each cell’s pilot utt set')
    ap.add_argument('--tokens', choices=['off', 'sample', 'all'], default='all')
    ap.add_argument('--token-sample', type=float, default=0.10)
    ap.add_argument('--no-voices', action='store_true', help='skip campplus: peak/duration/token gates only')
    ap.add_argument('--model-dir', type=Path, default=Path('data_mix_clean/ft_ep1'),
                    help='provides campplus.onnx and speech_tokenizer_v3.onnx')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--repair-list', action='store_true',
                    help='print wavs that are unreadable, <1 s, or shorter than their header claims; deletes nothing')
    args = ap.parse_args()

    qd = args.root / 'qc'
    qd.mkdir(parents=True, exist_ok=True)
    libs = {k: [json.loads(l) for l in (args.root / f'libraries/{k}.library.jsonl').read_text().splitlines()]
            for k in ('slref', 'pathref')}

    # prompt-side embeddings: mostly already on disk from feature extraction; only the 102 hesitation clips
    # that training never used have to be computed once, then cached
    emb_cache = qd / 'library.emb.pt'
    known = {}
    for r in (Path('data'), Path('data_mix')):
        for s in ('train', 'dev', 'test'):
            p = r / s / 'utt2embedding.pt'
            if p.is_file():
                known.update({k: np.asarray(v, dtype=np.float32)
                              for k, v in torch.load(p, map_location='cpu', weights_only=True).items()})
    if emb_cache.is_file():
        known.update({k: np.asarray(v, dtype=np.float32)
                      for k, v in torch.load(emb_cache, map_location='cpu', weights_only=True).items()})
    todo = {r['utt']: r['wav'] for rows in libs.values() for r in rows if r['utt'] not in known}
    if todo and not args.no_voices:
        print(f'library embeddings missing: {len(todo)} (computing once)')
        # key by the LIBRARY utt, not Path(wav).stem: prepare_mix_data names pathology utts B<file> / C<case>_<file>,
        # so the stem of from_denoised/12009_seg_0002.wav is missing the B prefix the library rows carry
        fresh = embed_paths(list(todo.values()), str(args.model_dir / 'campplus.onnx'), args.threads)
        for u, p in todo.items():
            known[u] = np.asarray(fresh[p], dtype=np.float32)
        torch.save({k: torch.from_numpy(v) for k, v in known.items()}, emb_cache)
    print(f'prompt-side embeddings available: {len(known)}')

    lib_emb, lib_speaker, lib_utts = {}, {}, {}
    for k, rows in libs.items():
        if not known or any(r['utt'] not in known for r in rows):
            continue
        lib_emb[k] = np.stack([known[r['utt']] for r in rows])
        lib_speaker[k] = np.asarray([r.get('item') or r.get('case') for r in rows])
        lib_utts[k] = [r['utt'] for r in rows]

    summary, problems = {}, []
    for cell in args.cells:
        pairs = json.loads((args.root / f'cells/{cell}{"_pilot" if args.pilot else ""}_prompt_pairs.json').read_text())
        # diversity is a property of the assignment plan, which exists before a single clip is synthesized, so
        # it is always checked against the full cell pairs -- a 60-target pilot cannot prove library coverage
        full = json.loads((args.root / f'cells/{cell}_prompt_pairs.json').read_text())
        text = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
                for l in (args.root / f'cells/{cell}/text').read_text().splitlines()}
        wdir = args.root / 'synth' / cell
        fl = args.root / f'synth/{cell}.failed.json'
        failed = {x['utt'] for x in json.loads(fl.read_text())} if fl.is_file() else set()
        rows, broken = [], []
        for u in sorted(pairs):
            p = wdir / f'{u}.wav'
            if not p.is_file():
                continue
            try:
                x, sr = sf.read(p)
            except Exception as e:  # noqa: BLE001 - a truncated file is exactly what we are hunting
                rows.append({'utt': u, 'unreadable': True, 'err': str(e)[:80]})
                broken.append(str(p))
                continue
            if x.ndim > 1:
                x = x.mean(1)
            x = np.asarray(x)
            runs, _, _ = speech_runs(x, sr)
            # no internal gap when there is one speech run; no speech at all is caught by the peak gate
            gap = max((runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)), default=0.0)
            voiced = sum(b - a for a, b in runs)
            declared = int(sf.info(p).frames)
            if len(x) < MIN_DUR * sr or len(x) > MAX_DUR * sr or declared != len(x):
                broken.append(str(p))
            rows.append({'utt': u, 'unreadable': False, 'sec': round(len(x) / sr, 3),
                         'peak': round(float(np.abs(x).max()), 4), 'nspeech': len(runs), 'gap': round(gap, 3),
                         'voiced': round(voiced, 3),
                         # how much speech the clip actually contains, relative to what its text needs:
                         # catches both early-EOS truncation (<band) and letter-by-letter rambling (>band),
                         # neither of which peak/duration/dom_share can see
                         'vratio': round(voiced / (len(text[u]) / VOICED_CPS), 3) if text.get(u) else None})
        live = [r for r in rows if not r['unreadable']]

        if args.repair_list:
            for r in rows:
                if r['unreadable'] or r['sec'] < MIN_DUR or r['sec'] > MAX_DUR:
                    print(f"REPAIR {wdir / (r['utt'] + '.wav')} "
                          f"{'unreadable: ' + r.get('err', '') if r['unreadable'] else f'sec={r["sec"]}'}")
            print(f'{cell}: {len(rows)} found, {len(broken)} need repair')
            continue

        if args.tokens != 'off' and live:
            sel = [r['utt'] for r in live if args.tokens == 'all' or sample(r['utt'], args.token_sample)]
            tops = token_top1([str(wdir / f'{u}.wav') for u in sel],
                              str(args.model_dir / 'speech_tokenizer_v3.onnx'), args.threads)
            by_utt = {Path(p).stem: v for p, v in tops.items()}
            for r in live:
                if r['utt'] in by_utt and by_utt[r['utt']] is not None:
                    r['dom_id'], r['dom_share'] = by_utt[r['utt']]

        cross = None
        lib = cell.split('_', 1)[1]
        if live and not args.no_voices and lib in lib_emb:
            cache = qd / f'{cell}.emb.pt'
            have = {}
            if cache.is_file():
                have = {k: np.asarray(v, dtype=np.float32)
                        for k, v in torch.load(cache, map_location='cpu', weights_only=True).items()}
            need = [r['utt'] for r in live if r['utt'] not in have]
            if need:
                new = embed_paths([str(wdir / f'{u}.wav') for u in need],
                                  str(args.model_dir / 'campplus.onnx'), args.threads)
                for u in need:
                    have[u] = np.asarray(new[str(wdir / f'{u}.wav')], dtype=np.float32)
                torch.save({k: torch.from_numpy(v) for k, v in have.items()}, cache)
            libn = lib_emb[lib] / np.linalg.norm(lib_emb[lib], axis=1, keepdims=True)
            # paired control: the prompt dealt to the target half a rotation away in this cell. Dealing puts a
            # different speaker there, so cos(own prompt) - cos(control prompt) measures voice transfer without
            # depending on library size -- which is exactly what the nearest-neighbour test could not do
            order = sorted(full)
            idx = {u: i for i, u in enumerate(order)}
            half = max(len(order) // 2, 1)
            vecs = []
            for r in live:
                if r['utt'] not in have:
                    continue
                v = have[r['utt']]
                vn = v / (np.linalg.norm(v) + 1e-9)
                pu = pairs[r['utt']]['prompt_utt']
                r['cos_prompt'] = round(float(vn @ known[pu] / (np.linalg.norm(known[pu]) + 1e-9)), 4)
                cu = full[order[(idx[r['utt']] + half) % len(order)]]['prompt_utt']
                r['cos_control'] = round(float(vn @ known[cu] / (np.linalg.norm(known[cu]) + 1e-9)), 4)
                r['ctrl_win'] = bool(r['cos_prompt'] - r['cos_control'] >= CTRL_MARGIN)
                j = int(np.argmax(libn @ vn))
                r['nn_utt'] = lib_utts[lib][j]
                r['nn_same_speaker'] = bool(lib_speaker[lib][j] == pairs[r['utt']]['prompt_speaker'])
                vecs.append(vn)
            if len(vecs) > 2:
                pick = np.stack(vecs[:1000])
                cm = pick @ pick.T
                cross = float(cm[np.triu_indices(len(pick), 1)].mean())

        secs = [r['sec'] for r in live]
        cos = sorted(r['cos_prompt'] for r in live if 'cos_prompt' in r)
        ctrl = sorted(r['cos_control'] for r in live if 'cos_control' in r)
        cdiff = [r['cos_prompt'] - r['cos_control'] for r in live if 'cos_control' in r]
        cwin = [r['ctrl_win'] for r in live if 'ctrl_win' in r]
        dom = [r['dom_share'] for r in live if 'dom_share' in r]
        vr = [r['vratio'] for r in live if r.get('vratio') is not None]
        med_vr = float(np.median(vr)) if vr else None
        for r in live:
            if r.get('vratio') is not None and med_vr:
                r['vr_rel'] = round(r['vratio'] / med_vr, 3)
        rel = [r['vr_rel'] for r in live if r.get('vr_rel') is not None]
        defects = [r for r in live if r.get('vr_rel') is not None and not VR_REL_LO <= r['vr_rel'] <= VR_REL_HI]
        g = {'expected': len(pairs), 'have': len(rows),
             'unreadable': int(sum(1 for r in rows if r['unreadable'])),
             'failed_listed': len(failed), 'dom_checked': len(dom), 'cos_checked': len(cos),
             'out_of_band': len(defects)}
        voices_on = (lib in lib_emb) and not args.no_voices and bool(cos)
        # bool() everywhere: numpy scalars serialize as np.True_ and json.dumps then raises TypeError
        n_lib = len(libs[lib])
        n_spk = len({r.get('item') or r.get('case') for r in libs[lib]})
        gates = {
            'completeness': bool(g['have'] + len(failed) >= len(pairs)),
            'readable': bool(g['unreadable'] == 0),
            'duration': bool(secs) and all(MIN_DUR <= s <= MAX_DUR for s in secs),
            'peak': bool(secs) and all(r['peak'] >= PEAK_MIN for r in live),
            'dominant_token': (all(d <= DOM_MAX for d in dom) if args.tokens != 'off' and dom else 'skipped'),
            # not "every clip inside the band" -- ~1.7% of healthy clips sit outside it by the control
            # measurement, so the gate is the share inside, and the outside ones go to defects.jsonl to redo
            'content_ratio': bool(rel) and float(np.mean([VR_REL_LO <= x <= VR_REL_HI for x in rel])) >= VR_IN_FRAC,
            'voice_follow': (bool(np.median(cos) >= COS_MED_MIN
                                  and float(np.mean([c >= COS_OK for c in cos])) >= COS_OK_MIN
                                  and float(np.quantile(cos, 0.05)) >= COS_P05_MIN)) if voices_on else 'skipped',
            'voice_contrast': (bool(cwin) and float(np.mean(cwin)) >= CTRL_FRAC
                               and float(np.median(cdiff)) >= CTRL_MED) if voices_on else 'skipped',
            # the rotation covers the whole library once targets >= library, and every speaker as soon as the
            # dealt order has passed each group -- the bounds below are what that guarantee means exactly
            'diversity': bool(len({v['prompt_utt'] for v in full.values()}) == min(n_lib, len(full))
                              and len({v['prompt_speaker'] for v in full.values()}) == min(n_spk, len(full))),
            'no_voice_collapse': bool(cross < CROSS_MAX) if cross is not None else 'skipped',
        }
        q = lambda xs, p: float(np.quantile(xs, p)) if xs else None
        stats = {'sec_median': q(secs, 0.5), 'sec_max': max(secs) if secs else None,
                 'peak_min': min(r['peak'] for r in live) if live else None,
                 'gap_median': q([r['gap'] for r in live], 0.5), 'gap_max': max(r['gap'] for r in live) if live else None,
                 'dom_share_median': q(dom, 0.5), 'dom_share_max': max(dom) if dom else None,
                 'cos_median': q(cos, 0.5), 'cos_p05': q(cos, 0.05),
                 'cos_ge_0.60': round(float(np.mean([c >= COS_OK for c in cos])), 3) if cos else None,
                 'cos_control_median': q(ctrl, 0.5), 'ctrl_diff_median': q(cdiff, 0.5),
                 'ctrl_win': round(float(np.mean(cwin)), 3) if cwin else None,
                 # reported only, never gated: see the docstring on why clip/speaker NN cannot be a gate here
                 'nn_same_speaker': round(float(np.mean([bool(r.get('nn_same_speaker')) for r in live])), 3) if live else None,
                 'cross_synth_cos': round(cross, 3) if cross is not None else None,
                 # absolute vratio per cell, then the relative band this gate actually enforces
                 'vratio_median': med_vr, 'vratio_p05': q(vr, 0.05), 'vratio_p95': q(vr, 0.95),
                 'vr_rel_in_band': round(float(np.mean([VR_REL_LO <= x <= VR_REL_HI for x in rel])), 4) if rel else None,
                 'vr_rel_low': sum(1 for x in rel if x < VR_REL_LO),
                 'vr_rel_high': sum(1 for x in rel if x > VR_REL_HI),
                 'prompts_used_full': len({v['prompt_utt'] for v in full.values()}),
                 'speakers_used_full': len({v['prompt_speaker'] for v in full.values()}),
                 'library': [n_lib, n_spk],
                 'pilot_prompts': len({v['prompt_utt'] for v in pairs.values()}),
                 'pilot_speakers': len({v['prompt_speaker'] for v in pairs.values()})}
        stats = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()}
        bad = [k for k, v in gates.items() if v is False]
        summary[cell] = {'counts': g, 'stats': stats, 'gates': gates, 'failed_gates': bad}
        problems += [f'{cell}:{k}' for k in bad]
        (qd / f'{cell}.qc.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
        # the redo list: what to delete so the next unsharded pass regenerates exactly these
        (qd / f'{cell}.defects.jsonl').write_text(
            ('\n'.join(json.dumps({'utt': r['utt'], 'vratio': r['vratio'], 'vr_rel': r['vr_rel'],
                                   'sec': r['sec'], 'voiced': r['voiced'], 'peak': r['peak'],
                                   'prompt_utt': pairs[r['utt']]['prompt_utt'],
                                   'prompt_speaker': pairs[r['utt']]['prompt_speaker'],
                                   'text': text[r['utt']][:160]}, ensure_ascii=False) for r in defects) + '\n')
            if defects else '')
        print(f'{cell}: ' + ' '.join(f'{k}={"ok" if v is True else ("skip" if v == "skipped" else "FAIL")}'
                                     for k, v in gates.items()))
        print(f'   {json.dumps(stats, ensure_ascii=False)}')

    (qd / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    print(f"\n{'PASS' if not problems else 'FAIL ' + ' '.join(problems)}  -> {qd / 'summary.json'}")


if __name__ == '__main__':
    main()
