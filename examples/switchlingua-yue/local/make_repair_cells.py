#!/usr/bin/env python3
"""Build the repair cells: re-synthesize the clips QC flagged as missing content, with a shorter prompt.

Why they need a different prompt rather than just a retry: the defect tracks prompt length, and only prompt
length (measured on the full corpus, `qc/summary.json` + the per-duration breakdown):

    截断率(相对 vratio < 0.75x 本格中位)   prompt <6s   6-12s   12-20s    >=20s
    B_pathref (病理库, prompt 中位 10.2s)     3.3%      2.6%     7.9%    43.3%
    A_pathref                                 3.4%      3.5%     8.5%    39.0%
    B_slref   (SL 库, prompt 中位 5.0s)       0.7%      0.3%     0.0%     0.0%
    A_slref                                   1.7%      1.5%     0.0%     0.0%

ASR on the flagged clips shows the mechanism is deletion, not mispronunciation: 60.9% of the reference units
are missing below 0.5x and 34.0% between 0.5-0.72x, while insertions stay at 1.7-2.6%. A long prompt makes the
model end generation early, so feeding it the same long prompt again is expected to fail again -- hence the
substitution, drawn from the <=12 s part of the same library and preferring the same speaker when that speaker
has a short enough clip (keeps the 28-doctor mix intact).

Writes, per cell: `cells/<cell>_repair/text`, `cells/<cell>_repair_prompt_pairs.json`, plus
`qc/repair_plan.json` and `qc/redo_wav_list.txt`. It deletes nothing -- the runbook removes the listed wavs
explicitly, then runs synth_test.py with `--split <cell>_repair --llm <cell>` so the audio lands in the cell's
own directory.

    python -I local/make_repair_cells.py --max-prompt-sec 12
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data_synth'))
    ap.add_argument('--cells', nargs='*', default=['A_slref', 'A_pathref', 'B_slref', 'B_pathref'])
    ap.add_argument('--max-prompt-sec', type=float, default=12.0)
    args = ap.parse_args()

    libs = {}
    for name in ('slref', 'pathref'):
        rows = [json.loads(l) for l in
                (args.root / f'libraries/{name}.library.jsonl').read_text().splitlines()]
        libs[name] = rows

    plan, redo = {}, []
    for cell in args.cells:
        dfile = args.root / f'qc/{cell}.defects.jsonl'
        if not dfile.is_file() or not dfile.read_text().strip():
            print(f'{cell}: 无缺陷条目')
            continue
        defects = [json.loads(l) for l in dfile.read_text().splitlines()]
        lib = libs[cell.split('_', 1)[1]]
        pairs = json.loads((args.root / f'cells/{cell}_prompt_pairs.json').read_text())
        text = {l.split(maxsplit=1)[0]: l.split(maxsplit=1)[1]
                for l in (args.root / f'cells/{cell}/text').read_text().splitlines()}

        short = [r for r in lib if r['dur'] <= args.max_prompt_sec]
        dur_of = {r['utt']: r['dur'] for r in lib}
        by_speaker = defaultdict(list)
        for r in short:
            by_speaker[r.get('item') or r.get('case')].append(r)
        cursor = defaultdict(int)          # per-speaker rotation, so repairs do not pile onto one clip
        global_i = 0

        new_pairs, subs = {}, []
        for d in sorted(defects, key=lambda x: x['utt']):
            u = d['utt']
            orig = pairs[u]
            pool = by_speaker.get(orig['prompt_speaker'], [])
            if pool:
                p = pool[cursor[orig['prompt_speaker']] % len(pool)]
                cursor[orig['prompt_speaker']] += 1
            else:                            # this speaker has no clip under the cap: take the next global one
                p = short[global_i % len(short)]
                global_i += 1
            if p['utt'] == orig['prompt_utt']:   # never re-feed the prompt that just failed
                global_i += 1
                p = short[global_i % len(short)]
            new_pairs[u] = dict(orig, prompt_utt=p['utt'], prompt_wav=p['wav'], prompt_text=p['text'],
                                prompt_speaker=p.get('item') or p.get('case'),
                                library=cell.split('_', 1)[1], repaired_from=orig['prompt_utt'],
                                repair_reason=f"vr_rel={d['vr_rel']} vratio={d['vratio']} "
                                              f"old_prompt_dur={dur_of[orig['prompt_utt']]:.1f}s")
            subs.append({'utt': u, 'vr_rel': d['vr_rel'], 'old_prompt': orig['prompt_utt'],
                         'old_prompt_dur': dur_of[orig['prompt_utt']],
                         'old_prompt_speaker': orig['prompt_speaker'], 'new_prompt': p['utt'],
                         'new_prompt_dur': p['dur'], 'new_speaker': p.get('item') or p.get('case'),
                         'kept_speaker': (p.get('item') or p.get('case')) == orig['prompt_speaker']})
            redo.append(str(args.root / f'synth/{cell}/{u}.wav'))

        d = args.root / f'cells/{cell}_repair'
        d.mkdir(parents=True, exist_ok=True)
        (d / 'text').write_text('\n'.join(f'{u} {text[u]}' for u in sorted(new_pairs)) + '\n')
        (args.root / f'cells/{cell}_repair_prompt_pairs.json').write_text(
            json.dumps(new_pairs, indent=1, ensure_ascii=False) + '\n')
        same = sum(1 for s in subs if s['kept_speaker'])
        plan[cell] = {'defects': len(defects), 'same_speaker_kept': same,
                      'speakers_covered': len({s['new_speaker'] for s in subs}),
                      'old_prompt_dur_median': sorted(s['old_prompt_dur'] for s in subs)[len(subs) // 2],
                      'new_prompt_dur_max': max(s['new_prompt_dur'] for s in subs),
                      'low': sum(1 for x in defects if x['vr_rel'] < 1), 'high': len(defects) - sum(
                          1 for x in defects if x['vr_rel'] < 1)}
        print(f"{cell}: 返工 {len(new_pairs)} 条，同医生保留 {same} 条，"
              f"新 prompt 最长 {plan[cell]['new_prompt_dur_max']:.1f}s")

    (args.root / 'qc/repair_plan.json').write_text(json.dumps(plan, indent=1, ensure_ascii=False) + '\n')
    (args.root / 'qc/redo_wav_list.txt').write_text('\n'.join(redo) + ('\n' if redo else ''))
    print(f'\n总计 {len(redo)} 条 -> qc/redo_wav_list.txt（脚本不删文件，删除走 runbook 里显式一条 xargs rm）')


if __name__ == '__main__':
    main()
