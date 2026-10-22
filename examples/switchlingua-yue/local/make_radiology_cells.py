#!/usr/bin/env python3
"""Turn data/radiology/core_balanced.jsonl into a synthesis cell with evidence-backed normalization.

The scene is dictated in Cantonese-English code-switch, so numbers are spoken in Cantonese even inside English
sentences -- the same convention the pathology fine-tune was trained on (`十乘十乘二 mm`, `三點八 c m`,
`七十五 mm`). Digits therefore become Hong Kong Traditional Chinese numerals; English prose and units stay as
written. The corpus's own CJK is already HK Traditional (住佢係咁咗咯唔啊啦...), so no script conversion is needed.

Probing the fine-tuned TTS (data_mix_clean/ft_ep1) and reading it back with three ASR models (base qwen3, the
domain fine-tuned qwen3, whisper-yue) showed which written forms the model cannot speak:

  * `3.2 x 2.1 cm` speaks the cross as the letter X and can lose a dimension   -> 三點二乘二點一 cm
  * `1-2 mm` loses the lower bound                                            -> 一至二 mm
  * `<4 mm` loses both the comparator and the number                          -> 少於四 mm
  * `C5-6` / `L4/5` / `L5/S1` are not recovered at all                        -> C五至六 / L四至五 / L五至S一
  * `C5 and C6` is broken too (letter+digit tokens)                           -> never emit that form
  * `03/11/2024` is not recovered                                             -> 二零二四年十一月三日
  * `86%`, `7th`, `15 mm`, `10.7 cm`, `o'clock` are fine as written           -> leave alone

Units stay exactly as written (`mm`, `cm`, `ml`, `HU`, `cm3`, `cm/s` ...) per the user's decision; only the
number in front of a unit is converted, and a digit that belongs to an alphanumeric token (cm3, 3D) is not.
English number words already spelled out ("two", "one") are left in English.

Writes (into data_radiology/):
  corpus/<tag>/{text,source_map.jsonl}    normalized snapshot + raw record per id
  normalization_audit.jsonl               every changed row, with the rules that fired
  cells/<tag>_slref/{text} + pairs        the full cell and its `_pilot` subset
  plan.json                               input md5, rule counts, prompt assignment stats

    python -I local/make_radiology_cells.py --dry-run
    python -I local/make_radiology_cells.py

extended.jsonl is a strict superset of core_balanced.jsonl (4,001 shared ids + 16,032 new), so the second
pass excludes what the first already covered instead of re-synthesizing it:

    python -I local/make_radiology_cells.py --input data/radiology/extended.jsonl \
        --out data_radiology_ext --tag RADX --exclude-ids-from data/radiology/core_balanced.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']
# The scene is read in Cantonese: numbers, ×, ranges and dates are spoken in Cantonese even inside an English
# sentence -- exactly the convention the pathology fine-tune was trained on (`十乘十乘二 mm`, `三點八 c m`,
# `七十五 mm`). So digits become HK Traditional Chinese numerals rather than English words.
CN_DIGITS = '零一二三四五六七八九'
MONTH_CN = {'January': '一月', 'February': '二月', 'March': '三月', 'April': '四月', 'May': '五月',
            'June': '六月', 'July': '七月', 'August': '八月', 'September': '九月', 'October': '十月',
            'November': '十一月', 'December': '十二月'}


def cn_nat(n: int) -> str:
    """Natural Cantonese reading: 15 -> 十五, 105 -> 一百零五, 2024 -> 二千零二十四."""
    assert n >= 0
    if n < 10:
        return CN_DIGITS[n]
    if n < 20:
        return '十' + (CN_DIGITS[n - 10] if n > 10 else '')
    if n < 100:
        return CN_DIGITS[n // 10] + '十' + (CN_DIGITS[n % 10] if n % 10 else '')
    for unit, val in (('萬', 10000), ('千', 1000), ('百', 100)):
        if n >= val:
            hi, rest = divmod(n, val)
            s = (CN_DIGITS[hi] if hi < 10 else cn_nat(hi)) + unit
            if rest == 0:
                return s
            return s + ('零' if rest < val // 10 else '') + cn_nat(rest)
    raise AssertionError(n)


def cn_digits(s: str) -> str:
    """Digit by digit, for years and decimal fractions: 2024 -> 二零二四."""
    return ''.join(CN_DIGITS[int(c)] for c in s)


def cn_number(tok: str) -> str:
    if '.' in tok:
        a, b = tok.split('.', 1)
        return f'{cn_nat(int(a))}點{cn_digits(b)}'
    return cn_nat(int(tok))


def norm_dates(t: str) -> tuple[str, int]:
    """DD/MM/YYYY -> 二零二四年十一月三日 (Hong Kong convention, year spoken digit by digit)."""
    n = 0

    def slash(m):
        nonlocal n
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        n += 1
        return f'{cn_digits(m.group(3))}年{cn_nat(mo)}月{cn_nat(d)}日'

    t = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', slash, t)

    def dmy(m):
        nonlocal n
        n += 1
        return f'{cn_digits(m.group(3))}年{MONTH_CN[m.group(2)]}{cn_nat(int(m.group(1)))}日'

    t = re.sub(r'\b(\d{1,2})\s+(' + '|'.join(MONTHS) + r')\s+(\d{4})\b', dmy, t)

    def my(m):
        nonlocal n
        n += 1
        return f'{cn_digits(m.group(2))}年{MONTH_CN[m.group(1)]}'

    t = re.sub(r'\b(' + '|'.join(MONTHS) + r')\s+(\d{4})\b', my, t)
    return t, n


def norm_levels(t: str) -> tuple[str, int]:
    n = 0

    def sub(m):
        nonlocal n
        n += 1
        l1, d1, l2, d2 = m.group(1), int(m.group(2)), m.group(4), int(m.group(5))
        if l2:                      # T12-L1, L5/S1 -> T十二至L一, L五至S一
            return f'{l1}{cn_nat(d1)}至{l2}{cn_nat(d2)}'
        return f'{l1}{cn_nat(d1)}至{cn_nat(d2)}'      # C5-6 -> C五至六, L4/5 -> L四至五

    t = re.sub(r'\b([CTLSD])(\d{1,2})\s*([-/])\s*([CTLSD])?(\d{1,2})\b', sub, t)
    return t, n


def norm_letter_digits(t: str) -> tuple[str, int]:
    """Letter+digit tokens are the family the probe showed breaks: `C5 and C6` came out as "c and c loc".
    Sequential MRI notations are normalised first (T1-weighted, T2/FLAIR), then any remaining level token."""
    n = 0

    def sig(m):
        nonlocal n
        n += 1
        return f'T{cn_nat(int(m.group(1)))} {m.group(2)}'      # T1-weighted -> T一 weighted

    t = re.sub(r'\bT([12])(A|WI)\b',
               lambda m: f'T{cn_nat(int(m.group(1)))} ' + ('A' if m.group(2) == 'A' else 'W I'), t)
    t = re.sub(r'\bT([12])\s*[-/]\s*(FLAIR|weighted)\b', sig, t)

    def lvl(m):
        nonlocal n
        n += 1
        return f'{m.group(1)}{cn_nat(int(m.group(2)))}'       # C5 -> C五

    t = re.sub(r'\b([CTLSD])(\d{1,2})\b', lvl, t)
    return t, n


def normalize(text: str) -> tuple[str, list[dict]]:
    out, edits = text, []

    def apply(rule: str, fn):
        nonlocal out
        new = fn(out)
        if isinstance(new, tuple):
            new, k = new
        else:
            k = 1 if new != out else 0
        if new != out:
            edits.append({'rule': rule, 'n': k})
            out = new

    apply('strip leading bullet', lambda t: re.sub(r'^\s*[—–]\s*', '', t))
    apply('strip leading punctuation', lambda t: re.sub(r'^[,;:]\s*', '', t))
    apply('strip quotes', lambda t: t.replace('"', '').replace('“', '').replace('”', ''))
    apply('collapse duplicate punctuation', lambda t: re.sub(r'\s+', ' ',
                                                             re.sub(r'([,;:])\s*\1+', r'\1', t)).strip())
    apply('date DD/MM/YYYY -> 年月日', lambda t: norm_dates(t))
    apply('vertebral levels -> CJK numerals', lambda t: norm_levels(t))
    apply('letter+digit tokens -> CJK numerals', lambda t: norm_letter_digits(t))
    apply('multiplication x -> 乘', lambda t: re.sub(r'(?<=\d)\s*[x×]\s*(?=\d)', '乘', t))
    # only up to three digits on each side: a radiologic range is small, while `2903-255` is an extension
    apply('numeric range -> 至', lambda t: re.sub(r'\b(\d{1,3}(?:\.\d+)?)\s*[-–—]\s*(\d{1,3}(?:\.\d+)?)\b',
                                                  r'\1至\2', t))
    apply('numeric "to" range -> 至', lambda t: re.sub(r'\b(\d{1,3}(?:\.\d+)?)\s+to\s+(\d{1,3}(?:\.\d+)?)\b',
                                                       r'\1至\2', t))
    apply('comparator < -> 少於', lambda t: re.sub(r'<\s*(?=\d)', '少於', t))
    apply('comparator > -> 大於', lambda t: re.sub(r'>\s*(?=\d)', '大於', t))
    apply('percent -> 百分之', lambda t: re.sub(r'(\d+(?:\.\d+)?)\s*%', r'百分之\1', t))
    apply('ordinal -> 第', lambda t: re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', r'第\1', t))
    apply('abbreviation dots', lambda t: t.replace('Dr.', 'Doctor').replace('approx.', 'approximately'))
    # units are kept exactly as written (user decision); only put a space between a digit and a unit so the
    # number conversion below can see the number without touching the unit (5cm3 -> 5 cm3 -> 五 cm3)
    apply('space before unit', lambda t: re.sub(
        r'(?<=\d)(?=(?:mm|cm3|cm2|cm/s|cm|ml|mL|mg|HU|kV|mGy|mSv|Gy|cc|mmHg|bpm|L|IU)\b)', ' ', t))
    # years are spoken digit by digit (二零二四), except when the number is a quantity with a unit (2000 ml)
    apply('year -> 二零二四', lambda t: re.sub(
        r'\b(?:19|20)\d{2}\b(?!\s*(?:ml|mL|mm|cm|mg|g|kg|cc|L)\b)', lambda m: cn_digits(m.group(0)), t))
    # every remaining number, but never a digit inside an alphanumeric token (cm3, 3D)
    apply('numbers -> CJK numerals', lambda t: re.sub(
        r'(?<![A-Za-z0-9])(\d+(?:\.\d+)?)(?![A-Za-z])', lambda m: cn_number(m.group(1)), t))
    return out, edits


def pick(rows: list[dict], n: int, wants: list[tuple]) -> list[dict]:
    taken, out = set(), []
    for pred, k in wants:
        got = 0
        for r in rows:
            if got >= k:
                break
            if r['utt'] not in taken and pred(r):
                out.append(r)
                taken.add(r['utt'])
                got += 1
    by = {}
    for r in rows:
        if r['utt'] not in taken:
            by.setdefault(r['scene'], []).append(r)
    while len(out) < n and any(by.values()):
        for g in sorted(by):
            if by[g] and len(out) < n:
                out.append(by[g].pop(0))
    return sorted(out, key=lambda r: r['utt'])


def deal(rows: list[dict], speaker: str) -> list[dict]:
    """Round-robin across speakers (shortest clip first within a speaker) so the scan below meets a new voice
    early; without this, picking the first unused prompt leaves the pilot at 56 of 60 distinct speakers."""
    groups = {}
    for r in rows:
        groups.setdefault(r[speaker], []).append(r)
    for v in groups.values():
        v.sort(key=lambda r: r['dur'])
    out = []
    for i in range(max(len(v) for v in groups.values())):
        for g in sorted(groups):
            if len(groups[g]) > i:
                out.append(groups[g][i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=REPO / 'data/radiology/core_balanced.jsonl')
    ap.add_argument('--out', type=Path, default=Path('data_radiology'))
    ap.add_argument('--library', type=Path, default=Path('data_synth/libraries/slref.library.jsonl'))
    ap.add_argument('--tag', default='RAD', help='cell/corpus name prefix: <tag>_slref')
    ap.add_argument('--exclude-ids-from', type=Path,
                    help='jsonl whose ids are already covered elsewhere (extended.jsonl is a strict '
                         'superset of core_balanced.jsonl: only the 16,032 new ids need synthesizing)')
    ap.add_argument('--pilot', type=int, default=60)
    ap.add_argument('--cps', type=float, default=16.0, help='characters per second of generated speech, measured')
    ap.add_argument('--max-prompt-sec', type=float, default=12.0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    raw = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    excluded = 0
    if args.exclude_ids_from:
        skip = {json.loads(l)['id'] for l in args.exclude_ids_from.read_text().splitlines() if l.strip()}
        before = len(raw)
        raw = [r for r in raw if r['id'] not in skip]
        excluded = before - len(raw)
    rows, audit, rule_counts = [], [], Counter()
    for r in raw:
        text, edits = normalize(r['text'])
        for e in edits:
            rule_counts[e['rule']] += e['n']
        if edits:
            audit.append({'utt': r['id'], 'edits': edits, 'before': r['text'], 'after': text})
        rows.append({'utt': r['id'], 'text': text, 'raw': r['text'], 'scene': r['scene'],
                     'split': r['split'], 'group': r['group_id'], 'type': r['text_type'],
                     'lang': r['text_language']})

    lib = [json.loads(l) for l in args.library.read_text().splitlines()]
    report = {'input': str(args.input),
              'input_md5': hashlib.md5(args.input.read_bytes()).hexdigest(),
              'input_rows': len(raw) + excluded, 'excluded_ids': excluded, 'targets': len(rows),
              'changed_texts': len(audit),
              # rows touched per rule; date/level entries count substitutions (a row may contain several)
              'rule_row_counts': dict(rule_counts),
              'library': str(args.library), 'library_size': len(lib),
              'split': dict(Counter(r['split'] for r in rows)),
              'scene': dict(Counter(r['scene'] for r in rows)),
              'type': dict(Counter(r['type'] for r in rows))}
    print(json.dumps(report, indent=1, ensure_ascii=False))
    for a in audit[:5]:
        print('  e.g.', a['utt'], '|', a['before'][:90], '=>', a['after'][:90])

    if args.dry_run:
        print('--dry-run: nothing written')
        return

    (args.out / f'corpus/{args.tag}').mkdir(parents=True, exist_ok=True)
    (args.out / f'corpus/{args.tag}/text').write_text(
        '\n'.join(f"{r['utt']} {r['text']}" for r in rows) + '\n')
    (args.out / f'corpus/{args.tag}/source_map.jsonl').write_text('\n'.join(json.dumps(
        {'utt': r['utt'], 'text_raw': r['raw'], 'text_tts': r['text'], 'scene': r['scene'],
         'split': r['split'], 'group_id': r['group'], 'text_type': r['type'], 'lang': r['lang']},
        ensure_ascii=False) for r in rows) + '\n')
    (args.out / 'normalization_audit.jsonl').write_text(
        '\n'.join(json.dumps(a, ensure_ascii=False) for a in audit) + '\n')

    # prompt assignment: dealt rotation over the SL library, but never feed a prompt much longer than the
    # target (the model warns and degrades when the prompt text dominates); expected duration ~ chars / cps
    pairs, used, order = {}, Counter(), deal(lib, 'item')
    assigned, covered, ptr = set(), set(), 0
    spk = lambda p: p.get('item') or p.get('case')       # noqa: E731
    for r in sorted(rows, key=lambda x: x['utt']):
        cap = min(args.max_prompt_sec, max(2.5, 1.5 * len(r['text']) / args.cps))
        # NB: the variable must not be called `pick` -- that shadows the module-level pick() used for the pilot
        chosen, fallback = None, None
        for k in range(len(order)):
            q = order[(ptr + k) % len(order)]
            if q['utt'] in assigned or q['dur'] > cap:
                continue
            if spk(q) not in covered:        # prefer a voice not used yet: keeps speaker coverage maximal
                chosen, ptr = q, (ptr + k + 1) % len(order)
                break
            if fallback is None:
                fallback = (q, (ptr + k + 1) % len(order))
        if chosen is None and fallback:
            chosen, ptr = fallback
        if chosen is None:                   # nothing unused under the cap: accept a longer prompt
            for k in range(len(order)):
                q = order[(ptr + k) % len(order)]
                if q['utt'] not in assigned:
                    chosen, ptr = q, (ptr + k + 1) % len(order)
                    break
        if chosen is None:                   # more targets than library entries: reuse from the dealt order
            chosen = order[ptr % len(order)]
            ptr += 1
        assigned.add(chosen['utt'])
        covered.add(spk(chosen))
        used[chosen['utt']] += 1
        p = chosen
        pairs[r['utt']] = {'prompt_utt': p['utt'], 'prompt_wav': p['wav'], 'prompt_text': p['text'],
                           'same_item': False, 'cos': None, 'fallback': False, 'library': 'slref',
                           'prompt_speaker': p.get('item'), 'prompt_dur': p['dur']}

    # coverage repair: 4,001 picks over 7,272 clips can leave a voice untouched (measured: 1,700 of 1,701).
    # Hand each missing voice to one of the longest targets -- their cap is the loosest -- keeping prompts distinct.
    text_of = {r['utt']: r['text'] for r in rows}
    missing = [q for q in order if spk(q) not in covered and q['dur'] <= args.max_prompt_sec]
    repaired = 0
    for q, u in zip(missing, sorted(pairs, key=lambda x: -len(text_of[x]))):
        if spk(q) in covered:
            continue
        old = pairs[u]['prompt_utt']
        assigned.discard(old)
        used[old] -= 1
        if used[old] == 0:
            del used[old]
        pairs[u].update(prompt_utt=q['utt'], prompt_wav=q['wav'], prompt_text=q['text'],
                        prompt_speaker=spk(q), prompt_dur=q['dur'])
        assigned.add(q['utt'])
        covered.add(spk(q))
        used[q['utt']] += 1
        repaired += 1
    print(f'coverage repair: {repaired} target(s) re-pointed, {len(covered)} voices covered')
    cell = f'{args.tag}_slref'
    (args.out / 'cells' / cell).mkdir(parents=True, exist_ok=True)
    (args.out / 'cells' / cell / 'text').write_text('\n'.join(f"{r['utt']} {r['text']}" for r in rows) + '\n')
    (args.out / 'cells' / f'{cell}_prompt_pairs.json').write_text(
        json.dumps(pairs, indent=1, ensure_ascii=False) + '\n')

    # predicates run on the NORMALISED text, so they look for the Cantonese forms actually sent to the model
    want = {
        'multiplication': (lambda r: '乘' in r['text'], 8),
        'percent': (lambda r: '百分之' in r['text'], 4),
        'date': (lambda r: re.search(r'年[一二三四五六七八九十]+月', r['text']) is not None, 4),
        'range': (lambda r: re.search(r'[零一二三四五六七八九十百千]至[零一二三四五六七八九十百千]', r['text']) is not None, 4),
        'level': (lambda r: re.search(r'[CTLSD][一二三四五六七八九十]', r['text']) is not None, 6),
        't1_t2': (lambda r: 'T一' in r['text'] or 'T二' in r['text'], 4),
        'comparator': (lambda r: '少於' in r['text'] or '大於' in r['text'], 2),
        'unit_digits': (lambda r: re.search(r'\b(?:cm3|cm/s|mmHg|cc)\b', r['text']) is not None, 3),
        'ordinal': (lambda r: '第' in r['text'], 3),
        'control': (lambda r: r['type'] == 'control_english', 4),
        'codeswitch': (lambda r: r['lang'] == 'mixed', 6),
        'instruction': (lambda r: r['type'].startswith('instruction'), 4),
        'no_numbers': (lambda r: not re.search(r'[\d零一二三四五六七八九十百千萬點]', r['text']), 5),
        'long': (lambda r: len(r['text']) > 300, 2),
    }
    pilot = pick(rows, args.pilot, list(want.values()))
    (args.out / 'cells' / f'{cell}_pilot').mkdir(parents=True, exist_ok=True)
    (args.out / 'cells' / f'{cell}_pilot' / 'text').write_text(
        '\n'.join(f"{r['utt']} {r['text']}" for r in pilot) + '\n')
    (args.out / 'cells' / f'{cell}_pilot_prompt_pairs.json').write_text(
        json.dumps({r['utt']: pairs[r['utt']] for r in pilot}, indent=1, ensure_ascii=False) + '\n')

    dur_of = {p['utt']: p['dur'] for p in lib}
    plan = dict(report, cells={cell: {'n': len(rows), 'n_pilot': len(pilot),
                                      'n_prompts': len(used), 'speakers': len(covered),
                                      # voices that actually have a clip under the length cap; one library item
                                      # (1830) has a single 21.8 s clip, so 1,700 of 1,701 are reachable
                                      'speakers_usable': len({r['item'] for r in lib if r['dur'] <= args.max_prompt_sec}),
                                      'prompt_reuse_max': max(used.values()),
                                      'prompt_dur_median': sorted(dur_of[u] for u in used)[len(used) // 2]}},
                pilot_overlap={k: sum(1 for r in pilot if pred(r)) for k, (pred, _) in want.items()})
    (args.out / 'plan.json').write_text(json.dumps(plan, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(plan['cells'], indent=1, ensure_ascii=False))
    print(json.dumps(plan['pilot_overlap'], indent=1, ensure_ascii=False))
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()