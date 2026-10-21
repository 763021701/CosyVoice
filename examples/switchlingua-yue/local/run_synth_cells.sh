#!/bin/bash
# Launch the four generation cells on the packaged ft_ep1 model. Deliberately no logic: data_synth/plan.json
# already says which cells exist and how big they are, so this file only wraps local/synth_test.py such that
# one cell == one command, and a crashed or killed worker is restarted with that identical command.
#
#   bash local/run_synth_cells.sh smoke    # 33 clips on both B cells, measures today's per-clip seconds
#   bash local/run_synth_cells.sh pilot    # 60 stratified clips x 4 cells, into the cell's own synth dir
#   bash local/run_synth_cells.sh full     # everything (31,566 clips), one nohup worker per cell
#   bash local/run_synth_cells.sh status   # per-cell counts vs plan.json, failed.json, disk, GPU
#
# Pilot writes into synth/<cell>/ (arm name = cell name, split name = <cell>_pilot) so the full run's resume
# pass skips those clips instead of synthesizing them a second time.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
model=data_mix_clean/ft_ep1
out=data_synth/synth
export OMP_NUM_THREADS=4
mode=${1:-status}

cells=$(python -I -c "import json;print(' '.join(json.load(open('data_synth/plan.json'))['cells']))")
mkdir -p data_synth/logs $out

run() {  # $1 = split (dir under data_synth/cells), $2 = arm name (dir under $out), $3 = extra args (optional)
  # ${3:-} not $3: under `set -u` an omitted third argument killed every background worker instantly while the
  # script still reported exit 0, so a "completed" pilot meant nothing.
  python local/synth_test.py --model-dir $model --llm "$2=$model/llm.pt" \
    --data-root data_synth/cells --split "$1" --out-root $out --seed 1986 ${3:-} \
    >> data_synth/logs/"$2".log 2>&1
}

case $mode in
  smoke)  # longest texts, one library each: the pathology prompts are ~2x the length of SL ones
    for c in B_slref B_pathref; do
      echo "smoke $c $(date +%H:%M:%S)"
      ( time run "${c}_pilot" "$c" "--limit 33" ) 2>&1 | tail -6
    done ;;
  pilot)
    pids=()
    for c in $cells; do
      run "${c}_pilot" "$c" &
      pids+=($!)
      echo "pilot $c pid $! -> data_synth/logs/$c.log"
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    echo "pilot done $(date +%H:%M), worker rc=$rc"
    bash local/run_synth_cells.sh status
    exit $rc ;;
  full)
    # the pilot gate table is the only thing standing between a bad prompt library and 15 hours of audio
    if [ -f data_synth/qc/summary.json ] && grep -q '"failed_gates": \[[^]]' data_synth/qc/summary.json; then
      echo "refusing: qc/summary.json still lists failed gates"; exit 1
    fi
    # `nohup run` cannot work: nohup execs a binary, not a shell function. Background the function itself and
    # wait on the pids, so this branch returns a real rc instead of claiming success when a worker died at once
    pids=()
    for c in $cells; do
      run "$c" "$c" < /dev/null &
      pids+=($!)
      echo "full $c pid $! -> data_synth/logs/$c.log"
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    echo "full done $(date +%H:%M), worker rc=$rc"
    bash local/run_synth_cells.sh status
    exit $rc ;;
  status)
    python -I - "$cells" <<'PY'
import json, sys
from pathlib import Path
plan = json.loads(Path('data_synth/plan.json').read_text())
for c in sys.argv[1].split():
    have = len(list(Path(f'data_synth/synth/{c}').glob('*.wav'))) if Path(f'data_synth/synth/{c}').is_dir() else 0
    fl = Path(f'data_synth/synth/{c}.failed.json')
    print(f'{c:12s} {have:6d}/{plan["cells"][c]["n"]:6d}  failed {len(json.loads(fl.read_text())) if fl.is_file() else 0}')
PY
    echo "-- procs: $(pgrep -fc "synth_[t]est.py" 2>/dev/null || echo 0)   disk: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4" free ("$5" used)"}')   gpu: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)" ;;
  *) echo "usage: $0 {smoke|pilot|full|status}"; exit 2 ;;
esac
