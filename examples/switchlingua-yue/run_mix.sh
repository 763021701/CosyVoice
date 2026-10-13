#!/bin/bash
# Mixed fine-tune: SwitchLingua code-switching (10.5h) + pathology dictation (1.4h train), LLM only.
# The question this run answers is whether the domain data can be added without losing what the pure
# SwitchLingua run measured, so the sealed SL test (data/test, 727 utt) stays exactly as it was and every
# new number is reported per source.
#
#   bash run_mix.sh 0 3   # prep + features + parquet + union lists
#   bash run_mix.sh 4 4   # train
#   bash run_mix.sh 5 8   # tier-1 selection, prompt pairs, synthesis, ASR scoring
#
# data_mix/{train,dev,test} hold ONLY the new domain (local/prepare_mix_data.py); the training list is the
# union with data/*/parquet/data.list, so SwitchLingua's parquet is reused as-is and never re-extracted.
set -euo pipefail
stage=${1:-4}
stop_stage=${2:-4}

. ./path.sh || exit 1

pretrained_model_dir=${PRETRAINED:-/root/autodl-tmp/.cache/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}
data_dir=$(pwd)/data
mix_dir=${MIX_DIR:-$(pwd)/data_mix}
lists=$mix_dir/lists
exp_dir=${EXP_DIR:-$(pwd)/exp/cosyvoice3_mix/llm/torch_ddp}
tb_dir=${TB_DIR:-$(pwd)/tensorboard/cosyvoice3_mix/llm/torch_ddp}
seed=1986

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: pathology sources -> Kaldi dirs, 8:1:1 disjoint by case, labels unified to '. Full stop.'"
  python local/prepare_mix_data.py --out $mix_dir --ratios 8,1,1 --seed $seed
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: campplus embedding + speech_tokenizer_v3 tokens for the new domain only"
  for x in train dev test; do
    python ../../tools/extract_embedding.py --dir $mix_dir/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx --num_thread 16
    python ../../tools/extract_speech_token.py --dir $mix_dir/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx --num_thread 8
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: parquet for the new domain"
  for x in train dev test; do
    python local/make_parquet.py --dir $mix_dir/$x --per_file 500
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: union lists -- train and cv both mix, but every eval list stays single-source"
  mkdir -p $lists
  cat $data_dir/train/parquet/data.list $mix_dir/train/parquet/data.list > $lists/train.list
  cat $data_dir/dev/parquet/data.list   $mix_dir/dev/parquet/data.list   > $lists/cv.list
  for x in dev test; do
    cp $data_dir/$x/parquet/data.list $lists/sl_$x.list
    cp $mix_dir/$x/parquet/data.list  $lists/new_$x.list
  done
  wc -l $lists/*.list
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: fine-tune the LLM from the base llm.pt, same config as the SwitchLingua-only run"
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  mkdir -p $exp_dir $tb_dir
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=1987 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1238" \
    local/cosyvoice_train.py \
      --train_engine torch_ddp \
      --config conf/cosyvoice3.yaml \
      --train_data $lists/train.list \
      --cv_data $lists/cv.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model llm \
      --checkpoint $pretrained_model_dir/llm.pt \
      --model_dir $exp_dir \
      --tensorboard_dir $tb_dir \
      --ddp.dist_backend nccl \
      --num_workers 2 \
      --prefetch 100 \
      --pin_memory \
      --use_amp
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: tier-1 per source -- SL dev/test for comparability, new dev/test for the domain"
  L="--qwen-pretrain-path $pretrained_model_dir/CosyVoice-BlankEN"
  for src in sl_dev new_dev new_test sl_test; do
    echo "-- $src"
    python local/eval_token.py --conf conf/cosyvoice3.yaml $L --data $lists/$src.list \
      --llm base=$pretrained_model_dir/llm.pt \
      $(for i in 0 1 2 3 4 5 6 7 8 9; do printf -- "--llm ep%s=%s/epoch_%s_whole.pt " $i $exp_dir $i; done) \
      > $exp_dir/tier1_$src.txt 2>&1
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: same-case voice prompts for the new test split (case = doctor = speaker here)"
  python local/build_prompt_pairs.py --data-root $mix_dir --split test
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  # Three arms on the new domain because its tier-1 minimum is ep0 while SL's is ep1 -- synthesis has to say
  # which one actually reads the dictation better. Two arms on the sealed SL test: old_ep1 is the previous
  # run's ep1, symlinked in rather than re-synthesized, so all three arms get scored on one intersection.
  ARMS_NEW=${ARMS_NEW:-"base=$pretrained_model_dir/llm.pt ep0=$exp_dir/epoch_0_whole.pt ep1=$exp_dir/epoch_1_whole.pt"}
  ARMS_SL=${ARMS_SL:-"base=$pretrained_model_dir/llm.pt ep1=$exp_dir/epoch_1_whole.pt"}
  mkdir -p $mix_dir/synth_sltest
  ln -sfn $data_dir/synth/ep1 $mix_dir/synth_sltest/old_ep1
  synth() {  # $1 = data root (text/wav.scp/prompt pairs), $2 = out root, $3 = space-separated name=path
    local args=()
    for a in $3; do args+=(--llm "$a"); done
    python local/synth_test.py --model-dir $pretrained_model_dir "${args[@]}" \
      --data-root $1 --split test --out-root $2
  }
  echo "-- synth new: $ARMS_NEW"
  synth $mix_dir $mix_dir/synth "$ARMS_NEW"
  echo "-- synth sl: $ARMS_SL"
  synth $data_dir $mix_dir/synth_sltest "$ARMS_SL"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Stage 8: ASR round-trip per domain"
  Q=/root/autodl-tmp/miniconda3/envs/qwen3_asr_ft/bin/python
  $Q local/asr_score.py --data-root $mix_dir  --out-root $mix_dir/synth       --arms base ep0 ep1
  $Q local/asr_score.py --data-root $data_dir --out-root $mix_dir/synth_sltest --arms base ep1 old_ep1
fi
