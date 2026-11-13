#!/bin/bash
# Continuation fine-tune of the CosyVoice3 LLM.
#
# Each row is (prompt utterance, target utterance) from the same speaker. The speech-token column is
# prompt tokens followed by target tokens, and the text is both transcripts. The trainer's loss covers
# the whole speech-token span, so the target tokens are predicted with that speaker's prompt tokens
# already in context -- the same order CosyVoice3LM.inference uses for inference_zero_shot.
#
# Requires the Kaldi dirs and tokens from run.sh stages 0-1. Only the LLM is trained. Stop after one
# or two epochs: a long run pulls the prior back toward the corpus mean and the prompt stops steering.
#
#   bash run_continuation.sh 0 0    # pack train/dev parquets
#   bash run_continuation.sh 1 1    # train
#
# Synthesize with inference_zero_shot. prompt_text is the fixed instruct prefix plus the prompt
# transcript, prompt_wav is that prompt utterance, tts_text is the new sentence. Swap prompt_wav
# across real speakers; do not use inference_instruct2 for this checkpoint.

set -euo pipefail
stage=${1:-0}
stop_stage=${2:-0}

. ./path.sh || exit 1

pretrained_model_dir=${PRETRAINED:-/root/autodl-tmp/.cache/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}
data_dir=${DATA_DIR:-$(pwd)/data}
out_dir=${OUT_DIR:-$(pwd)/data_continuation}
exp_dir=${EXP_DIR:-$(pwd)/exp/cosyvoice3_cont/llm/torch_ddp}
tb_dir=${TB_DIR:-$(pwd)/tensorboard/cosyvoice3_cont/llm/torch_ddp}
max_epoch=${MAX_EPOCH:-2}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: same-speaker continuation pairs"
  for x in train dev; do
    python local/make_continuation_parquet.py \
      --dir $data_dir/$x \
      --out $out_dir/$x/parquet \
      --pretrained $pretrained_model_dir
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: fine-tune the LLM on continuation pairs (max_epoch=$max_epoch)"
  mkdir -p "$exp_dir"
  conf=$exp_dir/cosyvoice3.yaml
  sed "s/^    max_epoch: .*/    max_epoch: $max_epoch/" conf/cosyvoice3.yaml > "$conf"
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=1987 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1238" \
    local/cosyvoice_train.py \
      --train_engine torch_ddp \
      --config "$conf" \
      --train_data $out_dir/train/parquet/data.list \
      --cv_data $out_dir/dev/parquet/data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model llm \
      --checkpoint $pretrained_model_dir/llm.pt \
      --model_dir "$exp_dir" \
      --tensorboard_dir "$tb_dir" \
      --ddp.dist_backend nccl \
      --num_workers 2 \
      --prefetch 100 \
      --pin_memory \
      --use_amp
fi
