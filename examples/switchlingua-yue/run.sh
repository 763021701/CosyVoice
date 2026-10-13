#!/bin/bash
# Fine-tune the CosyVoice3 LLM on SwitchLingua Cantonese-English code-switching speech (10.5h).
# Run from this directory. Only `llm` is trained: flow/hift stay on the pretrained weights, which the
# R1/R2/R3 diagnostics in the sibling S2R_TC project showed are not where the ASR-training cost is.
#
#   bash run.sh 0 2     # split + extract features + parquet
#   bash run.sh 5 5     # train the LLM
#
# Unlike examples/libritts/cosyvoice3, there is no download stage (the dataset is already on disk), the
# paths are two levels up rather than three (this dir sits directly under examples/), and stage 2 uses
# local/make_parquet.py instead of tools/make_parquet_list.py, which dies on an undefined `spk_list`.

set -euo pipefail
stage=${1:-5}
stop_stage=${2:-5}

. ./path.sh || exit 1

CSV=${CSV:-/root/autodl-tmp/workspace/dataset/SwitchLingua_audio/Cantonese.csv}
WAV_DIR=${WAV_DIR:-/root/autodl-tmp/workspace/dataset/SwitchLingua_audio/Cantonese_wav16k}
pretrained_model_dir=${PRETRAINED:-/root/autodl-tmp/.cache/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}
data_dir=$(pwd)/data
seed=1986

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: item-disjoint 8:1:1 split -> Kaldi dirs"
  python local/prepare_data.py \
    --csv "$CSV" --wav-dir "$WAV_DIR" --out "$data_dir" --ratios 8,1,1 --seed $seed
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: campplus embedding + speech_tokenizer_v3 token, offline so training never tokenizes"
  for x in train dev test; do
    python ../../tools/extract_embedding.py --dir $data_dir/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx --num_thread 16
    python ../../tools/extract_speech_token.py --dir $data_dir/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx --num_thread 8
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: parquet"
  for x in train dev test; do
    python local/make_parquet.py --dir $data_dir/$x --per_file 1000
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: same-item voice prompts (the dataset has no speaker id)"
  python local/build_prompt_pairs.py --data-root $data_dir --split test
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: fine-tune the LLM. dev is the CV split; test is not read by training."
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1237" \
    local/cosyvoice_train.py \
      --train_engine torch_ddp \
      --config conf/cosyvoice3.yaml \
      --train_data $data_dir/train/parquet/data.list \
      --cv_data $data_dir/dev/parquet/data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model llm \
      --checkpoint $pretrained_model_dir/llm.pt \
      --model_dir $(pwd)/exp/cosyvoice3/llm/torch_ddp \
      --tensorboard_dir $(pwd)/tensorboard/cosyvoice3/llm/torch_ddp \
      --ddp.dist_backend nccl \
      --num_workers 2 \
      --prefetch 100 \
      --pin_memory \
      --use_amp
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: tier-1, teacher-forced CE / top-1 on dev for every epoch -- this is the checkpoint selector"
  EXP=$(pwd)/exp/cosyvoice3/llm/torch_ddp
  python local/eval_token.py --conf conf/cosyvoice3.yaml \
    --qwen-pretrain-path $pretrained_model_dir/CosyVoice-BlankEN \
    --data $data_dir/dev/parquet/data.list \
    --llm base=$pretrained_model_dir/llm.pt --llm rl=$pretrained_model_dir/llm.rl.pt \
    $(for i in 0 1 2 3 4 5 6 7 8 9; do printf -- "--llm ep%s=%s/epoch_%s_whole.pt " $i $EXP $i; done)
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage 7: tier-2 synthesis of the test split, one arm per llm checkpoint"
  EXP=$(pwd)/exp/cosyvoice3/llm/torch_ddp
  python local/synth_test.py --model-dir $pretrained_model_dir \
    --llm base=$pretrained_model_dir/llm.pt \
    --llm ep1=$EXP/epoch_1_whole.pt --llm ep9=$EXP/epoch_9_whole.pt \
    --data-root $data_dir --out-root $data_dir/synth
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  # qwen_asr's vendored modeling uses check_model_inputs(func=...) and will not import under this env's
  # transformers 5, so scoring runs in the env that has transformers 4.57.
  echo "Stage 8: tier-2 ASR round-trip + per-language metrics, with the real-audio floor"
  /root/autodl-tmp/miniconda3/envs/qwen3_asr_ft/bin/python local/asr_score.py \
    --data-root $data_dir --out-root $data_dir/synth --arms base ep1 ep9
fi
