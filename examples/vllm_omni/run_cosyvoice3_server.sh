#!/usr/bin/env bash
# 用 vLLM-Omni 启动 CosyVoice3 服务，默认按离线数据合成来配。
#
# 两阶段共用一张卡：stage 0（talker）出语音 token，stage 1（code2wav）做
# flow matching + HiFT。实测（TensorRT，batch 8，同一批短句）：
#   async_chunk 开着时整批 RTF 0.061
#   关掉 async_chunk 后整批 RTF 0.037
# 离线合成不需要边出 token 边出声音，默认 NO_ASYNC_CHUNK=1：每句 token
# 到齐后 flow 和 HiFT 只跑一次。在线流式设 NO_ASYNC_CHUNK=0。
# 两阶段的 max_num_seqs 默认都是 8。客户端一次 POST /v1/audio/speech/batch
# 放 8 条；再加大 batch，吞吐几乎不再涨。
#
# 依赖 vLLM 0.28 + vLLM-Omni 0.28 + torch 2.13 + TensorRT 10.16，
# 装在 cosyvoice_omni 环境里，不改正在用于训练和合成的 cosyvoice 环境
# （vLLM 0.21 / torch 2.11）。
#
# 用法：
#   bash examples/vllm_omni/run_cosyvoice3_server.sh
#   PORT=8091 MAX_NUM_SEQS=8 bash examples/vllm_omni/run_cosyvoice3_server.sh
#   NO_ASYNC_CHUNK=0 bash examples/vllm_omni/run_cosyvoice3_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_PREFIX="${COSYVOICE_OMNI_ENV:-/root/autodl-tmp/miniconda3/envs/cosyvoice_omni}"
VLLM="${ENV_PREFIX}/bin/vllm"
MODEL="${MODEL:-/root/autodl-tmp/.cache/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
PORT="${PORT:-8091}"
HOST="${HOST:-0.0.0.0}"
SERVED_NAME="${SERVED_NAME:-cosyvoice3}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

if [[ ! -x "${VLLM}" ]]; then
    echo "vllm not found at ${VLLM}. Create the cosyvoice_omni env first." >&2
    exit 1
fi
if ! "${ENV_PREFIX}/bin/python" -c "import s3tokenizer" >/dev/null 2>&1; then
    echo "s3tokenizer is required in ${ENV_PREFIX}. Install with: ${ENV_PREFIX}/bin/pip install s3tokenizer==0.3.0" >&2
    exit 1
fi
if ! "${ENV_PREFIX}/bin/python" -c "import tensorrt" >/dev/null 2>&1; then
    echo "tensorrt is required in ${ENV_PREFIX}. Install tensorrt 10.16 (CUDA 13) into that env." >&2
    exit 1
fi
# Blackwell SM 12.x 上 FlashInfer 采样器认不出算力，会在启动时退出。默认改用 PyTorch 采样器。
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
if [[ ! -d "${MODEL}" ]]; then
    echo "model dir not found: ${MODEL}" >&2
    exit 1
fi
# 用模型目录里的 fp32 ONNX 编 flow 引擎。TensorRT 10 会打开 FP16 builder flag。
# 不设的话，vLLM-Omni 会先尝试下载 fp16 ONNX。
if [[ -z "${COSYVOICE3_ESTIMATOR_ONNX:-}" && -f "${MODEL}/flow.decoder.estimator.fp32.onnx" ]]; then
    export COSYVOICE3_ESTIMATOR_ONNX="${MODEL}/flow.decoder.estimator.fp32.onnx"
fi

# 显存比例沿用官方 yaml：talker 0.4，code2wav 0.2。整段覆盖用 STAGE_OVERRIDES。
if [[ -z "${STAGE_OVERRIDES:-}" ]]; then
    STAGE_OVERRIDES="$(printf '{"0":{"max_num_seqs":%s,"gpu_memory_utilization":0.4},"1":{"max_num_seqs":%s,"gpu_memory_utilization":0.2}}' \
        "${MAX_NUM_SEQS}" "${MAX_NUM_SEQS}")"
fi

# 离线合成默认关掉 async_chunk。NO_ASYNC_CHUNK=0 恢复在线流式。
NO_ASYNC_CHUNK="${NO_ASYNC_CHUNK:-1}"
EXTRA=()
if [[ "${NO_ASYNC_CHUNK}" != "0" ]]; then
    EXTRA+=(--no-async-chunk)
fi

echo "Starting CosyVoice3 vLLM-Omni server"
echo "  model: ${MODEL}"
echo "  served name: ${SERVED_NAME}"
echo "  listen: ${HOST}:${PORT}"
echo "  max_num_seqs: ${MAX_NUM_SEQS}"
echo "  no_async_chunk: ${NO_ASYNC_CHUNK}"
echo "  estimator onnx: ${COSYVOICE3_ESTIMATOR_ONNX:-<auto>}"
echo "  stage overrides: ${STAGE_OVERRIDES}"

exec "${VLLM}" serve "${MODEL}" \
    --omni \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --served-model-name "${SERVED_NAME}" \
    --allowed-local-media-path "${ROOT}" \
    --stage-overrides "${STAGE_OVERRIDES}" \
    "${EXTRA[@]}"
