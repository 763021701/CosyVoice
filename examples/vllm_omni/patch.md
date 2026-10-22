# CosyVoice3 vLLM-Omni 补丁

针对已安装的 vLLM 0.28.0 + vLLM-Omni 0.28.0。重装 `vllm-omni` 会盖掉下面除启动脚本以外的全部改动。采样（RAS、`top_p`、`win_size`）不在这份补丁里。

包根目录：

```
/root/autodl-tmp/miniconda3/envs/cosyvoice_omni/lib/python3.12/site-packages/vllm_omni/
```

下面的路径都相对这个目录。改完要重启服务。fp32 ONNX 会编出新的 plan，不要继续用旧的 weakly-typed FP16 plan（本机曾是 `~/.cache/vllm_omni/cosyvoice3_trt/flow_estimator_f459dfad49f48d77.plan`）。

## 问题 1：并发时音色串了

`speech_token` / `speech_feat` / `embedding` 只在新请求的 prefill 上出现，而且只覆盖 `scheduled_new_reqs`，不是整个 batch。原逻辑把这条短列表按 batch 下标去取，下标超出时拿 `element[0]`。decode 行因此拿到了同一步里另一个请求的 prompt。日志症状是 `Error concatenating tensor for key embed.speech_token; keeping last tensor`（`outputs/mm_outputs.py`）。

### 1. `utils/mm_outputs.py`：`to_payload_element`

短列表不是广播。`idx` 超出列表、或该位置是 `None` 时返回 `None`。dict 里值为 `None` 的子键丢掉；子键全部丢掉后整个 dict 返回 `None`。`flatten_payload` 会保留值为 `None` 的键，所以空值必须在这里丢掉，否则后面的累积器会把真正的 prompt 盖成 `None`。

列表分支改成：

```python
elif isinstance(element, list):
    # For lists, clone tensors to avoid cross-request aliasing.
    # A short list is not a broadcast: borrowing element[0] hands one
    # request's prompt conditioning to every other request in the step.
    if pass_lists_through:
        return [elem.clone() if isinstance(elem, torch.Tensor) else elem for elem in element]
    if idx >= len(element):
        return None
    element = element[idx]
    if element is None:
        return None
    if isinstance(element, torch.Tensor):
        element = element.clone()
    return element
```

dict 分支在递归之后只保留非 `None` 的子键，空 dict 返回 `None`：

```python
elif isinstance(element, dict):
    out = {}
    for sk, sv in element.items():
        got = to_payload_element(
            sv,
            idx,
            start,
            end,
            pass_lists_through=pass_lists_through,
            seq_len=seq_len,
            scheduled_seq_len=scheduled_seq_len,
        )
        if got is not None:
            out[sk] = got
    return out or None
```

只改这里不够。短列表里靠前的 decode 行下标仍然小于列表长度，还会拿到别的请求的 prompt。必须做第 2 条，把列表拉成和 batch 一样长。

### 2. `model_executor/models/cosyvoice3/cosyvoice3.py`：按请求 id 对齐

在 `_split_prompt_conditioning` 之后、`forward` 之前，给 `CosyVoice3Model` 加上这两个方法。`prepare_runner_inputs` 每步都会被调用，包括 CUDA graph replay 的步；prompt 张量本身只在 `forward` 里拿得到。

```python
def prepare_runner_inputs(
    self,
    *,
    req_ids: list[str],
    num_computed_tokens,
    num_scheduled_tokens,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    new_req_ids: list[str] | None = None,
    **_: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remember which batch rows own this step's prompt-conditioning rows.

    ``speech_token`` covers ``scheduled_new_reqs`` only, in that order.
    ``new_req_ids`` is that list. Without it, new rows are the ones whose
    ``num_computed_tokens`` is 0, which matches a stable prefill reorder
    when every new request is a first chunk.
    """
    del num_scheduled_tokens
    self._prompt_cond_batch_size = len(req_ids)
    self._prompt_cond_drop = False
    if new_req_ids is not None:
        index = {str(rid): i for i, rid in enumerate(req_ids)}
        slots: list[int] = []
        for rid in new_req_ids:
            row = index.get(str(rid))
            if row is None:
                self._prompt_cond_drop = True
                slots = []
                break
            slots.append(row)
        self._prompt_cond_row_ids = slots
    else:
        computed = [int(value) for value in num_computed_tokens]
        self._prompt_cond_row_ids = [i for i, ntok in enumerate(computed) if ntok == 0]
    return input_ids, positions

def _align_prompt_rows(self, rows: list | None) -> list | None:
    """Place prefill-only rows onto the full scheduled batch.

    Decode rows get ``None`` so the payload splitter drops the key
    instead of copying another request's prompt.
    """
    if rows is None:
        return None
    n = getattr(self, "_prompt_cond_batch_size", None)
    slots = getattr(self, "_prompt_cond_row_ids", None)
    if getattr(self, "_prompt_cond_drop", False) and isinstance(n, int):
        logger.warning(
            "CosyVoice3 prompt conditioning request ids are not in this step's batch; dropping this step's prompt payload"
        )
        return [None] * n
    if not isinstance(n, int) or slots is None:
        return rows
    if len(rows) != len(slots):
        logger.warning(
            "CosyVoice3 prompt conditioning rows (%d) do not match new "
            "prefill rows (%d); dropping this step's prompt payload",
            len(rows),
            len(slots),
        )
        return [None] * n
    aligned: list = [None] * n
    for src, dst in zip(rows, slots):
        if 0 <= dst < n:
            aligned[dst] = src
    return aligned
```

talker 的 `forward` 里，`_split_prompt_conditioning(...)` 返回之后、`to_dict(OmniPayloadStruct(...))` 之前加上：

```python
speech_token_list = self._align_prompt_rows(speech_token_list)
speech_feat_list = self._align_prompt_rows(speech_feat_list)
embedding_list = self._align_prompt_rows(embedding_list)
speech_token_len_list = self._align_prompt_rows(speech_token_len_list)
```

长度对不上时整步丢掉 prompt，不要按位置硬套。新请求自己的 prefill 是它第一次写入 prompt 的那一步，所以丢掉的是这一步里对不上的行，不会把已经写对的旧请求改掉。

### 3. `worker/gpu_ar_model_runner.py`：把新请求 id 传进去

`execute_model` 里调用 `prepare_runner_inputs(...)` 时多一个参数。NPU runner 没改；没有这个参数时退回到 `num_computed_tokens == 0` 的行。

```python
new_req_ids=[req.req_id for req in scheduler_output.scheduled_new_reqs],
```

### 4. prompt 只保留第一次写入

`model_executor/stage_input_processors/cosyvoice3.py` 里原来的 `_FULL_PAYLOAD_REPLACE_KEYS` 只有 `embed.speech_token`、`embed.speech_feat`、`embed.embedding`。`embed.speech_token_len` 是 `[1, 1]`，不放进这个集合就会被按时间维拼起来。改成：

```python
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset(
    {"embed.speech_token", "embed.speech_feat", "embed.embedding", "embed.speech_token_len"}
)
# Prompt conditioning is fixed at the request's own prefill. A later step
# must not replace it: that is how a newer request's voice overwrites an
# older one.
_FULL_PAYLOAD_SET_ONCE_KEYS: frozenset[str] = _FULL_PAYLOAD_REPLACE_KEYS
```

`worker/omni_connector_model_runner_mixin.py`：

- 在 `_resolve_full_payload_replace_keys` 后面加 `_resolve_full_payload_set_once_keys`。它和 replace 的查找方式一样，读 stage input 模块上的 `_FULL_PAYLOAD_SET_ONCE_KEYS`。模块没有这个名字时返回空集。
- `accumulate_full_payload_output` 开头取出 `set_once_keys`。
- 循环里，`v is None` 之后、replace 之前：键已经在 `chunks` 或 `latest` 里就 `continue`。
- 原来的 `if k in replace_keys` 改成 `if k in replace_keys or k in set_once_keys`，这样第一次写入仍然按 replace 存成一份，而不是拼接到一起。

不要把所有 replace 键都改成 set-once。别的模型用 replace 表示「截至目前的完整结果」，例如 `model_outputs`，那些键还要允许后续覆盖。

## 问题 2：波形打满

两处叠在一起：TensorRT 10.16 对 fp32 ONNX 编出 weakly-typed FP16 引擎；code2wav 的 `dtype=auto` 变成 bf16，flow 前端跟着变成 bf16。HiFT 虽然初始化时 `.float()`，`load_state_dict` 仍会把已有参数写回 bf16。现象是整段波形 RMS 和峰值都是 1.0。仓库里的 flow 是 fp32（`fp16=False`）。

talker 保持 bf16，不要改 stage 0。

### 5. `model_executor/models/cosyvoice3/flow_estimator_trt.py`

fp32 ONNX 只开 `BuilderFlag.TF32`。不要开 `BuilderFlag.FP16`：TensorRT 10 还有这个 flag，设上之后得到的是 weakly-typed fp16 引擎，这个 DiT 会溢出。带 `autocast` / `fp16` 的 ONNX 仍走 strongly-typed。

`_convert_onnx_to_trt` 里，非 strongly-typed 的分支只留：

```python
if not strongly_typed:
    # fp32 ONNX stays fp32. TF32 is the reduced-precision matmul flag.
    # Do not set BuilderFlag.FP16: on TensorRT 10 that flag exists and
    # builds a weakly-typed fp16 engine, which overflows this DiT.
    tf32 = getattr(trt.BuilderFlag, "TF32", None)
    if tf32 is not None:
        config.set_flag(tf32)
```

原代码会在 `("FP16", "TF32")` 里取第一个存在的 flag。TensorRT 10 两个都有，所以实际编出来的是 FP16。

`build_flow_estimator_trt` 把 variant 放进缓存 key，避免继续加载同一份 fp32 ONNX 编出来的旧 FP16 plan：

```python
plan_path = _resolve_plan_path(
    onnx_path,
    prefix="flow_estimator",
    variant="strongly-typed-fp16" if strongly_typed else "fp32tf32",
)
```

本机重编后的 plan 是 `~/.cache/vllm_omni/cosyvoice3_trt/flow_estimator_9c60b2ce01425730.plan`。hash 含设备名和 TensorRT 版本，换机器会变。

### 6. `model_executor/models/cosyvoice3/speaker_embedding_trt.py`：`_resolve_plan_path`

给函数加 `variant: str = ""`。key 拼完设备和 TensorRT 版本之后：

```python
if variant:
    key = f"{key}|{variant}"
```

空 variant 时 key 与原来相同，campplus 的旧 plan 还能用。variant 只追加一次，追加两次会变成 `fp32tf32|fp32tf32`，和已经编好的 flow plan 对不上。

### 7. `model_executor/models/cosyvoice3/cosyvoice3_code2wav.py`：`load_weights`

`load_state_dict` 之后强制 fp32。只在初始化时对 HiFT 调 `.float()` 不够。

```python
self.flow_model.load_state_dict(torch.load(flow_path, map_location=device), strict=True)
self.flow_model.to(device=device, dtype=torch.float32).eval()

self.hift.load_state_dict(hift_state_dict, strict=True)
self.hift.to(device=device, dtype=torch.float32).eval()
```

`_forward_mel` 用 flow 参数的 dtype 去 cast prompt mel 和 embedding。flow 是 fp32 之后，这两份输入也是 fp32。

### 8. `deploy/cosyvoice3.yaml`：stage 1

stage 0 不要加 dtype。stage 1 在 `gpu_memory_utilization` 和 `enforce_eager` 之间加上：

```yaml
# fp32 flow. dtype=auto becomes bf16 and saturates the waveform.
dtype: float32
```

启动日志里 stage 1 应是 `dtype=torch.float32`。stage 0 仍是 `dtype=torch.bfloat16`，那是 talker，保持不动。

### 9. 本仓库 `examples/vllm_omni/run_cosyvoice3_server.sh`

这份在仓库里，重装包不会盖掉。stage 1 的 override 带上 dtype。`--stage-overrides` 是按键合并，yaml 里的 `dtype: float32` 在没被 override 盖掉时也会留下；写进脚本是为了这个示例自己显式带上。

```bash
STAGE_OVERRIDES="$(printf '{"0":{"max_num_seqs":%s,"gpu_memory_utilization":0.4},"1":{"max_num_seqs":%s,"gpu_memory_utilization":0.2,"dtype":"float32"}}' \
    "${MAX_NUM_SEQS}" "${MAX_NUM_SEQS}")"
```

`COSYVOICE3_ESTIMATOR_ONNX` 继续指向模型目录里的 `flow.decoder.estimator.fp32.onnx`。不设的话，vLLM-Omni 会先找 fp16 ONNX。

## 怎么确认

服务起来后看三件事：

- 日志有 `Building flow-estimator TensorRT engine ... (fp32+TF32)`，并且 stage 1 是 `dtype=torch.float32`。
- 单句波形 RMS 在 0.1 附近，不是 1.0。2026-10-22 的复测：单句 RMS 0.075、峰值 0.72。
- 两个不同音色重叠：先发长句 A，约 1 秒后发短句 B。A 的说话人余弦贴近 A 的参考（复测 0.852），B 贴近 B（复测 0.917）。日志里不再出现 `keeping last tensor`。

参考之间的说话人余弦大约 0.09 时，这个对比才分得开。同一条参考音的 batch 看不出问题 1。
