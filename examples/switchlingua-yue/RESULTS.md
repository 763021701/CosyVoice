# SwitchLingua → CosyVoice3 LLM 微调：实验记录

研究对象：**离散 speech token 的生成质量**，用 ASR 训练可用性作为下游判据。flow / vocoder 已确认不是瓶颈（voice-clone 路径的训练效果与真实语音一致），因此三段实验都只训 LLM，flow/hift 全程固定。

日期 2026-10-10 ~ 2026-10-13。机器单卡（97G）。所有写入都在本项目内；`S2R_TC`、`dataset/` 只读。

---

## 0. TL;DR

1. **CosyVoice3 LLM 能学会英粤 code-switching**：10.5h SwitchLingua 微调一个 epoch，封存 test 的 ASR 回转错误率从 53.90% 降到 **18.70%**（真实音频 floor 22.19%，即比真人还低），top-1 token 命中 +6.65pp。
2. **加 1.27h 病理口述不伤害 code-switching**（18.05% / 18.74%，与纯 SL 的 18.73% 同档，SL 封存集完全同一批 727 条）。
3. **但病理域本身没学会，而且一度学崩了**：混合 ep1 在 26 条上产生 **13 条数字静音**，标点词 `. Full stop.` 的念出率从 59.3% 掉到 18.6%。
4. **根因在音频，不在标签写法**：MossFormer2 去噪切片里有大量多秒静音，speech tokenizer 忠实地把静音编成近恒定的低位 token（id 243/270），模型学会了「吐一长串静音 token」。同一批切分在**未增强原声**上主导 token 占比从 0.716/0.585/0.548 降到 0.171/0.082/0.236 —— 退化是增强引入的。
5. **清洗后崩溃完全消失**（0 静音）、punct 念出率回到 42.4%（清洗前 18.6%），两个独立解码器同向。新域仍不如 base（49.7% vs 32.1%），且这次同时改了「去静音」和「数据量减半」，**两件事尚未分离**。
6. 方法学两条硬教训：**混合训练的 CV 曲线必须分源看**（聚合曲线选 ep1，新域最优点是 ep0）；**teacher-forced CE 在这个任务上是可失效的指标**（三次与合成结果解耦）。

---

## 1. 三段实验的设置

| | 数据 | 训练集 | 起点 | epoch | 权重目录 |
|---|---|---|---|---|---|
| **R1 纯 SL** | SwitchLingua Cantonese 10.5h | 5767 utt | 基座 `llm.pt` | 10 | `exp/cosyvoice3` |
| **R2 混合(脏)** | + 病理口述 1.27h | 5767 + 431 | 基座 `llm.pt` | 10 | `exp/cosyvoice3_mix` |
| **R3 混合(清洗)** | + 病理口述 0.63h | 5767 + 227 | 基座 `llm.pt` | 10 | `exp/cosyvoice3_mixclean` |

三段都用 `conf/cosyvoice3.yaml`（只改 `max_epoch: 200→10`，注释写明原因），只训 `--model llm`，lr 1e-5、grad accumulate ×2、`--use_amp`。R2/R3 与 R1 用**同一份 SL parquet**（并集在 lists 层实现，SL 侧不重新抽特征）。

数据源：
- `dataset/SwitchLingua_audio/Cantonese.csv` + `Cantonese_wav16k`（英粤 code-switch，16k，无 speaker id）
- `Test_Samples/data/processed/segmented/from_denoised`（501 段，人工 `[CUT:]` 切分，44.1k MossFormer2 SE）
- `Test_Samples/samples/pathology_0821/output_v6_asr_sync/metadata.jsonl`（188 段，人工复核，16k）

划分：全部按 **item / case 互斥** 8:1:1（`utt/text/case` 三对泄漏计数均为 0）。病理的 case = 一份录音 = 一个医生，所以 case 互斥≈医生互斥。SL 26 条重复文本的 item 用 union-find 合并后再分。

刻意排除的：`from_lid_denoised` 与 `from_denoised` 是同一批音频的两种切法（per-case 小时数与词数完全相同），LID 切法还会产出 2.1s 只剩 `"一三四八八"` 的碎片；只取人工切法。

---

## 2. 目录说明

每个 `data*` 内部都是同样的四层：

| 层 | 文件 | 谁写 / 谁读 |
|---|---|---|
| Kaldi | `wav.scp text utt2spk spk2utt instruct` | `prepare_data.py`/`prepare_mix_data.py`/`clean_pathology.py` 写；抽特征、prompt 配对、tier-2 读 |
| 特征 | `utt2embedding.pt spk2embedding.pt utt2speech_token.pt` | `tools/extract_embedding.py`、`tools/extract_speech_token.py`（ONNX，本机 CPU） |
| parquet | `parquet/parquet_*.tar` + `data.list` | `local/make_parquet.py` 写；**训练只读这个**（音频字节已嵌入） |
| lists | `lists/{train,cv,sl_dev,new_dev,new_test,sl_test}.list` | stage 3 生成并集；`train.py` 与 `eval_token.py` 读 |

`.tar` 是上游命名遗留，内容实为 Apache Parquet。

| 目录 | 大小 | 内容 |
|---|---|---|
| `data/` | 3.7G | SL 独有：5767/778/727、`split_report.json`、R1 的 `synth/{base,ep1,ep9}`、`ft_model/`（即用模型目录）、`adhoc/`（gallbladder + 标点探针） |
| `data_mix/` | 932M | 脏病理半区 431/71/57、`label_audit.jsonl`（159 条改写）、`lists/`、R2 的 `synth/`、`synth_sltest/`、`synth_nofail/`（软链，实际 0 字节）、`synth_seed2/`（seed 对照） |
| `data_mix_clean/` | 560M | 清洗后 227/51/33，`wav.scp` 指向**本项目内的裁剪副本**，`clean_report.json` + 每 split 的 `silence_stats.jsonl`，R3 输出 |
| `data_mix_xp/` | 15M | 单用途对照：清洗模型 + **未裁剪带长停顿的 prompt** |

软链关系（删错会静默打断对照臂）：

```
data_mix/synth_sltest/old_ep1       → data/synth/ep1        ← 两个对照臂都指它
data_mix_clean/synth_sltest/old_ep1 → data/synth/ep1
data_mix_clean/synth_sltest/base    → data_mix/synth_sltest/base   (SL 侧未变，复用不重算)
data_mix_xp/{test,train}            → data_mix_clean/{test,train}
data/ft_model/llm.pt  ⇄  data/ep1_llm_stripped.pt   (同 inode，links=2，单删一个释放 0)
```

清洗版 `synth_sltest/base` 是软链：SL 的 prompt、模型、seed 都没变，重合成 727 条是浪费 —— 但因此**删 `data_mix/synth_sltest/base` 会让 R3 的 SL 对照失效**。

---

## 3. 文本标签口径（病理口述特有）

医生会把标点**念出来**，而且不是每句都念。三处证据：

- 语料惯例本身就是「词 + 符号」：`", full stop."`（旧 TXT）/ `". Full stop."`（新 metadata、cut_tags），99/99 与 93/93 次都是这个形态；`comma` 写作 `, comma,`。
- `end_boundary` 实测停顿：句尾 `. Full stop.` 停顿中位 **0.57s**，裸 `.` **0.48s**，`;` 0.60s → **音频里是「词 + 停顿」两件事**，所以 `full stop.`（词+句号）比 `full stop` 或 `.` 更忠实。
- Qwen3-ASR 转真实录音时把 `Full stop` 听成 `footstop / free stop / fuel stop`（都含 stop）→ 词确实发声。

本轮统一到 **`. Full stop.`** 后做的机械归一（逐条记进 `data_mix/label_audit.jsonl`）：

| 规则 | 处数 | 依据 |
|---|---|---|
| `m m` → `mm` | 255 | metadata 用 `mm`；真实音频 6 条里 5 条被 ASR 写成 `mm` |
| 裸 `period` → `. Period.` | 46 | **保留 "period" 这个词本身**（医生说的就是 period），只统一符号 |
| 裸 `comma` → `, comma,` | 40 | metadata 形态 |
| `dash`（21 处在 test）| 不动 | 它本来就是普通英文词 |

基座 tokenizer 侧的事实：`.` = 单 token **13**，`。` = **1773**（两个不同 token），`full`+`stop` = 8878/2936（句首）或 2480/2936（带前导空格）。`[breath]` 是单 token 151647，但本训练集里 0 次，属分布外，不建议用。

**残留缺陷**：`c m` 有 **150 处未归一**（train 131 / test 19，而 train 另有 33 处 `cm`）—— 同一单位两种写法。R2/R3 都是在含 `c m` 的数据上训的。

---

## 4. 代码/环境补丁（都在 `local/cosyvoice_train.py`，不改共享 `cosyvoice/`）

| 补丁 | 症状 | 性质 |
|---|---|---|
| `cosyvoice_join`（挂在 `cosyvoice.utils.executor` 上，不是 train_utils） | `AttributeError: 'ProcessGroup' object has no attribute 'options'` | torch 2.11 |
| `force_qwen2_fp32` | `mat1 and mat2 must have the same dtype, but got Float and BFloat16` | transformers 5 按 config dtype 加载 BlankEN |
| **`fix_incremental_decoding`** | 合成全是乱码/重复（`一一一…`、泰语、韩语） | transformers 5 不再从 past 长度推断位置 |
| `local/make_parquet.py` | 上游 `tools/make_parquet_list.py:54` 用了从未定义的 `spk_list`，每个 worker NameError | 上游 bug |

第三个是关键：`Qwen2Encoder.forward_one_step` 传 `masks[:, -1, :]` 且不传 `cache_position`。增量 vs 全序列最后隐状态最大差 **45.07**，补上 `cache_position` + 全长 mask 后 **7.0e-04**。teacher-forced loss 完全看不见这个 bug（`forward()` 整序列一次过），所以**微调看起来健康、音频全废**。同期还发现 `inference_bistream` 的 `cache[0][0].size(2)` 在 transformers 5 下抛 `TypeError: 'DynamicCache' object is not subscriptable`。

定位路径值得记：VC 路径（真实 token、绕过 LLM）→ 完美粤语音频 ⇒ flow/hift 无辜；采样器全扫一遍（RAS off、tau_r 0.9、紧 top-p）⇒ 不是采样；同前缀整段重推 ⇒ 逐字正确 ⇒ 问题在增量解码。

---

## 5. 评测设计与量具局限

**两层评测**
- tier-1：`eval_token.py` teacher-forced CE + top-1，同一批 batch 比较多个 `llm.pt`。两个实现细节：`model.mix_ratio=[5,10**6]` 让 `prepare_lm_input_target` 的双流随机门（llm.py:323）永不触发 ⇒ 确定性单流；`torch.autocast('cuda', bf16)` 与 `train_utils.batch_forward` 一致。按 `speech_token_len` 加权。**分源跑四个列表**（`sl_dev/new_dev/new_test/sl_test`）。
- tier-2：`synth_test.py`（同一 CosyVoice3 实例热换 `llm`，flow/hift 逐字节相同，`text_frontend=False`，seed 1986）→ `asr_score.py` 回转 + 按语言分桶的单位错误率 + `punct_metric.py`（标点词说/漏/造）+ 峰值法静音计数 + `token_distance.py`。

**量具的边界（都实测过，报告时必须带）**

| 指标 | 在本域可用性 | 数字 |
|---|---|---|
| ASR 回转错误率 | **SL 域可用**（floor 22.19%，有分辨力）；**病理域失效**（floor 86.08%，清洗后 96.34%）：`Foveolar hyperplasia is noted` → `除非呢个place是不footstop` | 因此病理域只能做**跨臂比较**，不能读「比真人好」 |
| CE / top-1 | 与合成**三次解耦**：R1 ep9 更好 top-1 但更差音频；R2/R3 新域 CE 全部优于 base 而合成全部差于 base | 选点用 CE，验收必须 tier-2 |
| token 空间距离 | 动态范围不足：同医生不同句 baseline 1.476，三臂 1.375–1.383 | 不作为主指标 |
| 标点词检出 | 真实音频自身只有 15–19% 被检出 → 上限受口音支配 | **两个解码器同向**才可采信（Qwen3-ASR + 粤英 `whisper-large-v3-turbo-cantonese-yue-english-ct2`） |
| 峰值/静音计数 | 客观、不依赖解码器 | `peak<0.01` 判静音 |

另外：`asr_score.py` 只在**各臂交集**上打分；`synth_nofail/` 是「排除静音 clip」的第二口径（软链，不占空间）。曾出现两版 ep0 错误率同为 46.62%，核对后确认是真巧合（33 条假设无一条相同、wav md5 不同）。

推理侧的坑（读代码 + 本机实测）：训练侧**零归一**（`processor.tokenize` 只 encode，实测 6272 条逐字节一致），推理侧 `text_normalize` 会做替换与断句；且本机 `cosyvoice` env **没装 wetext/ttsfrd**，所以数字不会展开，但中文分支仍会把全局 `.` 换成 `。`（token 13→1773）、删掉中英交界空格、并按 80 字符断句。`<| |>` 存在时前端自动跳过 —— 这就是 CosyVoice3 的 prompt_text（含 `<|endofprompt|>`）与 tts_text 走两套处理的原因。本实验一律 `text_frontend=False`。

---

## 6. 结果

### 6.1 R1 纯 SwitchLingua（封存 test 727 utt / 98489 token）

| | CE | top-1 | ASR overall | en | ins/ref | 时长比 | >2× 条数 |
|---|---|---|---|---|---|---|---|
| 真实音频 | — | — | 22.19% | 17.15% | 1.31% | 1.00 | — |
| base | 4.1619 | 16.71% | **53.90%** | 31.47% | 11.89% | 1.51 | 136 |
| **ep1** | **3.2839** | **23.36%** | **18.70%** | 13.61% | 0.92% | 0.94 | **0** |
| ep9 | 7.8921 | 17.94% | 22.95% | 16.76% | — | 0.92 | 2（+4 条零 token） |

聚合 CV：ep0 3.3197 → **ep1 3.2594** → ep2 3.2866 → ep3 3.4011 → ep4 3.7345 → ep5 4.4154 → ep6 5.3224 → ep7 6.3994 → ep8 7.0812 → ep9 7.7415（train acc 0.81 @ep9）。

普通话漂移（160 条抽样，ASR `language` 字段）：真实 147 粤 / 0 中；**base 120 粤 / 26 中（16%）**；ep1 129/0；ep9 129/0。

> 曾有一次口头报告把这些数字当作「已测」，实际三臂 ASR 当时并未运行；已当面撤回并重测。此后所有数都必须来自工具输出。

### 6.2 R2 混合（脏音频）

分源 tier-1（base → ep0 / ep1）：

| | base | ep0 | ep1 |
|---|---|---|---|
| sl_test CE | 4.1619 | 3.3704 | **3.2834** |
| sl_test top-1 | 16.71% | 22.56% | **23.47%** |
| new_test CE | 4.5505 | **4.2566** | 4.3301 |
| new_dev CE | 5.0013 | **4.5764** | 4.6179 |

聚合 CV 最低在 **ep1**（3.3779），但新域最低在 **ep0** —— 混合 cv.list 会把选点带偏。

新域 tier-2（57 条，两个留出医生）：

| 臂 | ASR(排除静音/全量) | punct strict→ | 静音 | 时长比 |
|---|---|---|---|---|
| base | 32.33% / 34.27% | 57.5% | 0 | 1.11 |
| R1 的 ep1（对照） | 41.09% / 41.94% | 60.3% | 0 | 0.91 |
| R2 ep0 | 44.81% / 46.96% | 26.0% | 0 | 1.06 |
| R2 ep1 | 63.44% / 69.16% | 19.2% | **13** | 1.26 |

崩溃刻画：
- 13 条静音（`peak 0.000`，8–14s）**全部在 BSample1 这一个 case**；用了 13 个不同 prompt、prompt 音频均正常 ⇒ 与 prompt 配对无关。
- **seed 7 重跑同 26 条：仍 13 条静音，但只有 9 条重合** ⇒ 约 50% 的随机失败率，不是特定文本有毒。
- LLM 吐的是近恒定低位 token（`243`×194、`270`×89）；另一型是 `2430`×244/280（峰值 0.99 的单调噪声）。
- 时长口径不支持「把词换成停顿」：以 `. Full stop.` 结尾的 18 条反而更短（1.048 vs 其余 1.22/1.401）。

SL 侧（同一封存集）：R2 ep1 **18.05%**（en 13.79，zh 17.99，ins 0.92，dur 0.91）vs R1 ep1 18.73% vs base 53.88% ⇒ **加域数据无损，甚至略好**。

**根因链（这是本实验最有价值的部分）**

1. R2 训练音频的主导 token 占比：中位 0.099，p90 0.216，**max 0.716**，`>0.35` 有 6 条；而 SL train 中位 0.067、max 0.350、0 条超阈。
2. 三条最差的：`B13290_seg_0013`（71.6% 全 id 243）、`B13488_seg_0022`（58.5% 2188）、`B13290_seg_0012`（54.8% 2188）。
3. **同一批切分换成未增强原声**：0.716→**0.171**、0.585→**0.082**、0.548→**0.236**；28 条抽样均值 0.165→0.097，超阈 3→0 ⇒ 退化由 MossFormer2 SE 在稀疏/安静段把音频压成近恒定序列造成。
4. 逐条核查这三条的静音结构：`seg_0012` 15.39s 里 **12.86s 是开头静音**，`seg_0013` 8.85s 里 **6.61s 是开头静音**，`seg_0022` 内部停顿 3.98s ⇒ 被当成样本的长静音就是那个吸引子。

### 6.3 R3 混合（清洗后）

清洗前提量化（同一能量检测器，阈值 = 该文件 95 分位帧能 − 25dB）：

| | 内部最长停顿 p50/p75/p90 | ≥1s | ≥2s |
|---|---|---|---|
| pathology 559 条 | 0.89 / 1.67 / 2.61 | **44.4%** | 18.4% |
| switchlingua 400 条抽样 | 0.30 / 0.43 / 0.68 | **1.8%** | 0.0% |

规则：内部 ≥1.0s 丢弃、首尾裁到 0.10/0.30s、音频复制进 `data_mix_clean/`（原件不动）。存活 **train 227/431、dev 51/71、test 33/57**，每个 case 都还有样本。清洗后训练音频主导 token：中位 **0.039**、max **0.208**、`>0.35` **0 条**。

结果（同一 33 条文本配对，两个解码器）：

| reader | 臂 | punct 前 → 后 | err 前 → 后 |
|---|---|---|---|
| qwen3-asr | base | 59.3% → 52.5% | 32.4% → 32.1% |
| qwen3-asr | ep0 | 27.1% → **49.2%** | 46.6% → 46.6% |
| qwen3-asr | ep1 | 18.6% → **42.4%** | 65.1% → **49.7%** |
| whisper-yue | base | 55.9% → 59.3% | 44.8% → 45.8% |
| whisper-yue | ep0 | 23.7% → 32.2% | 56.2% → 60.6% |
| whisper-yue | ep1 | 23.7% → 32.2% | 76.8% → 61.4% |

- **稳定性完全恢复**：三个臂 `0 静音 / 0 quiet`（R2 在同一批文本上是 4 条，全集 13 条）。
- ep0 **两版都 0 静音**，punct 却从 27.1→49.2 ⇒ 提升不是「去掉静音条」的算术效应，而是真的把词念出来了。
- 新域仍不如 base（49.7% vs 32.1%）。清洗后 test 上 base CE 5.0873 > ep0 4.8029，ASR 却给 base 高分 ⇒ CE 在这个域是**反向**指标（套话+停顿在 teacher-forced 下易预测）。
- SL 侧仍无损：R3 ep1 **18.74%**（en 13.04，zh 19.13，ins 0.86，dur 0.90）≈ R1 18.73%；sl_test CE 3.2868 / top-1 23.37%。
- R3 聚合 CV：3.4222 / **3.3670** / 3.3938 / 3.5191 / 3.8550 / 4.4522 / 5.3123 / 6.3741 / 7.3106 / 7.9669（10 epoch，18 分钟）。

**交叉 prompt 实验（部署相关）**：清洗后的 ep1 用**未裁剪、含多秒停顿**的真实 prompt 合成 33 条 —— **0 静音、0 quiet、时长 0.960、ASR 47.32%、punct 49.2%**（比用裁剪 prompt 的 42.4% 还高；base 同向 52.5% vs 59.3%）。

> 结论：**训练数据必须干净，但推理 prompt 保留真实停顿更好** —— 长停顿是「口述风格」的线索之一，裁掉会丢掉模仿信号。

---

## 7. 尚未定论 / 下一步

**必须分离的混淆**：R3 同时改了「去掉长静音」和「域数据 1.27h→0.63h（份额 11%→6%）」。现有证据支持「去静音消除了崩溃并恢复 punct」，**不支持**「同量下更好」。候选做法（按性价比）：

1. 清洗后数据上采样 2×（份额回 ~11%），一次训练即可分离两因素；
2. 在长停顿处**把一条切成两条**而不是丢弃（保量又去静音，需要词级时间戳来切文本，成本高）；
3. 病理模型**单独从 base 训**、不与 SL 混 —— 目前所有证据都指向 base 在该域最强（32.1% vs 49.7%）；
4. 补 `c m`→`cm` 归一（150 处）；
5. 若要试 `inference_sft`，需先 `add_zero_shot_spk` + `save_spkinfo` 造 `spk2info.pt`（基座模型目录里没有）。

**已知未修**：
- `run.sh` stage 6 与 `run_mix.sh` stage 5 硬编码 `ep0..ep9`，清理权重后重跑会 FileNotFound（改成从磁盘 glob 生成列表即可）。
- 病理域 `>10s` 占 51%（SL 1.5%），而 `max_frames_in_batch: 2000` 是 50fps mel 帧 = 每 batch 约 40s 音频 ⇒ 24.7% 的病理条只能凑成 batch≤2、21 条 >20s 只能 batch=1，SL 条则 8 条/batch。两类数据的每步样本数差 4–8 倍，这件事本身还没被当作变量测过。
- prompt 上限 30s（`frontend.py:97` assert），`metadata` 里原有 5 条 30–34.9s 已被丢弃。

**存储现状**：`exp` 54G → 9.5G（每运行保留 ep0/ep1 + 全部 30 个 `.yaml` sidecar，逐 epoch CV 指标无损；8 个 `tier1_*.txt`、4 个 `asr_results.json`、`token_distance.json`、`clean_report.json`、全部音频证据都在）。磁盘 100% 满主要是**外部**任务造成的：`/root/autodl-tmp/ttsds_cache` 178G 且 3 个 `b26_run_ttsds.py` 进程在写（单文件 2.2–2.5G），不在本项目控制内。

---

## 8. 复现

```bash
cd examples/switchlingua-yue && . ./path.sh

# R1 纯 SwitchLingua
bash run.sh 0 2          # 划分 + 抽特征 + parquet
bash run.sh 5 5          # 训 LLM（local/cosyvoice_train.py 内含三个兼容补丁）
bash run.sh 6 6          # tier-1 选点
bash run.sh 7 8          # tier-2 合成 + ASR 回转

# R2 / R3 混合：清洗前后只差 MIX_DIR / EXP_DIR
python local/prepare_mix_data.py --out data_mix --ratios 8,1,1 --seed 1986
python local/clean_pathology.py --src data_mix --out data_mix_clean   # 加 --dry-run 只测不写

MIX_DIR=$PWD/data_mix       EXP_DIR=$PWD/exp/cosyvoice3_mix/llm/torch_ddp \
  bash run_mix.sh 1 8
MIX_DIR=$PWD/data_mix_clean EXP_DIR=$PWD/exp/cosyvoice3_mixclean/llm/torch_ddp \
  bash run_mix.sh 1 8

# 标点词的第二解码器（须在 whisper_ft env；粤英微调模型）
/root/autodl-tmp/miniconda3/envs/whisper_ft/bin/python -I local/fw_hyp.py \
  --data-root data_mix_clean --out-root data_mix_clean/synth --arms base ep0 ep1
python -I local/punct_metric.py --asr data_mix_clean/synth/asr_fw.json \
  --data-root data_mix_clean --split test
```

脚本清单：`local/{prepare_data,prepare_mix_data,clean_pathology,make_parquet,cosyvoice_train,eval_token,build_prompt_pairs,synth_test,asr_score,punct_metric,token_distance,fw_hyp}.py`；`run.sh`（R1）、`run_mix.sh`（R2/R3，参数化）。`exp/`、`data*/`、`*.pt` 均在 `.gitignore` 内，不会污染提交。
