# attnbench：gla / kda / gka / mesa_net 效率测评（H100）

用 [Undermyth/attnbench](https://github.com/Undermyth/attnbench) 这套框架，对
4 个注意力变体的**执行效率**（延迟 / 吞吐）做横向对比。测评内容对应 fla-org 的
PR [fla-org/flash-linear-attention#1352](https://github.com/fla-org/flash-linear-attention/pull/1352)
（“[GKA] Add Gated KalmaNet op, layer, and model”）。

## 测什么

| 变体 | 内核来源 | attnbench core |
|---|---|---|
| `gla` | fla `chunk_simple_gla` / `fused_recurrent_simple_gla` | `attnbench/cores/gla.py`（框架自带） |
| `kda` | fla `chunk_kda` / `fused_recurrent_kda` | `attnbench/cores/kda.py`（框架自带） |
| `gka` | fla `chunk_gka` / `fused_recurrent_gka`（PR #1352） | `attnbench/cores/gka.py`（**已改为 PR 版内核**） |
| `mesa_net` | fla `chunk_mesa_net` / `mesa_net_decoding_one_step` | `attnbench/cores/mesa_net.py`（**新增**） |

对每个 (变体 × batch × 序列长度 × prefill/decode) 组合测量：

- 单层 attention 延迟（in-model）、attention/(attention+mlp) 占比；
- 整模型 tokens/s（prefill 为 trunk 墙钟时间；decode 为完整生成步含 lm_head）；
- 可选的 core-only 微基准（chunk / recurrent 内核延迟）。

## 对本仓库相对上游 attnbench 的改动

1. **`attnbench/cores/gka.py` 重写**：上游 gka core 用的是项目早期的
   `fla.ops.gka.gka_chebyshev_solve`（`gka_chebyshev_gla` / `torch_decoding_one_step`）。
   现在换成 PR #1352 的正式内核 `fla.ops.gka.chunk_gka` / `fused_recurrent_gka`
   （接口 `g/gk/alpha/scale/ridge_ratio/num_iter/initial_state`）。
2. **新增 `attnbench/cores/mesa_net.py`**：按 fla-org 的 `fla/layers/mesa_net.py`
   对齐数学（q/k L2 归一化、`g=logsigmoid(a)`、`beta=sigmoid(b)`、逐 (head, key_dim)
   的 `lamb=softplus(lambda_params)+lower_bound`），接 `chunk_mesa_net` /
   `mesa_net_decoding_one_step`。
3. **新增 `configs/gla_kda_gka_mesa.yaml`**：4 变体、0.4B 骨架
   （dim=1024, n_heads=8, head_dim=128）、batch ∈ {1,4,8}、seq ∈ {512,2048,8192}、
   bf16、prefill+decode+microbench 全开。

## 服务器环境（重要）

框架依赖：`torch`(CUDA)、`flash-linear-attention`(fla)、`einops`、`jaxtyping`、
`rich`、`pyyaml`、`triton`。

**关键：fla 必须同时包含 `fla.ops.gka`（PR #1352 版）和 `fla.ops.mesa_net`。**
二者都包含在 PR 的分支里（该分支 = fla-org main + GKA），所以直接装它即可：

```bash
source /home/qpl/miniconda3/etc/profile.d/conda.sh
conda activate slai_eval            # 或新建一个专门环境，避免动到 CRIR 评测环境

pip install rich
pip install -e "git+https://github.com/elvisnunez7/flash-linear-attention@gka#egg=flash-linear-attention"
# 国内网络不行的话先 clone 再本地 install：
#   git clone -b gka https://github.com/elvisnunez7/flash-linear-attention && pip install -e ./flash-linear-attention
```

> 注意：PR 自述里提到 bf16/fp16 且 head dim K<64（或 V<32）时 `chunk_gka`
> 反向会报错（Triton 3.7.1 的 bug）；本测评是**前向延迟**，且 head_dim=128，
> 不受此限制。torch/triton 版本建议与 PR 测试环境接近（torch 2.12、triton 3.7）。

## 运行

```bash
cd /home/qpl/code/lmtemplate     # 或你放本仓库的位置
bash run_attnbench.sh
```

等价的手动步骤：

```bash
cd attnbench
python -m attnbench --list-variants                                    # 确认注册了 gla/kda/gka/mesa_net
python -m attnbench --config configs/gla_kda_gka_mesa.yaml --dry-run    # 打印解析后的计划
python -m attnbench --config configs/gla_kda_gka_mesa.yaml              # 正式跑
```

结果输出到 `attnbench/results/<timestamp>_h100/`：

- `results.txt`：终端表格（rich）
- `results.json`：全量细节（每次迭代的耗时数组 + 环境元信息）
- `results.csv`：长表（便于 pivot）

汇总对比（可选）：

```bash
python scripts/summarize.py results/<run_dir> --vs gla   # 各变体相对 gla 的倍率
```

## 结果格式说明（和 PR #1352 的差异）

PR #1352 里的表格是 **fla 自己的 kernel 基准**（`benchmarks/ops/`），列是
`fwd (ms) / fwd+bwd (ms) / working memory (GB) / decode (µs)`，而且 head 数用 H=32。

attnbench 测的是**整模型前向延迟/吞吐**（不含训练反向），列是
`attn_ms / attn_ratio / tps / chunk_ms / recurrent_ms`，head 数沿用我们 0.4B
模型的 H=8、head_dim=128。

两者度量不同：attnbench 更适合回答“整模型推理/训练前向有多快、attention 占多少”，
PR 的表更适合回答“这个 kernel 本身 fwd/bwd 各多少毫秒、吃多少显存”。
如果你需要**和 PR 完全一致的 fwd/fwd+bwd/显存表**，那要用 fla 的
`benchmarks/ops/` 脚本跑，而不是 attnbench；需要的话我再补一套。
