# LongBench 评测代码（Anti-Hebbian Attention / 线性注意力基线）

在 **SLAI 集群（H100）** 上，用 [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) 对
`lmtemplate` 训练的 **4 个模型族**跑 **LongBench**。

- 架构：`https://github.com/Undermyth/lmtemplate`（核心已拷到本目录 `module/`）
- 权重：服务器上 `checkpoints/` 下的 Lightning `.ckpt` 文件（不再从 HuggingFace 下载）

---

## 模型与权重

`checkpoints/` 下每个子目录是一个模型族，里面是不同训练 token 数（5bt ~ 38bt）的 checkpoint：

| 目录 | 模型 | 架构（`attn_impl`） | 来源分支 |
|------|------|--------------------|----------|
| `gdn-muon/` | Gated DeltaNet | `gdn` → `fla.layers.gated_deltanet.GatedDeltaNet` | `base/gdn` |
| `gla/` | Gated Linear Attention | `gla` → `fla.layers.simple_gla.SimpleGatedLinearAttention` | `base/gla` |
| `kda/` | Kalman Delta Attention | `kda` → `fla.layers.kda.KimiDeltaAttention` | `base/kda` |
| `gka/` | Gated KalmaNet | `gka` → `fla.layers.gka.GatedKalmaNet` | 本地 fla（Undermyth fork `fix/mesa-cg`） |

`gdn-muon` 的 `muon` 表示用 Muon 优化器训练，文件名里的 `0.4b` 表示约 0.4B 参数
（`dim=1024, n_layers=24`）；`bt` 是训练的十亿 token 数，`step` 是训练步数。

---

## 目录结构

```
.
├── module/                 # 架构代码（modeling.py 内含 attn_impl 开关）
│   ├── modeling.py         # ModelConfig / ModelForCausalLM / Layer（attn 按 attn_impl 切换）
│   ├── attention.py        # 模板自带的 FlashAttention 块（attn_impl="attention" 时用）
│   └── cache_utils.py
├── eval_longbench.py       # 主脚本：发现 checkpoints/ 下所有 .ckpt 并逐个跑 LongBench
├── make_results_table.py   # 汇总脚本：把各 checkpoint 的 LongBench 总分汇成对比表/CSV
├── tokenizer/              # 已下载到本地的 Llama-2 tokenizer（无需 HF 登录）
├── requirements.txt
├── setup_env.sh
├── run_longbench.slurm     # Slurm 作业（H100）
├── run_longbench.sh        # 交互式直接运行
└── README.md
```

### 对架构源码的改动

`lmtemplate` 的 4 个分支只差 `Layer` 里 attention 这一行。这里把 `module/modeling.py` 合并成一个文件，
用 `ModelConfig.attn_impl` 选择注意力核：

- `attention`（默认）= 模板自带的 FlashAttention 块；
- `gla` / `gdn` / `kda` = 对应 fla 层；
- `gka` = `fla.layers.gka.GatedKalmaNet`（`ridge_strength=0.01`）。

另外保留了两处让新版 lm-eval 能正确接管的修复：`tie_weights()` no-op + `tie_word_embeddings=False`。

---

## 一、环境准备

```bash
bash setup_env.sh                 # 建 conda 环境 slai_eval 并装依赖
conda activate slai_eval
```

> **不需要 HF 登录**：tokenizer 已下载到本地 `./tokenizer`；LongBench 数据集是公开的，
> 脚本里已设置 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像下载。

`flash-linear-attention`（`fla`）的版本会影响 fla 层的参数名/接口，**必须与训练 checkpoint 时使用的版本兼容**。
如遇权重 key 对不上，先对齐 fla 版本（默认装 `>=0.4.1`，可尝试 `==0.4.2`）。

---

## 二、运行

### Slurm 提交（推荐）

改 `run_longbench.slurm` 里的 `--partition` / `--gres` / 环境名，然后：

```bash
sbatch run_longbench.slurm
```

### 交互式

```bash
bash run_longbench.sh
```

### 直接用 `eval_longbench.py`

```bash
# 全部 checkpoint（自动发现 ./checkpoints 下所有 *.ckpt）
python eval_longbench.py --checkpoint-dir ./checkpoints --tasks longbench

# 只跑某个模型族
python eval_longbench.py --variant gla

# 只跑单个 checkpoint
python eval_longbench.py --checkpoint ./checkpoints/gla/gla-0.4b-muon-5bt-step10240.ckpt --variant gla

# 冒烟测试（每个任务只跑 5 条）
python eval_longbench.py --limit 5
```

结果写入 `results/<目录名>/<checkpoint名>/results.json`，并在 `results/summary.json` 汇总各 checkpoint 的
`longbench` 总分。

---

## 三、任务选择

- `longbench`：全部 21 个子任务（默认，最后给总 `score`）
- 子组：`longbench_single` / `longbench_multi` / `longbench_summarization` / `longbench_fewshot` / `longbench_synthetic` / `longbench_code`
- 单任务：`longbench_narrativeqa` 等；LongBench-E 用 `longbench_e`

---

## 四、结果解读

每个 checkpoint 的完整结果在 `results/<族>/<checkpoint>/results.json`，汇总在 `results/summary.json`。

- 每个任务一行，`score` 是该任务主指标（F1 / ROUGE / 检索准确率 / 代码相似度等）；
- `longbench` 组的总分是 21 个子任务分数的**平均**。

**注意刻度**：lm-eval 里这些 `score` 是 **0–1 小数**，官方 LongBench / 论文里通常是 **×100 的 0–100 整数**。
即 `score=0.45` 等价于官方口径的 `45`；引用或画图时记得 ×100。

跑完后用汇总脚本把所有 checkpoint 的总分汇成对比表：

```bash
python make_results_table.py
```

它会读 `results/*/*/results.json`，按「模型族 × 训练 token 数」列出每个 checkpoint 的 `longbench` 总分（已 ×100），
并生成 `results/longbench_table.csv`，方便画 gdn/gka/gla/kda 随训练步数变化的曲线。

---

## 五、断点续跑

每个 checkpoint 用独立的请求缓存（默认 `results/cache/<checkpoint>.sqlite`，lm-eval 会加 `_rank0.db` 后缀）。
`run_longbench.slurm` 里已加 `#SBATCH --requeue`，节点重启/被抢占后自动重新排队，重跑自动续算。

**注意**：缓存 key 只和 prompt+生成参数有关，不区分模型权重。换权重/配置/seed/任务后要 `rm -rf cache/`。

---

## 六、`gka`（Gated KalmaNet）的 fla 依赖

`gka` 用的是 `fla.layers.gka.GatedKalmaNet`，这个层**不在官方 `flash-linear-attention` 里**，
而在 Undermyth 的 fork 的 `fix/mesa-cg` 分支：

```bash
pip install -e "git+https://github.com/Undermyth/flash-linear-attention@fix/mesa-cg#egg=flash-linear-attention"
```

（或直接用服务器上已经装好的、带 `fla/layers/gka.py` 的 fla 环境。）否则 `eval_longbench.py` 跑到
`gka` 时会 `ModuleNotFoundError: fla.layers.gka`。

---

## 七、注意事项

1. **数据集下载慢/失败**：脚本已用 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像；tokenizer 是本地 `./tokenizer`，无需 HF 登录。
2. **权重 key 对不上**：优先怀疑 fla 版本不一致；再用 `--n-layers/--dim/...` 对齐配置。
3. **先冒烟测试**：正式跑之前先 `--limit 5` 确认能加载、能生成。
4. **batch-size 保持 1**：长上下文生成更稳。


