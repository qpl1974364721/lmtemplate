# CRIR 评测代码（Anti-Hebbian Attention / 线性注意力基线）

在 **SLAI 集群（H100）** 上，用 [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
对 `lmtemplate` 训练的 **gla / gdn / kda 三个模型**跑 **CRIR** 基准。

CRIR 对应论文 [Preconditioned DeltaNet](https://arxiv.org/abs/2604.21100)（arXiv:2604.21100）Table 4
的两大类下游任务：

- **Commonsense Reasoning（常识推理，9 个）**：LAMBADA、WikiText、ARC-Easy、ARC-Challenge、
  HellaSwag、PIQA、WinoGrande、BoolQ、SciQ
- **In-context Retrieval（上下文检索，5 个）**：FDA、SWDE、SQuAD、TriviaQA（TQA）、DROP，
  2K 上下文，cloze 补全 + "contains" 准确率

所有任务都通过 lm-eval-harness 零样本评测。5 个 ICR 任务已本地化在 `tasks/` 下（因为官方 harness
只在新版本才带 `fda/swde/squad_completion`，并且一直没有 cloze 版 TriviaQA / DROP）。

- 架构：`https://github.com/Undermyth/lmtemplate`（核心已拷到本目录 `module/`）
- 权重：服务器上 `checkpoints/` 下的 Lightning `.ckpt` 文件（不从 HuggingFace 下载）

---

## 模型与权重

`checkpoints/` 下每个子目录是一个模型族，里面是不同训练 token 数（5bt ~ 38bt）的 checkpoint：

| 目录 | 模型 | 架构（`attn_impl`） |
|------|------|--------------------|
| `gdn-muon/` | Gated DeltaNet | `gdn` → `fla.layers.gated_deltanet.GatedDeltaNet` |
| `gla/` | Gated Linear Attention | `gla` → `fla.layers.simple_gla.SimpleGatedLinearAttention` |
| `kda/` | Kalman Delta Attention | `kda` → `fla.layers.kda.KimiDeltaAttention` |
| `gka/` | Gated KalmaNet | `gka` → `fla.layers.gka.GatedKalmaNet`（本次**不跑**） |

`gdn-muon` 的 `muon` 表示用 Muon 优化器训练，文件名里的 `0.4b` 表示约 0.4B 参数
（`dim=1024, n_layers=24`）；`bt` 是训练的十亿 token 数，`step` 是训练步数。

本次默认只跑 `20bt` 的 checkpoint，且排除 gka：

- `checkpoints/gla/gla-0.4b-muon-20bt-step40960.ckpt`
- `checkpoints/gdn-muon/gdn-0.4b-muon-20bt-step40960.ckpt`
- `checkpoints/kda/kda-0.4b-muon-20bt-step40960.ckpt`

---

## 目录结构

```
.
├── module/                 # 架构代码（modeling.py 内含 attn_impl 开关）
├── tasks/                  # 本地化的 5 个 ICR 任务（icr_tasks.py + 5 个 yaml）
├── crir_bench.py           # 任务列定义 / 指标抽取 / 两组平均分（eval 与表格共用）
├── eval_crir.py            # 主脚本：发现 checkpoints/ 下 .ckpt 并逐个跑 CRIR
├── make_crir_table.py      # 汇总脚本：生成论文 Table 4 风格的表格
├── tokenizer/              # 已下载到本地的 Llama-2 tokenizer（无需 HF 登录）
├── requirements.txt
├── setup_env.sh
├── run_crir.slurm          # Slurm 作业（H100）
├── run_crir.sh             # 交互式单卡直接运行
├── run_crir_multi.sh       # 多卡分片并行运行（NUM_GPUS=N）
└── README.md
```

### 对架构源码的改动

`lmtemplate` 的多个分支只差 `Layer` 里 attention 这一行。这里把 `module/modeling.py` 合并成一个文件，
用 `ModelConfig.attn_impl` 选择注意力核：`gla` / `gdn` / `kda` 对应 fla 层。另保留两处让 lm-eval 能
正确接管的修复：`tie_weights()` no-op + `tie_word_embeddings=False`。

---

## 一、环境准备

```bash
bash setup_env.sh                 # 建 conda 环境 slai_eval 并装依赖
source /home/qpl/miniconda3/etc/profile.d/conda.sh
conda activate slai_eval
```

> **不需要 HF 登录**：tokenizer 已下载到本地 `./tokenizer`；CRIR 数据集都是公开的，
> 脚本里已设置 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像下载。

`flash-linear-attention`（`fla`）的版本会影响 fla 层的参数名/接口，**必须与训练 checkpoint 时使用的版本兼容**。
本次只跑 gla/gdn/kda，用官方 `flash-linear-attention>=0.4.1` 即可，不需要 gka 的 Undermyth fork。

---

## 二、运行

### 第一次跑之前（重要）

旧 LongBench 的缓存/结果和 CRIR 不通用，先清掉：

```bash
rm -rf cache results
```

### Slurm 提交（推荐）

改 `run_crir.slurm` 里的 `--partition` / `--gres` / 环境名，然后：

```bash
sbatch run_crir.slurm
```

### 交互式（单卡）

```bash
bash run_crir.sh
```

### 多卡并行（推荐，提速）

```bash
nvidia-smi -L                     # 看实例有几张卡
NUM_GPUS=3 bash run_crir_multi.sh # 3 个 checkpoint，3 张卡正好并行
```

### 直接用 `eval_crir.py`

```bash
# 全部 gla/gdn/kda 的 20bt checkpoint（自动发现并排除 gka）
python eval_crir.py --checkpoint-dir ./checkpoints --filter 20bt --exclude-variant gka

# 只跑某个模型族
python eval_crir.py --filter 20bt --variant gla

# 只跑单个 checkpoint
python eval_crir.py --checkpoint ./checkpoints/gla/gla-0.4b-muon-20bt-step40960.ckpt --variant gla

# 冒烟测试（每个任务只跑 5 条，注意：多数 commonsense 任务是 loglikelihood，
# --limit 5 也足够验证能加载、能算 logprob、能生成）
python eval_crir.py --limit 5
```

结果写入 `results/<目录名>/<checkpoint名>/results.json`，并在 `results/summary.json` 汇总每个
checkpoint 的各列分数与两组平均分。

---

## 三、任务与指标

| 组 | 任务 | lm-eval 任务名 | 指标（表里展示） |
|----|------|---------------|------------------|
| Commonsense | LAMBADA | `lambada_openai` | `perplexity`（ppl↓）+ `acc` |
| Commonsense | WikiText | `wikitext` | `word_perplexity`（ppl↓） |
| Commonsense | ARC-Easy | `arc_easy` | `acc` |
| Commonsense | ARC-Challenge | `arc_challenge` | `acc_norm`（表里写作 acc_n） |
| Commonsense | HellaSwag | `hellaswag` | `acc_norm` |
| Commonsense | PIQA | `piqa` | `acc` |
| Commonsense | WinoGrande | `winogrande` | `acc` |
| Commonsense | BoolQ | `boolq` | `acc` |
| Commonsense | SciQ | `sciq` | `acc` |
| ICR | FDA | `crir_fda`（本地） | `contains` |
| ICR | SWDE | `crir_swde`（本地） | `contains` |
| ICR | SQuAD | `crir_squad`（本地） | `contains` |
| ICR | TriviaQA | `crir_tqa`（本地） | `contains` |
| ICR | DROP | `crir_drop`（本地） | `contains` |

- **Commonsense Avg** = 8 个准确率任务的均值（ARC-E、ARC-C、HellaSwag、LAMBADA、PIQA、
  WinoGrande、BoolQ、SciQ；**不含** LAMB/Wiki 的 ppl）。
- **ICR Avg** = 5 个 ICR `contains` 准确率的均值。

ICR 的 5 个任务都是 zero-shot cloze 补全（`generate_until`，`until="\n"`，最多生成 48 token），
分数是「生成串里是否包含正确答案」的 case-insensitive 子串匹配准确率，与论文一致。

---

## 四、结果解读

每个 checkpoint 的完整结果在 `results/<族>/<checkpoint>/results.json`，汇总在 `results/summary.json`。

```bash
python make_crir_table.py
```

会读 `results/*/*/results.json`，生成论文 Table 4 风格的表格：

- `results/crir_table.md`（Markdown 表）
- `results/crir_table.csv`（同款列）

准确率已 ×100（如 44.49 表示 44.49%），ppl 保持原值（越低越好）。

---

## 五、断点续跑

每个 checkpoint 用独立的请求缓存（默认 `cache/<checkpoint>.sqlite`，lm-eval 会加 `_rank0.db` 后缀）。
`run_crir.slurm` 里已加 `#SBATCH --requeue`，节点重启/被抢占后自动重新排队，重跑自动续算。

**注意**：缓存 key 只和 prompt+生成参数有关，不区分模型权重。换权重/配置/seed/任务后要 `rm -rf cache/`。

---

## 六、常见问题

1. **数据集下载慢/失败**：脚本已用 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像；tokenizer 是本地 `./tokenizer`，无需 HF 登录。
2. **权重 key 对不上**：优先怀疑 fla 版本不一致；再用 `--n-layers/--dim/...` 对齐配置。
3. **先冒烟测试**：正式跑之前先 `--limit 5` 确认能加载、能算 logprob、能生成。
4. **batch-size 保持 1**：`modeling.py` 的 `generate()` 不做逐条 EOS 掩码，批生成会提前/延后截断。
5. **gka 已排除**：默认 `--exclude-variant gka`，且多卡脚本只收集 gla/gdn-muon/kda 三个目录。
