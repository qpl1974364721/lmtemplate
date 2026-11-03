# CRIR 评测代码（Anti-Hebbian Attention / 线性注意力基线）

在 **SLAI 集群（H100）** 上，用 [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
对 `lmtemplate` 训练的 **gla / gdn / kda 三个模型**跑 **CRIR** 基准。

CRIR 对应论文 [Preconditioned DeltaNet](https://arxiv.org/abs/2604.21100)（arXiv:2604.21100）Table 4
的两大类下游任务：

- **Commonsense Reasoning（常识推理，9 个）**：LAMBADA、WikiText、ARC-Easy、ARC-Challenge、
  HellaSwag、PIQA、WinoGrande、BoolQ、SciQ
- **In-context Retrieval（上下文检索，5 个）**：FDA、SWDE、SQuAD、TriviaQA（TQA）、DROP，
  4K 上下文。其中 FDA/SWDE/SQuAD/DROP 是 cloze 补全 + "contains" 准确率，
  TQA 使用 lm-eval 官方的 `triviaqa`（closed-book `rc.nocontext`，`exact_match`）。

所有任务都通过 lm-eval-harness 零样本评测。FDA/SWDE/SQuAD/DROP 4 个 ICR 任务已本地化在 `tasks/` 下
（因为官方 harness 只在新版本才带 `fda/swde/squad_completion`，并且一直没有 cloze 版 DROP）；
TQA 直接用 lm-eval 自带的 `triviaqa`。

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
| `gka/` | Gated KalmaNet | `gka` → `fla.layers.gka.GatedKalmaNet`（可单独跑：`bash run_crir_gka.sh`） |

`gdn-muon` 的 `muon` 表示用 Muon 优化器训练，文件名里的 `0.4b` 表示约 0.4B 参数
（`dim=1024, n_layers=24`）；`bt` 是训练的十亿 token 数，`step` 是训练步数。

本次默认只跑 gla/gdn/kda 的 `20bt` checkpoint（排除 gka）；gka 可单独跑：

- `checkpoints/gla/gla-0.4b-muon-20bt-step40960.ckpt`
- `checkpoints/gdn-muon/gdn-0.4b-muon-20bt-step40960.ckpt`
- `checkpoints/kda/kda-0.4b-muon-20bt-step40960.ckpt`
- `checkpoints/gka/gka-20bt-step40960.ckpt`（单独跑 gka 时）

---

## 目录结构

```
.
├── module/                 # 架构代码（modeling.py 内含 attn_impl 开关）
├── tasks/                  # 本地化的 4 个 ICR 任务（FDA/SWDE/SQuAD/DROP；TQA 用 lm-eval 自带 triviaqa）
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
本次默认只跑 gla/gdn/kda，用官方 `flash-linear-attention>=0.4.1` 即可。**若要跑 gka，需要 Undermyth 的 fork，锁定到 commit `45412236ed34d88aceadfd695a6d4ab1d614e4d7`（含 `fla.layers.gka.GatedKalmaNet`，且修好了 q/k 归一化的 dtype 问题）；官方 fla 没有这个层。**

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
# 全部 gla/gdn/kda 的 20bt checkpoint（自动发现，排除 gka）
python eval_crir.py --checkpoint-dir ./checkpoints --filter 20bt --exclude-variant gka

# 单独跑 gka
python eval_crir.py --checkpoint-dir ./checkpoints --filter 20bt --variant gka

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
| ICR | TriviaQA | `triviaqa`（lm-eval 官方） | `exact_match` |
| ICR | DROP | `crir_drop`（本地） | `contains` |

- **Commonsense Avg** = 8 个准确率任务的均值（ARC-E、ARC-C、HellaSwag、LAMBADA、PIQA、
  WinoGrande、BoolQ、SciQ；**不含** LAMB/Wiki 的 ppl）。
- **ICR Avg** = 5 个 ICR 任务准确率的均值（FDA/SWDE/SQuAD/DROP 为 `contains`，TQA 为 `exact_match`）。

FDA/SWDE/SQuAD/DROP 是 zero-shot cloze 补全（`generate_until`，`until="\n"`，最多生成 48 token），
分数是「生成串里是否包含正确答案」的 case-insensitive 子串匹配准确率；TQA 用 lm-eval 官方
`triviaqa`（`mandarjoshi/trivia_qa` 的 `rc.nocontext` 闭卷问答，指标为 `exact_match`）。

### 只重跑 TQA（lm-eval 官方 `triviaqa`，产出 5 个结果）

目标：`GDN / GLA / KDA / GKA(新版 bf16) / GKA(旧版 local fp16)` 共 5 个 TQA 结果。

| 结果 | fla | dtype | 说明 |
|---|---|---|---|
| GDN / GLA / KDA / GKA(新版) | Undermyth fork commit `4541223` | bfloat16 | 全 bf16 |
| GKA(旧版) | Undermyth fork `fix/mesa-cg` | bfloat16（gka 内核内部 fp16） | `l2_norm(..., output_dtype=torch.float16)` + `o_proj` pre-hook |

**服务器完整流程（从干净结果开始）**：

```bash
cd /home/qpl/code/lmtemplate
source /home/qpl/miniconda3/etc/profile.d/conda.sh
conda activate slai_eval
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/qpl/.cache/huggingface

# 0) 拉最新代码（含 o_proj pre-hook 和 TQA 改动）
git pull origin CRIR
grep -n "cast_input_to_weight_dtype" module/modeling.py   # 确认 pre-hook 在位
python - <<'PY'
from crir_bench import ICR
assert ("triviaqa", ("exact_match",), "TQA (acc)", "acc") in ICR
print("official triviaqa/exact_match OK")
PY

# 1) 安装新版 fla（commit 4541223）。不要复用旧的 fla-gka-4541223：
#    服务器上那个浅克隆实际停在 4c62f28，并不是 4541223。
FLA_NEW_TAG=45412236ed34d88aceadfd695a6d4ab1d614e4d7
FLA_NEW_DIR="$PWD/flash-linear-attention-$FLA_NEW_TAG"
if [ ! -f "$FLA_NEW_DIR/fla/layers/gka.py" ]; then
  wget -O fla-4541223.zip \
    "https://codeload.github.com/Undermyth/flash-linear-attention/zip/$FLA_NEW_TAG"
  unzip -q fla-4541223.zip
fi
pip uninstall -y flash-linear-attention
pip install --no-deps "$FLA_NEW_DIR"
python check_gka_version.py new

# 2) 跑 GDN / GLA / KDA / GKA(新版) 的 TQA
rm -rf results_tqa cache_tqa

# 单卡：
bash run_crir_tqa.sh

# 或多卡并行不同模型（不会把单个模型拆到多卡）：
NUM_GPUS=4 bash run_crir_tqa_multi.sh                       # → results_tqa/

# 3) 复制旧版 fla（fix/mesa-cg）到工作目录再安装，避免改动 /home/csy/pkgs
FLA_OLD_SRC=/home/csy/pkgs/flash-linear-attention
FLA_OLD_DIR="$PWD/.fla-old-fix-mesa-cg"
rm -rf "$FLA_OLD_DIR" && mkdir -p "$FLA_OLD_DIR"
cp -a "$FLA_OLD_SRC/." "$FLA_OLD_DIR/"
pip uninstall -y flash-linear-attention
pip install --no-deps "$FLA_OLD_DIR"
python check_gka_version.py old

# 4) 跑 GKA(旧版 local fp16) 的 TQA
rm -rf results_tqa_oldgka cache_tqa_oldgka
bash run_crir_tqa_oldgka.sh                                 # → results_tqa_oldgka/

# 5) 合并 5 个结果、重算 ICR Avg、出表
python make_crir_table.py --results-dir ./results \
    --tqa-dir ./results_tqa --tqa-oldgka-dir ./results_tqa_oldgka
```

- `results_tqa/`：GDN / GLA / KDA / GKA(新版 bf16) 的 TQA。
- `results_tqa_oldgka/`：GKA(旧版 local fp16) 的 TQA。
- `make_crir_table.py` 把两个 TQA 目录分别合并进 `results/` 里的旧全量结果（FDA/SWDE/SQuAD/DROP），
  重算 `ICR Avg = mean(TQA, FDA, SWDE, SQuAD, DROP)`，输出 `results/crir_table_tqa.md` / `.csv`。
- 输出表格 5 行：`GDN`、`GLA`、`KDA`、`GKA (new, bf16)`、`GKA (old, local fp16)`。
- 断点续跑：TQA 请求缓存分别在 `cache_tqa/`（新版）和 `cache_tqa_oldgka/`（旧版）；服务器重启后重跑对应脚本会续算。
- 切换 fla 后一定要 `rm -rf` 对应的 cache 目录（缓存 key 不区分 fla 版本）。
- `run_crir_tqa.sh` 和 `run_crir_tqa_oldgka.sh` 会在开跑前分别检查新版/旧版 gka；版本不符会立即退出，避免把同一版 GKA 跑成两行。
- `run_crir_tqa_multi.sh` 只并行“不同模型”：每个 worker 使用一张 GPU 和 `batch-size=1`。旧版 GKA 由于要切换 FLA 环境，仍在单独阶段运行。

### 生成性能修复

`module/modeling.py` 的 `generate()` 现在会：

- 使用 `torch.inference_mode()`；
- 预分配输出 buffer，不再每生成一个 token 执行一次 `torch.cat`；
- position id 直接在 GPU 上递增，不再每步从 CPU 创建并拷贝；
- 读取 lm-eval 的停止符，并每 `8` 个 token 检查一次 EOS、换行、句号和逗号。

检查间隔为 8 时最多会多生成 7 个 token，但 lm-eval 会在第一个停止符处截断，因此不改变评分文本。先做小样本性能验证：

```bash
rm -rf results_tqa_smoke cache_tqa_smoke
/usr/bin/time -v python eval_crir.py \
    --checkpoint checkpoints/gdn-muon/gdn-0.4b-muon-20bt-step40960.ckpt \
    --variant gdn-muon --tokenizer ./tokenizer --tasks triviaqa \
    --dtype bfloat16 --max-length 4096 --batch-size 1 --device cuda:0 \
    --limit 20 --output-dir ./results_tqa_smoke --no-cache
```

确认秒/条和 GPU 利用率明显改善，再进行完整的 `17,944` 条测评。

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
5. **gka**：默认脚本（`run_crir.sh` / `run_crir_multi.sh`）排除 gka；要单独跑 gka 用 `bash run_crir_gka.sh`（需要 Undermyth fork 的 fla，见上文）。
