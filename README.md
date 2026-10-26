# DeltaNet 0.4B 训练代码

在 **SLAI 集群（H100）** 上从零训练一个 **0.4B 参数的 DeltaNet** 语言模型：
24 层、hidden 1024、8 头，20B tokens 的 FineWeb-Edu，Muon + AdamW。

- 算法参考：[Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)（Yang et al., arXiv:2406.06484）
- 骨架参考：`lmtemplate` 的 pre-norm residual 结构（与本仓库其它分支一致）
- 训练配置：`D:\Users\邱培林\Desktop\delta_net.md`

> 本分支（`delta_net`）**只保留训练需要的代码**。原来的 LongBench / CRIR
> 评测脚本、`data.zip` 等都不在这个分支里。

---

## 一、模型结构

| 项目 | 取值 |
|---|---|
| tokenizer | Llama-2（`./tokenizer`，词表 32000，已本地化） |
| 层数 | 24 |
| hidden size | 1024 |
| 注意力头数 | 8（head dim = 128） |
| 每层结构 | RMSNorm → DeltaNet → 残差 → RMSNorm → GatedMLP → 残差 |
| 参数量 | ≈0.40B（embedding 与 lm_head 不共享） |
| token mixer | `fla.layers.delta_net.DeltaNet`（chunk 并行形式） |

DeltaNet 层的细节（`ModelConfig` 里可改）：

- `dn_mode: chunk`：训练用分块并行形式，序列维并行；`fused_recurrent` 只适合逐 token 解码。
- `dn_use_short_conv: true`、`dn_conv_size: 4`：q/k/v 上各有一个短卷积，**不要关**，论文和 fla 都指出关掉会明显掉点。
- `dn_use_gate: false`：论文主表是无输出门的 DeltaNet。想试 gated 变体把这里改 `true` 即可（会多一个 `g_proj` + 门控 RMSNorm）。
- `dn_qk_activation: silu`、`dn_qk_norm: l2`：q/k 先做 SiLU 再 L2 归一化。
- `dn_allow_neg_eigval: false`：对应 negative-eigenvalue 那篇的扩展，默认关。

## 二、训练配置（与 delta_net.md 一致）

| 项目 | 取值 |
|---|---|
| 数据 | `karpathy/fineweb-edu-100b-shuffle`（集群本地副本 `/home/transformers/datasets/karpathy--fineweb-edu-100b`） |
| 训练 tokens | 20B（`tokens`） |
| 序列长度 | 2048 |
| batch | 0.5M tokens/step（2^19 = 524,288，即 256 条序列） |
| 总步数 | 38,146（20B / 524,288） |
| 优化器 | 隐层 2-D 矩阵用 **Muon**；embedding / lm_head / 1-D 参数用 **AdamW** |
| 学习率 | embedding **2e-1**，hidden **2e-2**（Muon），unembedding **2e-3** |
| 调度 | linear warmup（0.5%）+ cosine 到 10%，**衰减视界 40B**（`schedule_tokens`） |
| 权重衰减 | 0 |
| 验证 | 每 1000 步在留出的 parquet 文件上算 token 加权 loss（每卡 20 个 batch） |

学习率是怎么分的（`train.py::build_optimizers`）：

| 参数 | 优化器 | lr |
|---|---|---|
| `emb.weight` | AdamW | 2e-1 |
| `lm_head.weight` | AdamW | 2e-3 |
| 所有 2-D 隐层矩阵（q/k/v/o、b_proj、MLP 的 up/gate/down） | Muon | 2e-2 |
| RMSNorm 的 gain、短卷积 kernel（1-D/3-D） | AdamW | 2e-2 |

> 注意：Muon 只接受 2-D 参数，而且学习率的量纲和 AdamW 不同（正交化后更新范数
> 约为 √(rows/cols)），所以 2e-2 只能和 Muon 搭配；把它给 AdamW 会直接训崩。

## 三、目录结构

```
.
├── module/
│   └── modeling.py        # LanguageModel / Layer / ModelConfig / chunked CE
├── muon.py                # Muon 优化器（Newton-Schulz 正交化）
├── data_stream.py         # 在线 tokenize：流式读 parquet + 打包 + 断点位置
├── train.py               # DDP 训练主脚本（断点续跑、验证、日志、ckpt）
├── prepare_data.py        # （可选）离线预 tokenize 成 train.bin，训练不再依赖
├── configs/
│   └── delta_net_0.4b.yaml
├── tokenizer/             # Llama-2 tokenizer（4 个文件）
├── requirements.txt
├── setup_env.sh           # 建 conda 环境 ncrc_train
├── run_train.sh           # 单机多卡直接跑
├── run_train.slurm        # Slurm 作业（--requeue）
└── README.md
```

## 四、跑起来

### 1. 环境

```bash
bash setup_env.sh              # 建 ncrc_train 环境并自检 import fla / DeltaNet
conda activate ncrc_train
```

版本是有硬约束的：**torch ≥ 2.9**（带 triton 3.5）配 **fla ≥ 0.5.2**（要 triton ≥ 3.3）。
torch 2.4 只带 triton 3.0，`import fla` 会直接报
`Autotuner.__init__() got an unexpected keyword argument 'do_bench'`。
`setup_env.sh` 默认已经改成 `torch==2.9.1 + cu130`，换 CUDA 版本用
`PYTORCH_CUDA=12.8 bash setup_env.sh`。

**如果这台机器只做数据预处理**（不训练、无 GPU），不用装 torch/fla，
用轻量依赖就够（约 1.5 GB 而不是 6 GB）：

```bash
conda create -y -n slai_prep python=3.11
conda activate slai_prep
pip install -r requirements-prep.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python prepare_data.py --source ... --out ...
```

`requirements-prep.txt` 只有 `datasets / transformers / numpy / huggingface_hub`。
预处理产物（`train.bin`）是纯二进制，**与 torch 版本、CUDA 版本、GPU 型号都无关**，
在哪台机器上生成都一样，可以直接拷到训练集群。

### 2. 数据（在线 tokenize，不需要预处理）

训练时直接流式读 parquet、用**当前配置的 tokenizer** 现 tokenize 并打包成定长序列
（`data_stream.py`）。所以：

- **换 tokenizer 只要改 `data.tokenizer` 再重启**，不用重新处理 40 GB 语料；
- 不占磁盘、不用等预处理；
- 每个 rank 只读自己的文件 `files[rank::world]`，天然不重复；
- 流的位置（第几个文件 / 第几篇文档 / 打包缓冲）会写进 checkpoint，重启时只
  解码跳过已读文档、**不重新 tokenize**，所以续跑不会浪费算力。

配置里只需要指到 parquet 目录：

```yaml
data:
  source: /nas/datasets/karpathy--fineweb-edu-100b
  tokenizer: ./tokenizer
  val_files: 20          # 最后 20 个 parquet 文件留作验证，不参与训练
  val_tokens: 10000000
```

**验证集怎么来的**：最后 `val_files` 个 parquet 文件不参与训练，启动时把它们
里的一段（`val_tokens`）编码进内存（约 20 MB）作为验证集。因为是**按文件**留出
的，同一篇文档不会既在训练集又在验证集。验证时用固定种子的窗口，跨 step 可比。

> `prepare_data.py` 仍然保留，如果你以后想离线预 tokenize 也能用，但训练已经
> 不依赖它了。

### 3. 冒烟测试

```bash
# 不读数据，随机 token 跑 20 步，验证模型/优化器/显存
bash run_train.sh --smoke-test

# 用真实数据跑 200 步
bash run_train.sh --num-steps 200
```

### 4. 正式训练

```bash
# 2 卡（accum=8），数据直接来自 /nas 的 parquet 目录（在线 tokenize）
NGPU=2 \
DATA=/nas/datasets/karpathy--fineweb-edu-100b \
OUT=/data/qpl/delta_net/runs/delta_net_0.4b \
  bash run_train.sh

# 或 Slurm
sbatch run_train.slurm
```

常用参数：

```bash
NGPU=4 bash run_train.sh                       # 4 卡（accum 自动变 4）
bash run_train.sh --micro-batch-size 8         # 显存不够时减半（accum 自动补上）
bash run_train.sh --resume none                # 从头开始
bash run_train.sh --resume runs/delta_net_0.4b/ckpt_0005000.pt
bash run_train.sh --tokenizer ./tokenizer_v2   # 换 tokenizer（会打警告）
bash run_train.sh --schedule-tokens 60000000000  # 衰减视界改成 60B
```

## 五、断点续跑（作业重启）

- 每 `save_every`（默认 500 步）写一次 `runs/delta_net_0.4b/ckpt_<step>.pt`，
  先写 `.tmp` 再 `os.replace`，所以不会出现半截 checkpoint。
- `--resume auto`（默认）会自动找 `ckpt_*.pt` 里步数最大的那个继续。
- Slurm 里带了 `--requeue`，节点故障后作业自动重新排队；重跑同一条命令即可接上。
- checkpoint 里除了权重/优化器，还存了**每个 rank 的数据流位置**（第几个文件、
  文件内第几篇文档、打包缓冲里剩的 token）和 **tokenizer 指纹**。续跑时会跳到
  那个文档继续读（只解码 parquet，不重新 tokenize），所以数据流和没中断时一致。
- 如果换了 tokenizer 再续跑，会打印 `tokenizer changed (old -> new)` 警告——此时
  数据分布变了，loss 不能和之前直接比。
- 收到 SIGTERM/SIGINT 时会先存一个 checkpoint 再退出。
- 只保留最近 `keep_last=3` 个 checkpoint，老的自动作废。

## 六、产物与日志

```
runs/delta_net_0.4b/
├── ckpt_0000500.pt      # model + muon + adamw + step + config
├── log.jsonl            # 每 log_every 步：loss / lr / tok/s；每 eval_every 步多一行 val_loss
└── train_config.json    # 本次运行的完整参数快照
```

`log.jsonl` 里训练行和验证行是分开的，验证行长这样：

```json
{"step": 1000, "val_loss": 2.7341}
```

想画曲线直接：

```bash
python -c "
import json
rows=[json.loads(l) for l in open('runs/delta_net_0.4b/log.jsonl')]
tr=[(r['step'],r['loss']) for r in rows if 'loss' in r]
va=[(r['step'],r['val_loss']) for r in rows if 'val_loss' in r]
print('train', tr[-1] if tr else None, '| val', va[-1] if va else None)
"
```

验证 loss 是 **token 加权**的，并且多卡时会做 all-reduce，所以是全局所有卡的平均值。
关掉验证用 `--eval-every 0`，调频率用 `--eval-every 250`。

每个 checkpoint 里除了权重还有 `model_config`，之后评测时直接：

```python
ckpt = torch.load("ckpt_0038146.pt", map_location="cpu")
cfg = ModelConfig.from_dict(ckpt["model_config"])
model = ModelForCausalLM(cfg)
model.load_state_dict(ckpt["model"])
```

## 七、和论文配置的差异（有意为之）

`delta_net.md` 是这次实验的准绳，所以和论文有这些不同：

1. **规模**：论文主结果是 1.3B / 100B tokens；这里是 0.4B / 20B tokens。
2. **纯 DeltaNet**：论文还试了两种混合（隔层滑窗注意力、两层全局注意力）。这里
   24 层全是 DeltaNet，没有混 softmax 层，便于和仓库里 gla/gdn/kda 三个 0.4B
   baseline 直接对比。
3. **无输出门**：`use_gate=false`，对应论文主表。
4. **无 weight decay、无梯度裁剪**：Muon 的更新已做正交化，参考实现也不裁剪。
5. **短卷积保留**：`use_short_conv=true`，与论文一致。

## 八、常见问题

**Q: 多卡启动卡住 10 分钟，然后报 `Watchdog caught collective operation timeout` 和
`DDP expects same model across all ranks`？**
这不是模型问题，是 NCCL 的 P2P 通道挂死了。某些 Blackwell 工作站（实测 RTX 6000D /
sm_120）上会出现：日志走到 `via P2P/CUMEM` → `Connected all rings` 之后就再没有进展，
连 1 个元素的 `all_reduce` 都完不成。关掉 P2P 改走共享内存即可：

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1                      # 机器上有 mlx5 网卡、且可能有一块是 down 的
export NCCL_SOCKET_IFNAME=<你 UP 的那块网卡>   # ip -br addr 里看，比如 ens5f0np0
```

最小复现/验证脚本：

```python
# /tmp/nccl_test.py
import os, torch, torch.distributed as dist
print(f"[{os.environ['RANK']}] start", flush=True)
dist.init_process_group("nccl")
print(f"[{os.environ['RANK']}] init done", flush=True)
r = int(os.environ["RANK"]); torch.cuda.set_device(r)
t = torch.ones(1, device=f"cuda:{r}")
dist.all_reduce(t); torch.cuda.synchronize()
print(f"[{r}] allreduce ok {t.item()}", flush=True)
dist.destroy_process_group()
```

```bash
NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens5f0np0 \
  timeout 45 torchrun --nproc_per_node=2 /tmp/nccl_test.py > /tmp/nccl.log 2>&1
```

正常时日志里是 `via SHM/direct/direct`；异常时是 `via P2P/CUMEM` 之后卡住。
关掉 P2P 后梯度走主机内存，0.4B 模型每步只有 ~1.5 GB，对速度影响可以忽略
（H100/NVLink 机器上**不要**关，会白白损失带宽）。

**Q: 报 `ImportError: libcuda.so` / triton 相关错误？**
`fla` 的 chunk kernel 依赖 triton，必须在 GPU 节点上跑，不能在登录节点。

**Q: 显存不够（OOM）？**
先看 `micro=16` 是否够用：0.4B / seq 2048 / 24 层的激活，`micro=16` 约 40 GB，
`micro=32` 约 80 GB（实测 83 GB 卡在 32 时会 OOM）。还不行就 8 → 4，`accum` 会按
`global_seqs / (world_size * micro)` 自动补上，总 batch 不变。也可以用
`--grad-checkpoint`（慢 ~20%）。

**Q: 吞吐大概多少？**
H100 上单卡实测（micro=8、accum=32、seq 2048）约 **10.4 万 tok/s**，8 卡 DDP 大约
**80 万 tok/s**，20B tokens 约 **7 小时**。RTX 6000D（sm_120，P2P 关闭）上 2 卡合计
约 **10.7 万 tok/s**，20B 约 50 小时。第一个 `log` 里的 `tok/s` 才是准的
（第 1 步有 triton 编译和预热，会显示成 6 万左右）。

**Q: 能不能多机？**
`run_train.slurm` 里 `--nodes` 改大即可，`torchrun` 已经按
`SLURM_NODEID` / `SLURM_NNODES` 配好了。DDP 全参数复制，0.4B 在 80GB 卡上没问题。

**Q: fla 版本？**
`>=0.5.2`，且必须配 torch ≥ 2.9（triton ≥ 3.3）。集群上实测可用的组合是
`torch 2.9.1+cu130 + flash-linear-attention 0.5.2`（`DeltaNet.__init__` 的 16 个参数
与 `GatedMLP(hidden_size, hidden_ratio)` 都已核对过）。如果以后 fla 改了构造函数，
`module/modeling.py::_build_token_mixer` 是唯一需要改的地方。

## 九、评测（验证 checkpoint 是否正确）

`eval_quick.py` 用 lm-eval 的 `HFLM` 直接跑训练产出的 checkpoint（不需要转 HF 格式）。

```bash
pip install "lm-eval>=0.4.9"          # 评测依赖，训练环境里默认没装

# 快速验证（2 个任务 × 500 条，约 10 分钟）
python eval_quick.py \
  --checkpoint /data/qpl/delta_net/runs/delta_net_0.4b_v2/ckpt_0038146.pt \
  --tasks piqa,arc_easy --limit 500

# 完整 Table-4 口径（9 个任务，约 2–3 小时）
python eval_quick.py \
  --checkpoint .../ckpt_0038146.pt \
  --output-dir ./eval_results
```

判读：论文 Table 4 的 **DN 基线**（340M / 15B tokens）平均 **47.41**；
我们 374M / 20B tokens，**46–49 属正常**，低于 40 说明 checkpoint 或加载有问题。

三个实现细节：

- **`--batch-size` 必须为 1**：模型是 packed-sequence 的循环结构，没有 padding mask；
- 模型会被转成 **bf16**（fla 的 chunk 内核只支持 bf16）；
- 输出在 `--output-dir/<ckpt名>.json`，控制台打印一张 task/metric/value 表和 8 项平均。

