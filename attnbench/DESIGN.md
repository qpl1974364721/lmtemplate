# attnbench 设计文档

> 状态：规划稿（含待确认项）。实现前需确认 §7 中的问题。
>
> 环境（已验证）：Python 3.13.15 / torch 2.14.0+cu126 / RTX 3090 24GB (cc 8.6)
> / fla 0.6.0（chunk_kda、chunk/fused_recurrent_gated_delta_rule、simple_gla 算子齐全）
> / rich, yaml, einops, jaxtyping 可用；无 pandas（CSV 手写）。

## 0. 目标回顾

1. **易扩展**：新增 attention 变体只需添加一个小文件并注册，不改 model/bench/report 代码。
2. **规格可配置**：batch size × sequence length 组合（网格或显式对），YAML + CLI 覆盖。
3. **指标**（每个 variant × B × S × prefill/decode）：
   - 单 attention 延迟（每层 attention 前向延迟）；
   - attention / (attention + mlp) 时间占比；
   - 整模型 tokens per second。
4. **双格式输出**：终端表格（rich）+ JSON（全量明细，含每 iter 数组）+ CSV（long 格式）。

## 1. 现有草稿的处置

| 草稿文件 | 处置 | 需修复的问题 |
|---|---|---|
| `cache_utils.py` | 迁移为 `attnbench/common/cache.py` | 基本可用；`_seen_tokens` 仅在 layer 0 累加（保留，但加注释） |
| `attn_impl.py` | 拆为 `attnbench/cores/base.py` + `cores/{gla,kda}.py` | 接口统一（aux 命名 `g_raw`/`beta_raw`），core 不再收未用的 `x` |
| `attn.py` | 改写为 `attnbench/model/attention.py`（统一脚手架） | `VARIANTS` 未定义（NameError）；`head_v_dim` 拼写错误；`build_attn` 只支持 kda/gla；缺 `layer_idx` 接线；改用自写 `Cache` 直连，去掉对 fla cache helper 的依赖 |
| `modeling.py` | 拆为 `attnbench/model/{layer,modeling}.py` | `Layer` 构造 `Attention()` 缺 variant/layer_idx；`position_embeddings` 传参不匹配；3/4 元组脆弱；`generate()` 的采样逻辑对 bench 是纯开销，精简为 greedy |

## 2. 总体架构

```
attnbench/
├── attnbench/
│   ├── __init__.py            # 触发变体自动发现；导出公开 API
│   ├── registry.py            # 注册表：register_core / register_block + 自动发现 + --list-variants
│   ├── config.py              # 全部 dataclass + YAML 加载 + CLI 覆盖合并 + 校验
│   ├── common/
│   │   ├── cache.py           # 统一 Cache（recurrent_state / KV(可选滑窗) / conv_state）
│   │   ├── rope.py            # RotaryEmbedding（自 modeling.py 迁出）
│   │   ├── short_conv.py      # fla ShortConvolution 薄封装（可选依赖）
│   │   └── mlp.py             # GatedMLP
│   ├── cores/                 # ① kernel 层：变体数学核（uniform 脚手架下使用）
│   │   ├── base.py            # AttnCore 协议 + mode 语义
│   │   ├── softmax.py         # torch SDPA（MHA + RoPE）
│   │   ├── gla.py             # fla chunk/fused_recurrent_simple_gla
│   │   ├── kda.py             # fla chunk/fused_recurrent_kda（含 A_log/dt_bias 参数）
│   │   └── gdn.py             # fla chunk/fused_recurrent_gated_delta_rule
│   ├── blocks/                # ② block 层：原生完整 attention 块（预留，暂不实现）
│   ├── model/
│   │   ├── attention.py       # 统一 Attention 脚手架（KDA 式外层，build_core 来自注册表）
│   │   ├── layer.py           # Layer: norm → attn → 残差 → norm → mlp → 残差
│   │   ├── modeling.py        # LanguageModel / ModelForCausalLM + 精简 generate（greedy）
│   │   └── timed.py           # TimedModule + TimeCollector（attn/mlp 计时钩子）
│   ├── bench/
│   │   ├── timer.py           # CUDA Event 工具 + 统计量（mean/std/p50）
│   │   ├── runner.py          # 编排：variant × (B,S) × phase，隔离构建/销毁，OOM 捕获
│   │   └── microbench.py      # 可选：core-only 微基准（隔离 kernel 延迟）
│   ├── report/
│   │   ├── table.py           # rich 终端表
│   │   ├── json_out.py        # 全量 JSON（meta + records + 每 iter 数组）
│   │   └── csv_out.py         # long 格式 CSV
│   └── cli.py                 # python -m attnbench（__main__.py 入口）
├── configs/base.yaml          # 默认基准配置
├── tests/                     # registry / 正确性（chunk vs recurrent）/ 冒烟
├── scripts/                   # 后续分析脚本（读 CSV/JSON）
└── results/                   # 输出目录（.gitignore 语义，虽无 git）
```

### 变体扩展机制（对应需求 1）

两层抽象，注册表驱动：

- **`AttnCore`（kernel 层）**：给定投影后的 `q/k/v`、辅助张量（`g_raw`/`beta_raw` 等）、
  `recurrent_state` 与 `mode ∈ {chunk, recurrent}`，返回 `(o, new_state)`。
  纯数学核，不含投影/归一化/cache 逻辑。**新增变体 = 在 `cores/` 加一个 ~40 行文件**：

  ```python
  # attnbench/cores/my_variant.py
  from dataclasses import dataclass
  from attnbench.registry import register_core
  from attnbench.cores.base import AttnCore

  @dataclass
  class MyVariantConfig:
      my_param: float = 1.0

  @register_core("my_variant", config_cls=MyVariantConfig, description="...")
  class MyVariantCore(AttnCore):
      def __init__(self, config, num_heads, head_dim):
          super().__init__(config, num_heads, head_dim)
          # 少量核内参数（如 KDA 的 A_log/dt_bias）
      def forward(self, q, k, v, aux, state, mode, **kw):
          ...  # 调用 fla 或自写 kernel
          return o, new_state
  ```

- **`Attention` 脚手架（block 层）**：所有变体共享同一外层结构（KDA 式：q/k/v 投影 +
  可选 short conv + f_proj/b_proj/g_proj + FusedRMSNormGated + o_proj）、同一 mode 选择
  逻辑（grad→chunk；q_len 小→recurrent；否则 config.mode）、同一 cache 更新逻辑。
  core 用不到的 aux 输入自行忽略（如 softmax 只用 q/k/v + position）。

  这样**所有变体的外层参数量/计算量完全一致**，`attn/(attn+mlp)` 与整模型 TPS 跨变体可比。

- 注册表通过 `cores/__init__.py` 的包扫描**自动发现**（放新文件即注册），CLI 提供
  `--list-variants`。同时保留 `register_block` 槽位：未来接入「原生完整块」类变体
  （如 fla 的 `GatedKalmaNet`）时走同一管线，不改 bench/report 代码。

### RoPE 处理

`Layer` 传 `position_ids`（不预算 cos/sin）。core 内部自行决定是否用（softmax 用、
线性注意力忽略）；softmax core 内建惰性 cos/sin cache（按见过的最大 seq 长度构建）。
RoPE 开销计入 softmax 的 attention 时间——这正是该变体的真实成本。

## 3. 规格/配置系统（对应需求 2）

`configs/base.yaml`（默认值），CLI 参数覆盖 YAML，YAML 覆盖 dataclass 默认：

```yaml
model:
  vocab_size: 32000
  n_layers: 24
  dim: 1024
  n_heads: 8            # head_dim = dim // n_heads
  expand_ratio: 4

variants: [softmax, gla, kda, gdn]
variant_opts: {}        # per-variant 覆盖，如 {kda: {chunk_size: 64}}

specs:                  # 网格（笛卡尔积）；也可用显式 pairs，二者可叠加
  batch_sizes: [1, 4, 8]
  seq_lens: [512, 2048, 8192]
  # pairs: [{batch_size: 2, seq_len: 1536}]

decode:
  new_tokens: 256       # decode 步数；decode 紧跟 prefill，状态跨步保持

bench:
  dtype: bfloat16
  warmup: 3             # 每 combo 独立 warmup（覆盖 triton autotune 首调用编译）
  iters: 10
  seed: 0
  microbench: true      # 是否附加 core-only 微基准
report:
  out_dir: results
  formats: [table, json, csv]
  tag: null             # 结果子目录后缀；null 用时间戳
```

CLI：

```
python -m attnbench --config configs/base.yaml \
    [--variants kda,gla] [--batch 1,4] [--seq 512,2048] [--decode-tokens 256] \
    [--iters 10] [--warmup 3] [--dtype bfloat16] [--microbench/--no-microbench] \
    [--out results] [--tag exp1] [--dry-run] [--list-variants] [--smoke]
```

- 优先级：**CLI > YAML > dataclass 默认**；`--dry-run` 打印解析后的完整矩阵（含展开
  后的 run 标注），不占 GPU。
- `--smoke`：强制微型配置（1 层、dim 64、短序列、iters 1），验证管线端到端可跑。
- 形状网格两种写法等价：嵌套 `specs: {batch_sizes, seq_lens, pairs}`（base.yaml 风格）
  或顶层平铺 `batch_sizes/seq_lens/pairs`；同一键两处同时出现会报歧义错误。
- **`variant_opts` 扫描**：某选项取列表值时，该变体按各取值的笛卡尔积展开为多个
  run，结果中以 `name@key=val` 标注区分（如 `gka@num_iter=30`）；标量选项对该变体
  的所有 run 生效。JSON 记录同时携带 `variant`（注册名）、`label`（run 标注）与
  `variant_opts`（实际生效配置），CSV 的 variant 列使用 label。

## 4. 计时口径（对应需求 3）

**Phase 语义**

- `prefill`：一次 forward 处理 [B, S] tokens（chunk 模式，use_cache=False）。
- `decode`：prefill 建立状态后循环 N 步、每步 1 token（recurrent/fused 模式，
  KV/recurrent state 跨步保持，位置连续）。decode 延迟随步数几乎恒定（线性注意力）或
  缓增（KV cache）。

**计时通道**

1. **in-model 通道（主口径）**：`TimedModule` 包住每层 `attn` 与 `mlp` 模块，在模块边界
   记录 CUDA Event（异步记录，region 结束 sync 后统一结算 elapsed）。得到：
   - `attn_ms_per_layer`：每层 attention 前向延迟（对层与 iter 求均值）←「单 attention 延迟」
   - `attn_ms_total` / `mlp_ms_total` → `attn_ratio = attn / (attn + mlp)`
   - `model_ms`：trunk 前向总时间（emb + N×layer + final norm，**默认不含 lm_head**）
2. **e2e 墙钟通道**：`time.perf_counter` 包住测量区域（含一次结尾 sync）。
   - `prefill.tps = B×S / wall_ms`；`decode.tps = B×N / wall_ms`
   - decode 的墙钟含每步 lm_head + argmax + Python 调度——这是真实生成吞吐。
   - trunk 与 e2e 同时记录，报告以 trunk 为主口径（见 §7-Q3）。
3. **microbench 通道（可选）**：预生成 `q/k/v/g/beta` 直接调 core：
   - chunk 形态：`[B, S, ...]`（prefill 形）；
   - recurrent 形态：`[B, 1, ...]` + 长度为 S 的 state（decode 形）；
   - 隔离 kernel 本身延迟（不含投影/归一化/cache），用于纯 kernel 对比。

**附加记录**：每 iter 明细数组（进 JSON，供算 std/p50）、`peak_mem_mb`、
softmax 的 KV cache 增长对 decode 的逐开影响（per-step 数组）。

**方法论要点**

- 每 combo：固定 seed → 全新构建模型（eval、bf16、cuda）→ warmup W → 计时 K →
  `del` + `empty_cache` + `gc.collect()`，消除 allocator/autotune 残留。
- decode：每步记 event 但**只在循环末 sync 一次**（避免每步 sync 掩盖异步性）；
  TPS 用墙钟，分解用 event。event 分解之和 ≠ 墙钟属正常（device 空档来自
  Python 启动开销，小模型/decode 短步时更明显），文档中说明。
- CUDA Event 记录本身有极小 host 开销，不影响 device 时间线的分解正确性。
- OOM 防护：每 combo try/except `torch.cuda.OutOfMemoryError` → 记录
  `status: oom` 进结果并继续下一个 combo，不让单点失败毁掉整轮。

## 5. 输出（对应需求 4）

结果目录：`results/<时间戳>[_<tag>]/`，内含 `results.json`、`results.csv`，表格打到终端。

**终端表（rich）**，每 (variant, B, S) 两行：

| variant | B | S | phase | attn_ms/layer | attn/ms | mlp/ms | attn/(attn+mlp) | model_ms | tokens/s | peak_mem |
|---|---|---|---|---|---|---|---|---|---|---|
| kda | 4 | 2048 | prefill | 0.83 | 39.8 | 48.3 | 45.2% | 88.1 | 93000 | 2100 |
| kda | 4 | 2048 | decode | 0.02 | 0.19 | 0.23 | 45.2% | 0.42 | 9520 | 2110 |

（列为示例；mean/std 显示策略：表内 mean，std 进 JSON。）

**CSV（long 格式，一行一记录）**，列固定、缺项留空，方便 pandas pivot：

```
variant,batch_size,seq_len,phase,status,attn_ms_per_layer,attn_ms_total,mlp_ms_total,attn_ratio,model_ms,step_ms,tokens,tps,peak_mem_mb,chunk_ms,recurrent_ms
kda,4,2048,prefill,ok,0.83,39.8,48.3,0.452,88.1,,8192,93000,2100,,
kda,4,2048,decode,ok,0.02,0.19,0.23,0.452,,0.42,1024,9520,2110,,
```

**JSON**：

```json
{
  "meta": {"timestamp": "...", "hostname": "...", "gpu": "...", "torch": "...",
           "fla": "...", "python": "...", "config": {"...已解析配置回显"}},
  "records": [
    {"variant": "kda", "batch_size": 4, "seq_len": 2048, "status": "ok",
     "prefill": {
       "attn_ms_per_layer": {"mean": 0.83, "std": 0.01, "p50": 0.83},
       "attn_ms_total": {"mean": 39.8, "std": 0.4, "p50": 39.8},
       "mlp_ms_total": {"mean": 48.3, "std": 0.5, "p50": 48.3},
       "attn_ratio": {"mean": 0.452},
       "model_ms": {"mean": 88.1, "std": 0.9, "p50": 88.1},
       "wall_ms": {"mean": 89.0, "std": 1.0, "p50": 89.0},
       "tokens": 8192, "tps": 93000.0, "peak_mem_mb": 2100,
       "iters": {"attn_ms_total": [...], "mlp_ms_total": [...], "wall_ms": [...]}},
     "decode": {
       "new_tokens": 256,
       "attn_ms_per_step_per_layer": {"mean": 0.02, "std": 0.003, "p50": 0.02},
       "attn_ratio": {"mean": 0.452},
       "step_ms": {"mean": 0.42, "std": 0.05, "p50": 0.41},
       "step_ms_per_step": [...],
       "tokens": 1024, "tps": 9520.0, "peak_mem_mb": 2110,
       "iters": {"step_ms": [...]}},
     "microbench": {"chunk_ms": {"mean": 0.71}, "recurrent_ms": {"mean": 0.017}}}
  ]
}
```

## 6. 测试

- `tests/test_registry.py`：注册/发现/列出/未知变体报错。
- `tests/test_correctness.py`：每个 core 在小形状上 chunk vs recurrent 输出一致性
  （allclose 宽容差）；state roundtrip（chunk 产出的 state 喂给 recurrent 一步，与
  直接 chunk 两步等价）；Cache 不变量（seen_tokens、层间隔离）。
- `tests/test_smoke.py`：`--smoke` 级微型配置端到端：跑通 runner、产出 JSON/CSV/表、
  microbench 开关两种路径。

## 7. 已确认决议（2026-10-11 与项目所有者确认）

- **Q1 结构模式** → 统一脚手架。所有变体共享相同外层投影，只换 core；`register_block`
  槽位保留（未来接入原生块类变体），当前不实现。
- **Q2 「单 attention 延迟」口径** → 两者都记录：in-model 每层 attention 前向延迟为主
  指标；`bench.microbench: true` 时附加 core-only 微基准（chunk/recurrent 两形态）。
- **Q3 整模型 TPS 范围** → trunk 为主口径：prefill TPS = B×S / trunk 墙钟（不含
  lm_head）；decode TPS = B×N / decode 循环墙钟（含每步 lm_head 末位 logits + argmax，
  即真实生成吞吐）。模型永不物化全序列 logits（避免 B×S×vocab 的显存尖峰）。
- **Q4 初始变体集** → softmax + gla + kda + gdn（fla 0.6.0 均有现成 chunk/fused 算子；
  softmax 用 torch SDPA + RoPE + 静态预分配 KV cache）。
- **Q5 草稿文件处置** → 仓库已 `git init`，原始草稿提交于历史（commit `3ace19d`）；
  新包结构完成并验证通过后删除根目录草稿文件。

## 8. 实现期发现（重要）

- **fla kernel 必须全部用关键字参数调用**（与 `fla.layers` 的写法一致）。
  `fused_recurrent_gated_delta_rule` 的位置签名是 `(q, k, v, g, gk, gv, beta, ...)`，
  位置传参时 `beta` 会静默落入 `gk`（per-key-dim decay）槽位，导致输出按步指数爆炸
  （本机实测 T=8 时数值达 1e12）。GDN core 初版踩了此坑，已在实现中修复；测试
  `test_chunk_matches_recurrent_loop` 会持续守护这一行为。
- fla 的 chunk 与 fused_recurrent 是不同扫描算法，fp32 下同输入输出差异 ~1e-2 量级
  （正确性测试容差按此设定；softmax 用 SDPA，容差 1e-4）。
- 脚手架必须在调用 core 前把 q/k/v 从 `[B,T,H*D]` 重排为 `[B,T,H,D]`：chunk kernel
  能容忍合并布局，但 fused_recurrent 需要独立 unpack `B,T,H,K,V`。
- softmax 变体的 KV cache 用**静态预分配缓冲**（容量 = prefill S + decode N，runner
  经 `set_kv_capacity` 注入），避免逐步 `torch.cat` 的 O(T²) 拷贝污染 attention 计时。

## 9. Gated KalmaNet (gka) 的接入（首个第三方风格变体）

`gka` 是项目自有变体（非 fla 官方），参考层为 `fla.layers.gka.GatedKalmaNet`，数学上
是「GLA 前置一个 Chebyshev 迭代的岭回归 query 精炼」：`S_t = U_t(H_t+λI)^{-1}`，
`q ← q + α(x_t − q)`，精炼后的 q 进入 GLA 递推。ops 为自有
`fla.ops.gka.gka_chebyshev_solve`（chunk：`gka_chebyshev_gla`；decode：
`torch_decoding_one_step`，均按关键字调用）。

按统一脚手架挂载为 core（`cores/gka.py`），映射决策：

- **默认旋钮取自项目草稿声明的 `GKAConfig`**：`ridge_strength=0.01, num_iter=30,
  chunk_size=64, gate_reduce="mean", alpha="ones"`（fla 参考层用 ridge 0.02、学习式
  a_proj/alpha_proj——均可经 `variant_opts` 配置切换；`alpha="learned"` 时 core 内建
  `alpha_proj`，消费脚手架新增的 `aux["x"]`，脚手架自动为声明了 `hidden_size` 字段的
  变体配置填充该值）。
- core 内部预处理照参考层：q/k l2 归一化（eps 1e-6）、`(sigmoid(beta)+eps)` 同时缩放
  k 与 v、遗忘门 `g = -exp(A_log)·softplus(reduce(g_raw)+dt_bias)`（A_log/dt_bias 为
  core 参数）、`gk=g`（`use_forgetting_gate_kk` 可关）。
- recurrent 模式遇空状态（全新单步 decode）回退到 1-token chunk 路径（数学等价且顺带
  产出状态）；decode 的 Chebyshev 单步走自有 triton kernel。
- 正确性测试覆盖 chunk vs 逐步 recurrent（T=16 时将 `chunk_size` 配为 16 满足整除
  约束）。
