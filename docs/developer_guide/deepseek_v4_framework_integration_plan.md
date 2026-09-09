# DeepSeek-V4-Flash 接入 SGLang-JAX：整模型编译计划

2026-09-06；分支 `integration/deepseek-v4`。
状态：本文件定义的 B1/context=256/4×TPU v5p 有界 milestone 已完成。
原生模型注册/加载、NNX 状态、自有 cache pool、整模型 `ModelRunner` JIT、
逐位数值门、Engine/scheduler smoke 和 A/B profile 均已通过；结果见
[框架接入报告](deepseek_v4_framework_integration.md)。下文保留原始验收计划，
其中列出的长上下文、并发和后续优化不属于本轮完成范围。

后续的 8K chunked-prefill milestone 现也已完成；结果与新的边界见
[8K 集成报告](deepseek_v4_8k_integration.md)。本文件仍按历史原样描述最初的
context=256 阶段。

## 目标与边界

在现有 4×TPU v5p 上，经 SGLang-JAX 正式模型加载、`ModelRunner` 和
prefill/decode 路径执行官方 DeepSeek-V4-Flash。保留原始 FP4/FP8 权重和
compact block scales，继续 online dequant；把 embedding、全部 43 层、
各层 cache 更新、最终 mHC collapse/norm 和输出头纳入一次完整 forward
编译调用。随后完成数值对照，并重测端到端 latency、执行开销和 HBM。

这里的“一次调用”指每步一次整模型顶层 executable 提交，不是只有一个
kernel，也不是 43 层并行。层间依赖和跨芯片通信仍然存在。prefill 与 decode
可有不同 executable，不要求整段多 token 生成都放进一个编译程序。

首个验收配置保持 batch=1、cache capacity=256、EP=4，每芯片 64 个专家，
非专家权重及 attention 复制，与已有 reference 一致。暂不同时引入新的
TP/EP 分布、专家 token compaction、内核 fusion、packed KV、大 batch、
长上下文、MTP/speculative decoding 或通用生产服务性能目标。
未支持的调度模式应明确拒绝，不能静默按单请求执行。

采样可先沿用框架的独立 sampler JIT；把 sampler 合入同一 executable
属于后续独立实验，不是完成上述整模型 forward 的必要条件。

## 已有基线与可复用入口

- [Bring-up 报告](deepseek_v4_v5p_bringup.md)：原始 checkpoint 已完整保存；
  101 项 TPU 回归通过，代表性真实 layer/head 的独立 CPU oracle 已通过。
  完整模型 132-token prefill + 4 步 decode 与 136-token replay 的 logits
  和可见 cache 逐元素一致。该结果不是完整官方 GPU 模型的 bitwise 对照。
- [Profile 报告](deepseek_v4_v5p_profile.md)：未插桩热态 decode 约
  64.2 ms/token；模型运行时 HBM 约 42.6 GiB/芯片。采样运行中的
  22.813 ms/token 是设备程序之外的平均空档，不是可直接兑现的加速量。
- `python/sgl_jax/srt/model_executor/deepseek_v4_reference.py`：保留为可检查的
  数值 reference；目前逐层 JIT、逐层 `block_until_ready`，并未接入正式模型。
- `python/sgl_jax/srt/model_loader/deepseek_v4_checkpoint.py`：已有原始字节和
  scale 读取器；它本身不是 serving model loader 的完整接入。
- `python/sgl_jax/srt/models/registry.py`：通过模型模块的 `EntryClass` 注册。
  `model_loader/loader.py` 调用模型构造和 `load_weights`。
- `python/sgl_jax/srt/model_executor/model_runner.py`：`jitted_run_model` 已包住
  完整 `model(...)`，接收模型状态、`ForwardBatch`、`MemoryPools` 和 logits
  metadata；无需另建一套逐层执行框架。
- `python/sgl_jax/srt/model_executor/compilation_manager.py`：已有形状分桶与
  prefill/decode 预编译。`aot_dispatch.py` 另有可选的低开销 AOT dispatch，
  当前环境变量默认关闭；先验证普通整模型 JIT，再决定是否使用。

以上测试和性能都是接入前基线，不能充当接入后验收结果。复现使用原始
checkpoint revision `60d8d70770c6776ff598c94bb586a859a38244f1`。

## 执行顺序与验收门

### 1. 冻结对照与数据格式契约

- 保存当前脏工作区的可恢复源码快照和已有测试/profile 产物，不覆盖旧结果。
  reference 与新路径分别记录源码 fingerprint、checkpoint revision、运行
  参数及 JAX/libtpu 版本；不能仅用当前分支 commit 代表未提交实现。
- FP4 保留 E2M1 packed bytes；FP8 保留 E4M3FN 原始位模式；两者保留原始
  compact E8M0 scales。整数容器不代表把低比特格式改成 INT8 数值权重。
- 不预展开完整 BF16 权重，不引入 CPU offload，不改官方 checkpoint。
  沿用现有 activation QAT、BF16 RNE 等显式数值边界。

验收：四个原始 vertical-slice case、低比特加载/转换/计算回归，以及代表性
真实权重 oracle 继续通过，既有 oracle 容差不放宽。

### 2. 原生模型注册、加载与状态接入

- 新建 `python/sgl_jax/srt/models/deepseek_v4.py`，按官方 config 的实际
  architecture 注册 `EntryClass`，接通 config、模型构造和 `load_weights`。
  不能借 V3 类名绕过 V4 的 attention/cache 语义。
- 复用原始 checkpoint reader 和已验证的数值算子，适配 NNX 模型状态。
  检查通用 quantization/weight-loading 路径不会二次量化或自动转成 BF16。
  权重作为常驻数组参数传入 JIT，不作为巨大的静态参数或闭包常量捕获。
- 明确 SWA、CSA、HCA 的 window KV、compressed KV、indexer/compressor 状态，
  接入 `ForwardBatch`、`MemoryPools`、request slot 和更新返回协议。
  现有 CSA backend 可复用的部分逐项核对，不能假设它覆盖所有 V4 层语义。
- 保持低比特权重 bytes/scales 的分片对应关系和 EP=4；首阶段不改变当前
  attention 的复制执行方式，以免同时引入新的数值/通信变量。

验收：通过框架入口完整加载 43 层的不同真实权重与每层全部 256 个专家；
逐芯片检查 dtype、shape、sharding 和实际存储量。request reset/reuse 不串
cache；超出支持范围的 batch/context/mode 有显式错误。

### 3. 一次完整 forward 编译与复用

- 将计算组织成可追踪的纯 JAX 数据流，由已有 `jitted_run_model` 包住完整
  forward。不能直接给含 NumPy 转换、Python 状态更新和逐层同步的
  `reference.step()` 套一个 JIT 就宣称完成接入。
- 把所有 43 层和 cache 数组更新放在 JIT 边界内；框架在边界外接收更新后的
  pool 引用是允许的。热路径中不做逐层 `block_until_ready`、`device_get`、
  主机 callback 或日志 I/O；逐层诊断保留在独立调试路径。
- batch shape、cache capacity、层类型等必要结构保持静态；token、position、
  sequence length 等运行值作为数组传入，避免每个位置都触发新编译。
  Python 层循环若在 tracing 时展开可以接受，不要求改成 `scan` 才算整模型。
- 复用 shape buckets 和预编译，记录首次 lowering/compile 时间、编译期 host
  RSS、executable 大小及临时 HBM。避免把所有历史调试中间量作为正式输出。
- 核对 cache donation/aliasing 与框架 pool 替换协议；不得复用已 donate 的
  数组做数值对照，也不得通过复制全部权重来规避状态问题。

验收：通过正式 `ModelRunner` 完成 prefill 与连续 decode；相同 bucket 内
不同 token/position 复用编译结果。trace 证明每步整模型 forward 只有一次
顶层 executable 提交，不再有 43 次主机逐层提交/等待。内部 Pallas 调用和
collective 仍允许多个，不把它们误算成整模型 dispatch 次数。

### 4. 数值对照与框架行为验证

- 先用相同输入 token 和相同初始 cache 对照 reference，逐步比较完整 logits
  及有效 cache/state，避免自由生成中一次选词差异污染后续全部输入。
- 重跑 132-token prefill + 4 步 cached decode 与完整 replay；覆盖压缩边界、
  不同短 prefill shape、cache reset/request slot reuse。再验证 greedy decode
  的 token IDs、EOS 和官方 chat 编码示例。
- bitwise 一致为首要目标。若整图编译改变融合/归约而出现差异，先逐层定位，
  回查量化与 BF16 舍入边界，并用独立 oracle 验证；不能只看生成文本或
  放宽原有容差就宣布通过。任何非 bitwise 的验收必须明确报告误差和依据。
- 调试对照和性能运行分开；测试 reference/new path 时顺序运行 TPU 进程，
  或显式隔离并核对 cache 生命周期，不让两份模型争用同一设备。
- 经框架 worker/scheduler 做有界单请求的完整 prefill/decode smoke，确认
  不依赖直接调用 reference 的旁路。未测试的长上下文/并发能力不宣称支持。

验收：新路径具有独立、可复现的 correctness 报告；已有四个 template/test
case 及全部相关低比特、真实层测试无回归。先过此门，再报告优化收益。

### 5. 同条件 A/B profile

首先比较 A：原 reference；B：框架整模型 JIT。保持 checkpoint、硬件、
sharding、输入 token、cache capacity 和精度相同，采样策略及主机选词时间
保持一致或单列。权重加载、首次编译、cache reset 不混入热态 latency。

记录并归档：

- Prefill latency、decode p50/p95、tokens/s、每步顶层 dispatch 数和重编译数。
- Host dispatch/sync 空档、Attention、routed/shared experts、mHC、输出头、
  collective 及其等待；明确嵌套事件去重、设备时间线与覆盖范围。
- 每芯片实际权重/cache HBM、热态 allocator 与峰值、编译器临时 HBM/VMEM，
  以及编译和运行的 host RSS。不把静态 bytes 当作硬件 traffic。
- Online dequant/conversion/GEMM 的可观测成本与 profiler 插桩扰动。
  整图后 HLO/source 名称可能变化，应更新归因和覆盖检查，不能硬套旧的
  “43 个 layer module”分析规则；未知时间单列而非强行归因。

若 B 仍有显著参数 dispatch 开销，再增加 C：现有 `AotDispatcher`。先核对
其缓存键对 V4 shape/dtype、sharding 和影响编译的静态 metadata 的适用性，
权重重绑定、donation 和 fallback 行为也必须测试。该路径使用 JAX 内部接口，
不能仅设置 `SGLANG_JAX_AOT_DISPATCH=1` 就假定正确或更快。

验收：提供 A/B（必要时 C）的 correctness、完整 trace 和资源对照；明确
空档实际减少多少、瓶颈转移到哪里。无预设 ms/token 或吞吐承诺，不能直接
从旧基线减去 22.813 ms。整模型编译也不自动消除原有 prefill 专家执行冗余。

## 完成定义与交付

- 原生注册/加载与框架执行可复现；正式热路径没有 reference 旁路。
- 官方原始低比特权重常驻，43 层、cache 更新和输出头进入一次 forward 编译调用。
- 有与保留 reference 的数值对照、独立 oracle 回归、框架单请求 decode 测试。
- 有无插桩热态性能、采样扰动、编译成本、HBM/VMEM 的接入前后对照报告。
- 源码与结果同步回本地并保留快照；官方模型继续在独立数据盘保存。
  GCP 登录资料和大型原始 profile 仍留在已忽略的本地目录，不写进公开文档。

完成这一阶段后，再依据新的 profile 决定 sampler fusion、专家调度、低比特
conversion/fusion 或 memory layout 的优先级。当前不以优化单个 kernel 代替
整模型框架接入，也不把“去掉逐层同步”的独立实验当作最终交付。
