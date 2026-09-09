# DeepSeek-V4-Flash：4×TPU v5p 实测 profile

2026-09-06。结论：完整模型运行时约 **42.8 GiB HBM/芯片**；逐层 reference
的热态单请求 decode p50 为 **64.77 ms/token**，接入 `ModelRunner` 的 43 层
整模型 JIT 后为 **50.14 ms/token（19.94 token/s）**，延迟降低 22.6%。主要
收益来自消除逐层主机提交/同步；设备计算本身基本未变。专家执行、Attention、
all-reduce 和低比特转换仍是下一阶段重点，不能仅用 HBM 带宽解释。

## 测量范围与可比性

- `integration/deepseek-v4` / `d5e58ee6` 加现有本地适配；本轮未修改数值实现。
- 官方 checkpoint revision：`60d8d70770c6776ff598c94bb586a859a38244f1`。
- 数值源码 fingerprint：`403f78e7c27a7a6ff7acd5cfe1790a5b2a4e6e9ad3801395ab30ea792eacb411`。
- 43 层、每层全部 256 个专家常驻；EP=4，每芯片 64 个专家；非专家权重复制。
- 4 个物理芯片 / 4 个 JAX device / 8 个 TensorCore，不能把 8 条设备时间线相加当延迟。
- Batch=1，132-token prefill，随后 4 个 cached decode，位置 132–135；cache capacity=256。
- JAX/jaxlib 0.11.1，libtpu 0.0.46.1；CPU 分析使用独立的 XProf 2.23.1 环境。
- 所有用于报告的运行均通过固定生成 token 检查，且每个 decode 的完整 logits
  与同进程未插桩控制运行逐元素完全一致。已有完整模型 replay/真实 layer oracle
  见 [bring-up 报告](deepseek_v4_v5p_bringup.md)。
- 这是 correctness-first reference runner，不是生产 serving、大 batch、长输出或长上下文 benchmark。

三种数字严格分开：host/device trace 的执行时间是实测；JAX allocator 的 HBM
是分配器快照；XLA/XProf memory viewer 的临时 HBM、VMEM、FLOPs/bytes 是编译器
静态信息。静态 bytes 不等于物理 HBM/VMEM traffic。

## 整模型框架 JIT：correctness 与 A/B

框架运行的 numerical source fingerprint 仍为
`403f78e7c27a7a6ff7acd5cfe1790a5b2a4e6e9ad3801395ab30ea792eacb411`；
新增框架源码 fingerprint 为
`b195d1c6cb8313ac70cbe11856e0e49815c7373bdb4850eb7cd471ee34bd9732`。
官方 43 层 checkpoint 通过正式 `ModelWorker` 加载，原始 FP4/FP8 bytes 和
compact E8M0 scales 保持不变。

132-token prefill 和随后 4 步 teacher-forced decode 的完整 logits、所有层
window/compressed KV 与 compressor state 均与逐层 reference **逐位一致**。
136-token replay 的最终 logits 与可见 cache 也逐位一致；官方 chat fixture 经
框架 greedy sampler 输出 `[22, 1]`（“4” + EOS）。为保留逐层 reference 的
BF16 数值边界，整图在每层 stream 后使用 `optimization_barrier`；它不产生新的
executable 或主机等待。

同进程、同权重、同输入的无插桩热态 A/B 如下。两条路径均包含 host batch
准备、提交、ready wait 和 host argmax；框架 device sampler 另行做 correctness，
没有只计入其中一侧。Prefill 数值取 3 次中位数，decode 取 12 步。

| 路径 | 132-token prefill p50 | Decode p50 | Decode p95 | Decode 吞吐 |
| --- | ---: | ---: | ---: | ---: |
| 逐层 reference | 2686.01 ms | 64.77 ms | 69.07 ms | 15.44 token/s |
| 框架整模型 JIT | 2700.06 ms | 50.14 ms | 51.64 ms | 19.94 token/s |

框架 decode p50 延迟降低 22.6%，等价吞吐提高约 29.2%；prefill 在这组 B1
短序列测量中慢约 0.5%，没有宣称 prefill 加速。首次 132-token prefill 编译约
88.8 秒，首次 decode 编译约 160.5 秒；这些冷启动成本不计入热态表格。

4-step XPlane 时间线比无插桩控制组稍慢，但适合解释差值。每条 TensorCore
时间线均只有 4 个 `jit_jitted_run_model`，即每 token 一次整模型 executable；
逐层 reference 则有 172 个 layer module（43×4）和 4 个 head module。
两边都完整观察到 1376 次专家 all-reduce（43 层×8 TensorCore×4 step）。

| 时间线区块 | 逐层 reference ms/token | 整模型 JIT ms/token |
| --- | ---: | ---: |
| 设备程序之外的空档 | 24.480 | 7.239 |
| 设备 module union | 44.538 | 44.079 |
| Trace host wall | 69.018 | 51.318 |

设备外空档减少 17.24 ms/token（约 70.4%），设备 module 时间只变化约
-0.46 ms/token。整图没有消除模型数学工作，也没有把 43 层并行化。

框架时间线内每 token 的 measured stage partition 为：Attention 14.417 ms、
routed experts 14.856 ms、专家 all-reduce（含等待）8.562 ms、shared expert
与 MoE combine 1.734 ms、mHC 1.537 ms、LM head 0.418 ms、router 0.304 ms、
layer norm 0.200 ms，另有 2.051 ms 尚未可靠归因。区块已去除嵌套重复，
不应再相加到 51.318 ms 之外。

整模型 decode executable 的编译器内存报告：参数 42.522 GiB、临时 HBM
181.80 MiB、输出 24.20 MiB，其中 24.07 MiB 可 alias；生成代码 43.86 MiB。
实测 allocator 在加载后为 42.5229 GiB/芯片，热态当前 42.8071 GiB，进程
累计峰值 42.8832 GiB，对应上限 95.7275 GiB。整图未产生完整 BF16 权重副本。

XProf roofline 将总体标为 HBM-bound，但其 Program 行只给约 64 GiB/s 的
编译器估计 HBM 带宽；custom Pallas 调用的物理 bytes/counters 覆盖不完整，
不能把这个数字当作实测带宽利用率。真实优化判断仍以时间线、显式 A/B 和后续
硬件 counters 为准。

## HBM 与主机内存

以下为接入前、未加内部插桩的逐层 reference 完整模型运行，单位 GiB
（2^30 bytes）。接入后的 allocator 数值见上一节。

| 每个物理芯片 | GiB |
| --- | ---: |
| 全部权重和初始 cache 加载后，allocator 实测 | 42.5228 |
| 热态运行后，allocator 当前值 | 42.6040 |
| 该进程累计 allocator 峰值 | 42.6801 |
| JAX 报告可用上限 | 95.7275 |
| 热态剩余容量 | 53.1235 |

四芯片热态合计约 170.42 GiB。此前更广的 chat/replay 运行峰值约 42.73 GiB/芯片；
不同形状/诊断对象存活期会改变峰值，并不矛盾。进程退出后权重释放；这些数值
描述模型驻留时的占用，不意味着脚本结束后仍有常驻推理服务。

下表是逐个读取 `addressable_shards` 得到的**物理数组存储量**，不是参数文件大小。
复制的张量在每个芯片都占空间。

| 存储项目 | GiB/芯片 |
| --- | ---: |
| Routed expert 原始 packed FP4 | 32.2500 |
| FP4 原始 E8M0 block scales | 2.0156 |
| Attention/shared expert 原始 FP8 | 5.4551 |
| FP8 原始 block scales | 0.0003 |
| Embedding | 0.9863 |
| LM head 与 final norm | 0.9866 |
| 其余原始 BF16 | 0.6615 |
| 其余原始 FP32 | 0.1313 |
| 路由整数表 | 0.0087 |
| Window KV | 0.0105 |
| Compressed KV | 0.0016 |
| Compressor 工作缓存 | 0.0114 |
| 数组总量 | 42.5189 |

数组总量和 allocator 相差约 4 MiB，主要是分配/对齐等统计口径。
本实现 KV 是 BF16，并非 packed FP4/FP8 KV；权重仍保持原始低比特格式。
单请求、短 cache 下这些缓存只有约 24 MiB/芯片，不能外推到长上下文或多请求。

未插桩进程的 host RSS：加载后 8.92 GiB，热态 9.00 GiB，导出 trace 后 11.63 GiB。
重型 LLO/内部插桩导出曾把 host 峰值推到约 52.27 GiB，这是 profiler 的额外开销，
不是模型 CPU 权重 offload。

## 热态端到端与逐层时间

未插桩的独立复测：prefill 约 2.685 s，decode 中位数约 63.9–64.4 ms；
decode 采样前/后控制组分别为 64.190 / 64.214 ms。模型加载约 100 秒，首次
prefill/decode 的编译不计入热态性能。

下表来自未插桩 `compiler_export_control`：prefill 3 次、decode 12 步。
每层 host 计时包含 dispatch、执行和该层 `block_until_ready`，不是纯设备核时间。

| 层类型 | 层数 | Decode 平均 ms/层 | 132-token prefill 平均 ms/层 |
| --- | ---: | ---: | ---: |
| SWA（0–1） | 2 | 1.328 | 61.771 |
| Hash-routed CSA（2） | 1 | 1.568 | 58.623 |
| HCA（3,5,…,41） | 20 | 1.356 | 65.538 |
| CSA（4,6,…,42） | 20 | 1.458 | 59.354 |

逐层原始记录全部保存在 `report.json` 的 `layer_host_timings`，包括每个 phase、
position、layer 和 tokens。不要把 warmup/compile 行混进平均值。

## Prefill：完整 XPlane 的分块统计

轻量采样的 132-token prefill 为 **2690.752 ms**，相对于约 2685 ms 的控制组，
采样扰动很小。完整 XPlane 的 HLO 数据包含 **344 次 all-reduce（43 层×8
TensorCore）**，4 种 layer executable 的计数与模型一致，head 也已执行。

Chrome viewer 默认的事件上限使它只导出了部分时间线；因此本表不使用该 JSON，
而是从完整 XPlane 的 `hlo_stats` 取 **self-time**，除以 8。不是将嵌套 total-time
相加，也不是取一个芯片的值再乘 4。`hlo_breakdown.json` 可复核这些计数。

| 区块 | ms/次 132-token prefill，TensorCore 平均 |
| --- | ---: |
| Routed experts 本地执行 | 2019.088 |
| 专家 all-reduce（含等待） | 306.018 |
| Attention | 292.713 |
| Shared expert 与 MoE 合并 | 22.867 |
| mHC | 5.335 |
| Router | 1.959 |
| LM head | 0.421 |
| 未归因 HLO | 12.335 |
| Host/设备间隙及其他未计入 HLO 的时间 | 30.016 |
| 合计 | 2690.752 |

Routed-expert 路径加上求和约占总时间 **86.4%**。其中 FP4 online-dequant matmul
调用本身约 **1970.726 ms**，含 dequant/activation conversion/GEMM 等，不能叫
纯 MXU 计算时间。Attention 的 FP8 matmul 子集约 117.657 ms，shared FP8 约
22.695 ms。当前专家路径没有按 expert 压紧 token；一个专家被激活时，对整个
token batch 做矩阵计算再用路由权重组合。这个 correctness baseline 的执行形状
有助于解释 prefill 的成本，但本轮没有改变它。

## Decode：设备时间线的分块归因

这份采样**没有内核内部插桩**。4 个 decode 的 host trace 平均为 67.395 ms/token，
比关闭采样的控制组约慢 5%。8 条 TensorCore 时间线均覆盖 4×43 层和 4 次 head。

下表先对每条时间线去除嵌套重复计时，再按 TensorCore 与 token 取平均；合计
67.395 ms。归因使用匹配 fingerprint 的 HLO source stack，融合算子按其 source
归属分类，所以不是将数学模块独立运行所得的时间。

| 区块 | ms/token | 说明 |
| --- | ---: | --- |
| 设备程序之外的空档 | 22.813 | Host dispatch、逐层同步、队列/握手等；不是纯 Python 计算时间 |
| Attention | 14.997 | 含投影、压缩/indexer、attention、转换与相关布局操作 |
| Routed experts 本地执行 | 14.537 | FP4 内核、专家选择/循环、局部复制与其他操作；不含下面的 psum |
| 专家跨卡 all-reduce | 8.506 | **包含到达等待与负载不均衡**，不是纯网络传输 |
| Shared expert 与 MoE 合并 | 1.579 | FP8 shared expert 路径等 |
| mHC | 1.488 | Pre/post、Sinkhorn 及相关融合 |
| Router | 0.269 | 打分与选专家 |
| LM head | 0.444 | 完整词表投影等 |
| 设备程序内尚未归因 | 2.762 | 部分布局/运行时操作与 trace 间隙，未强行分配给某个模块 |

每芯片的专家工作量并不相同：TensorCore 平均 all-reduce 时间范围约
6.63–9.68 ms/token，本地 routed-expert 路径约 13.74–15.84 ms/token。
更早到达 psum 的芯片会等其他芯片；不能据此声称 ICI 带宽打满。

逐层 reference 每 token 有 54 次设备程序调用：43 个 layer、1 个 head、10 个
准备/小操作。它在每层结束都同步；整模型 JIT 的对照结果见前文。

### 低比特 GEMM 内核的完整调用时间

下面是上一表的**子集**，不能再加到总时间上。内核时间同时含计算、online
dequant、activation conversion 和对应的数据移动，并非纯 MXU GEMM 时间。

| 设备内核调用汇总 | ms/token，TensorCore 平均 |
| --- | ---: |
| Attention FP8 online-dequant matmul | 7.906 |
| Routed expert FP4 online-dequant matmul | 7.008 |
| Shared expert FP8 online-dequant matmul | 1.488 |
| Attention 原始 BF16 matmul | 0.044 |
| mHC collapse-pre custom kernel | 0.344 |
| mHC Sinkhorn custom kernel | 0.176 |

例如 routed-expert 的 14.54 ms 中，FP4 matmul 调用仅约 7.01 ms；剩余还包括
循环、gather/格式整理、激活等。因此不能将全部 MoE 时间称为 FP4 解码成本。

## 内部插桩：conversion / dequant 的细分

另一次运行给低比特内核内部加 `jax.named_scope`，保留所有算术与官方 checkpoint。
插桩前 decode 控制组为 64.220 ms，插桩后为 70.422 ms；采样区间平均为
73.761 ms。**插桩改变了调度/执行开销，以下数值只能解释该插桩运行，不能直接
从原始 64 ms 中减去，更不是可实现的加速承诺。**

| 内部 region | 实测 ms/token，TensorCore 平均 |
| --- | ---: |
| Weight scale broadcast/expand | 3.929 |
| FP4 nibble unpack/数值解码 | 3.326 |
| FP8 权重数值解码 | 3.279 |
| Activation FP8 quantize/dequantize | 2.540 |
| E8M0 scale 数值解码 | 0.220 |
| 上述不重叠 region 合计 | 13.295 |

这些 region 在 VMEM 内执行。保持原始 checkpoint 并不代表没有 conversion；
当前路径每个输出 tile 都在线解码，而且 scale expand 和 A8 activation QAT
也有可观成本。M=8、N=128、full-K 的 baseline 会重复处理某些 activation/scales。
未被包在这些 region 内的 cast/multiply/数据搬运不包含在上表中。

## 编译器内存视图：HBM 中间量与 VMEM

以下数字来自未插桩 executable，不是 hardware traffic counters。参数约
0.95–0.98 GiB/层/芯片是已经常驻的权重参数，不能再次加到模型总 HBM 上。

| Executable | Decode 临时 HBM MiB/芯片 | Prefill 临时 HBM MiB/芯片 | Decode VMEM heap peak MiB |
| --- | ---: | ---: | ---: |
| SWA | 34.820 | 65.992 | 47.866 |
| Hash CSA | 36.069 | 95.225 | 47.925 |
| HCA | 36.005 | 50.060 | 46.422 |
| CSA | 36.037 | 61.950 | 46.480 |
| Head | 0.156 | 0.156 | 0.512 |

Decode layer 输出约 0.57–1.07 MiB/芯片，prefill 输出约 35.91–40.33 MiB/芯片。
reference 还返回诊断中间张量；这些也是输出分配的一部分，未假装全部消失。

XProf VMEM viewer 同时给出 layer `totalBufferAllocationMib` 约 62.46–63.93 MiB，
以及独立的 scoped allocation 信息。它们是不同的编译器统计口径，不能把
heap peak、total allocations 和 scoped peak 相加，也不能当作实际 VMEM traffic。
v5p 每 TensorCore 的 VMEM 为 64 MiB，详见 [JAX hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)。

逻辑 dequant BF16 tile 为 `128×K×2` bytes，即 K=4096 时 1 MiB、K=8192 时
2 MiB，此外还有 packed 输入、scale、FP32 中间值和编译器缓冲。没有常驻完整
BF16 专家张量；若把所有 routed experts 预展开为 BF16，仅这部分就约 129 GiB/芯片。

## 暂不能从这次 trace 得出的数字

- 精确的物理 HBM/VMEM 读写字节数、带宽利用率、纯 MXU/VPU busy 百分比。
  本次 XProf 导出未提供可用的相应 counter 数据；不能将显示的 0 当作真实零利用率。
- 带 custom-region/LLO 标记的 XProf overview 漏算部分 custom execution，甚至把
  它们计入 idle。报告使用原始时间线，而不是该 overview 的 idle 百分比。
- `Tensor Core` 行里的 1 ps 事件是指令/标记，不是可累加的矩阵运算持续时间。
- 最初的重型 prefill Chrome JSON 仅覆盖约前 4 层；轻量运行的原始 JSON 还出现
  文件内容未闭合问题，已保留原文件，并从 XPlane 重新导出。重新导出的 viewer
  JSON 仍受默认 100 万事件限制，只覆盖部分层；coverage 会报 false，不用于
  完整模型分块。上面的 prefill 表取自完整 XPlane HLO 统计，不受 viewer 裁剪影响。
  事件上限实现见 [XLA trace exporter](https://github.com/openxla/xla/blob/main/xla/tsl/profiler/convert/xplane_to_trace_events.cc)。
- 一次额外 kernel-scope wrapper 实验在编译期间因 `functools.partial.__name__`
  不存在而失败，未产生有效采样；该临时代码已撤回，失败目录保留但不进入统计。

因此，之前只算权重搬运得到约 3.4 ms 的 roofline 是一个极乐观的下限，并没有
计入上述同步、控制流、转换、重复读写等成本。Google 标称 HBM 带宽为
2.765 TB/s/芯片，不能拿权重字节数除以 64 ms 当成实测总带宽利用率。
见 [Google TPU v5p 规格](https://docs.cloud.google.com/tpu/docs/v5p)。

## 下一轮实验建议

首阶段[V4 原生框架接入](deepseek_v4_framework_integration.md)和整模型 JIT
已经完成 ModelWorker correctness/profile 门；以下是后续实验顺序，不是本报告
已经测得的额外收益。

1. 将剩余约 7.24 ms/token 的设备程序外时间继续拆成框架 batch 准备、参数
   dispatch、sampler 和 ready/host argmax。现有 AOT key 不包含 V4 的静态
   forward metadata，当前明确回退普通 pjit；先修正 cache key、donation 和
   weight-rebind 测试，再单独 A/B，不能直接把全部空档当成可消除 Python 开销。
2. 基于整模型 profile，检查本地专家循环、packed-weight gather/layout
   和跨芯片负载均衡；先区分
   实际计算与等待，再判断是否改调度或 sharding。
3. 对相同真实矩阵比较 scale layout、重复 A8 QAT 与 online-dequant fusion；
   使用同 tile BF16 控制组，并把存储/搬运变化一起计入，不把延迟差全叫 dequant。

mHC 当前约 1.5 ms，不是优先级最高的整体瓶颈；完整模型 HBM 容量目前也不是。

## 复现与本地产物

Reference profiler：`scripts/profile_deepseek_v4_reference.py`。框架 gate/profile：
`scripts/run_deepseek_v4_framework.py`。分析器：
`scripts/analyze_deepseek_v4_profile.py`。区间去重、归因与 dispatch coverage
已有 17 个 CPU 单元测试，并支持 early/late 8K chunk 与热态 decode 标记。
8K 的新 profile 和优化优先级见
[8K 集成报告](deepseek_v4_8k_integration.md)。

在 TPU 的现有模型环境中，指定已验证的 checkpoint 和完整模型 correctness report：

```bash
python scripts/profile_deepseek_v4_reference.py \
  --checkpoint "$V4_CHECKPOINT" \
  --correctness-report /home/koala/sglang-jax-results/deepseek-v4-full-inference-warm-chat.json \
  --output /home/koala/sglang-jax-results/v4-profile-new-run \
  --unannotated-only --decode-only
```

完整 prefill 使用 `--unannotated-only --prefill-only`。内部转换分析去掉
`--unannotated-only`，设置 `LIBTPU_INIT_ARGS=--xla_enable_custom_call_region_trace=true`。
输出目录必须是新的，避免覆盖旧采样。`--compute-trace` 可额外采一芯片的
`TRACE_COMPUTE_AND_SYNC`，其更大的测量扰动也须单独报告。

XProf 分析使用独立环境，避免改变模型的 JAX/protobuf 等依赖；纯 JSON 时间线
分析也可以在本地用 stdlib 执行：

```bash
python scripts/analyze_deepseek_v4_profile.py \
  --profile GCP_login/results/deepseek-v4-profile-unannotated-20260906 \
  --raw-only
```

本地私有目录（Git ignored）保留原始 `.xplane.pb`、`.trace.json.gz`、HLO、
`device-memory.pprof`、XProf JSON 导出、`workload_breakdown.json` 及所有逐层计时：

- `GCP_login/results/deepseek-v4-profile-unannotated-20260906/`：decode 主报告。
- `GCP_login/results/deepseek-v4-profile-20260906/`：内部转换及 compute/sync 采样。
- `GCP_login/results/deepseek-v4-profile-prefill-20260906/`：轻量完整 prefill 采样。
- `GCP_login/results/deepseek-v4-framework-20260906-run04-layer-barrier/`：
  整模型 correctness、A/B、编译器 HLO/memory 和两条完整 decode trace。

模型原始 checkpoint 留在独立数据盘；本轮没有预转换/重写它，也没有提交或推送 GitHub。
