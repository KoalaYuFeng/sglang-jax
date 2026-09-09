# DeepSeek-V4-Flash 原生框架接入

> 历史 B1 验收记录。当前分支正在迁移到原生多请求分页实现，见
> [正式 serving 集成](deepseek_v4_serving_integration.md)。下文的 page-size=1
> 命令与旧报告不能作为新入口的验收依据；新入口使用 128-token pages。

更新于 2026-09-07；分支 `integration/deepseek-v4`，未提交工作区实现。

## 当前状态

本文件最初定义的 **B1、context=256、4×TPU v5p 有界 milestone 已完成**，
后续 B1 8K chunked-prefill milestone 也已完成。已实现
原生注册、checkpoint loading、NNX 动态权重、`ForwardBatch` metadata、
request-owned cache pool，以及既有 `ModelRunner.jitted_run_model` 下的一次
整模型 forward。

验收结果：

- 27 项框架契约测试在四设备 CPU 和 4×TPU v5p 分别通过；包含整图调用、
  cache donation/update、sampler sharding、静态 metadata 和不安全 AOT 回退。
- 最终 TPU 合并回归为 **141 passed，0 failed**；覆盖框架测试以及
  mHC/HCA/CSA、低比特、KV update、TP/attention 和 vertical-slice 路径。
- 真实 43 层的 132-token prefill + 4 步 decode，其完整 logits 和所有层
  cache/state 每一步均与保留 reference **逐位一致**。136-token replay 的
  最终 logits 与可见 cache 也逐位一致。
- 官方 chat fixture 经框架 sampler 输出内部 token `[22, 1]`（“4” + EOS）。
  经 Engine/tokenizer/scheduler/ModelWorker/detokenizer 的公开 API 返回可见
  `output_ids=[22]`、`completion_tokens=2`，finish reason 为 EOS stop。
- Engine 顺序执行 132-token + 4-token 固定生成，并再次执行 chat；重复请求
  cache miss 为 0，证明单 request slot 的 reset/reuse 没有串 cache。
- 同条件热态 decode p50 从逐层 reference 的 64.77 ms 降至整图的
  50.14 ms（19.94 token/s）；XPlane 在全部 8 条 TensorCore 时间线上确认
  每 token 只有一次整模型 module 提交。详细数据见
  [v5p profile](deepseek_v4_v5p_profile.md)。
- 8186-token 真实 Engine 请求由 scheduler 自动拆成 `63×128 + 122`，
  greedy 输出与直接 ModelWorker oracle 完全一致；全 128-token chunk 不随
  position 重编译，steady throughput 约 46.28 token/s，8K 热态 decode
  约 64.81 ms/token。完整证据见
  [8K 集成报告](deepseek_v4_8k_integration.md)。

## 实现边界

- 模型入口：`python/sgl_jax/srt/models/deepseek_v4.py`。复用 reference 中的
  纯数值函数，不调用 `DeepSeekV4Reference.step()` 或其逐层 JIT runner。
- embedding、43 层、各层 window/compressed KV 和 compressor state 更新、
  mHC 输出 collapse、norm、输出头都在同一个整模型 JIT 内。Python 层循环
  只在 tracing 时展开；热路径没有逐层主机同步。每层 BF16 stream 后保留
  `optimization_barrier` 作为与逐层 reference 对齐的数值边界，但它不产生
  新 executable 或主机等待。
- 原始 FP4 E2M1 packed bytes、FP8 E4M3FN bytes 和 compact E8M0 scales
  常驻 HBM。沿用已验证的 tile 级 VMEM online dequant，不增加完整 BF16
  权重副本。通用 FP8 quantization 路径被明确绕过，HF 格式 metadata 保留。
- EP=4，每芯片持有每层 64 个专家；非专家权重、attention/cache 复制。
  与原 numerical reference 相同，不同时改变 sharding 或数值算子。
- 框架输入的 `P("data")` 显式转换为 attention 的 `P()`；最终 logits
  转为框架 sampler 所需的 `P("data", "tensor")`。这些转换在 JIT 内。
- cache 返回值通过 `MemoryPools.replace_all` 更新；下次调用可 donation。
  新请求在 prefill executable 内重置所有 cache 和 compressor scratch。

当前支持范围为 B1、TP=4/DP=1/EP=4、BF16 activations、context 为 128 的
倍数且不超过 8192。context 大于 256 时必须启用 128-token chunked prefill；
禁用 prefix/radix reuse、overlap、speculative/MTP、LoRA 和 PD。不能据此推断
已支持更长上下文、并发服务或 prefix sharing。输入 logprobs/hidden-state
capture 尚未支持；采样是独立的框架 sampler JIT。

Prefill 的所有完整 128-token chunk 共用一个 executable，absolute position
保持动态。最终 partial chunk 仍会按**真实输入长度**静态特化，即使框架
外层使用 128 padding bucket；这样不会让 padding 污染有状态 compressor，
但每种新 tail 长度首次出现时仍可能编译。Decode 的 token/position 是动态
数组，长度固定为 1，不按每个 position 编译。

现有可选 `AotDispatcher` 的 key 只覆盖 leaf shape/dtype，不能区分 V4 的
静态 forward mode/真实 prefill 长度，因此 V4 会明确回退普通整模型 pjit，
即使设置 `SGLANG_JAX_AOT_DISPATCH=1`。尚未验证 AOT 性能收益。

## 可复现验证入口

先顺序运行 numerical gate；此进程退出后再启动 Engine gate，不能同时让
两份完整模型占用同一组 TPU。

```bash
python scripts/run_deepseek_v4_framework.py \
  --checkpoint /path/to/original/checkpoint \
  --reference-report /path/to/deepseek-v4-full-inference-warm-chat.json \
  --output /path/to/new-framework-run \
  --warm-repeats 3 --profile

python scripts/run_deepseek_v4_serving_smoke.py \
  --framework-report /path/to/new-framework-run/report.json \
  --output /path/to/new-serving-run

python scripts/run_deepseek_v4_8k_worker.py \
  --checkpoint /path/to/original/checkpoint \
  --output /path/to/new-8k-worker-run

python scripts/run_deepseek_v4_8k_engine.py \
  --worker-report /path/to/new-8k-worker-run/report.json \
  --output /path/to/new-8k-engine-run
```

前者用正式 `ModelWorker`、request/token allocator 和 sampler，比较完整
logits/cache，执行 replay 和 chat fixture，再做同条件 A/B。后者经 Engine、
tokenizer、scheduler 和 ModelWorker，验证顺序请求、EOS 和 slot reuse。
报告记录 numerical/framework fingerprints；只有全部脚本验收通过才写
`complete=true`。输出目录必须是新目录，失败报告不覆盖。

本轮完成报告位于本地忽略目录：

- `deepseek-v4-framework-20260906-run04-layer-barrier/`：完整 numerical gate、
  A/B、compiler memory/HLO、reference/framework XPlane 与 XProf 导出。
- `deepseek-v4-serving-20260906-run02/`：完整 Engine/scheduler gate。
- `v5p-test-20260906T154713Z.log` 及同轮 JUnit XML：context=256 阶段的
  128 项回归。
- `deepseek-v4-8k-worker-20260907-run03/` 与
  `deepseek-v4-8k-engine-20260907-run03/`：8K worker/Engine 对照 gate。
- `deepseek-v4-8k-worker-20260907-run04-profile/` 与
  `deepseek-v4-8k-worker-20260907-run05-hot-profile/`：early/late prefill 和
  8K 热态 decode 的 XPlane/XProf 结果。
- `pytest-20260906T181942Z.xml`：141 项最终 TPU 回归。

原始数据和大体积 profile 位于本地忽略目录 `GCP_login/results`，代码通过
本地快照同步到 spot。官方模型仍保存在独立数据盘，不写入 Git。

最初整图目标见[原计划](deepseek_v4_framework_integration_plan.md)，8K 扩展见
[8K 集成报告](deepseek_v4_8k_integration.md)。
