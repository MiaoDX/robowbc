# Blog Post Drafts

_Tracks issue [#17](https://github.com/MiaoDX/robowbc/issues/17) and [#47](https://github.com/MiaoDX/robowbc/issues/47)._
_Two drafts: English for Medium / HuggingFace blog; Chinese for Zhihu / WeChat / SegmentFault._

**Publish after:** GEAR-SONIC real model inference works (issue #37) and benchmark numbers are ready.

---

## Status

| Channel | Status | Link |
|---------|--------|------|
| English — Medium draft | [ ] Not published | — |
| English — HuggingFace blog draft | [ ] Not published | — |
| English — Hacker News submission | [ ] Not submitted | — |
| English — ROS Discourse cross-post | [ ] Not submitted | — |
| Chinese — Zhihu draft | [ ] Not published | — |
| Chinese — WeChat article | [ ] Not published | — |

---

## English Draft

### Title

> **Why Every Humanoid Team Rebuilds WBC Deployment from Scratch — and How to Stop**

### Subtitle / deck

> 30+ papers in 2025 shipped bespoke C++ control stacks for the same Unitree G1.
> NVIDIA has four WBC implementations that still aren't unified.
> Here's what a one-interface solution looks like.

---

We surveyed 30+ whole-body control (WBC) papers from 2025 that validated on real Unitree G1
or H1 hardware. Every single one ships its own deployment stack: custom C++ inference code,
custom ZMQ or ROS wrappers, custom TOML/YAML configs. None of them reuse each other's runtime.

This isn't laziness. The researchers are world-class. It's a structural gap: there was no
reusable WBC inference runtime to reach for.

### The scale of the problem

| Metric | 2025 | 2026 (projected) |
|--------|------|-----------------|
| Humanoid robots shipped | 14,500+ | 30,000+ |
| G1/H1 WBC papers | 30+ | 8+ from ICLR 2026 alone |
| Unified open-source WBC runtimes | **0** | — |

NVIDIA alone has four WBC implementations: GEAR-SONIC, Decoupled WBC, HOVER, and WBC-AGILE.
They are not unified. Switching from GEAR-SONIC to HOVER requires touching inference code,
communication code, and config structure. Every team that deploys two models duplicates work.

### What the models all have in common

We analysed the output contract of 10 validated WBC implementations:

| Property | Consensus |
|----------|-----------|
| Output type | Joint position PD targets |
| Control frequency | 50 Hz (SoFTA: 100/50 Hz asymmetric) |
| Input | Proprioception + SE3/velocity command |
| Inference backend | ONNX Runtime or PyTorch |

Every model takes the same input shape and produces the same output shape. The only thing
that differs is the internal model architecture. This uniformity makes a thin abstraction
layer both possible and **natural**.

### The abstraction: one trait, many policies

```rust
pub trait WbcPolicy: Send + Sync {
    fn predict(&self, obs: &Observation) -> Result<JointPositionTargets>;
}
```

`Observation` encodes joint state + IMU + velocity command. `JointPositionTargets` is a
fixed-length vector of position setpoints at 50 Hz. Every WBC model we've seen maps onto
this contract.

robowbc implements this trait for GEAR-SONIC, Decoupled WBC, HOVER, BFM-Zero, WholeBodyVLA,
and WBC-AGILE. Switching models is a config change:

```bash
# GEAR-SONIC on G1
cargo run --release --bin robowbc -- run --config configs/sonic_g1.toml

# HOVER on H1 — same binary, different TOML
cargo run --release --bin robowbc -- run --config configs/hover_h1.toml
```

No code changes. No recompilation.

### Benchmarks

Do not hand-maintain a latency table in this post draft. Lift the exact rows
from `artifacts/benchmarks/nvidia/SUMMARY.md` and the paired JSON artifacts.

Even with measured rows checked in, keep the story anchored to the canonical
cases:

| Case ID | Reader-facing meaning |
|--------|------------------------|
| `gear_sonic_velocity/replan_tick` | Control-loop path at the locomotion replanning boundary |
| `gear_sonic_tracking/standing_placeholder_tick` | Encoder+decoder tracking path |
| `decoupled_wbc/walk_predict` | Movement command through the walk checkpoint |
| `decoupled_wbc/balance_predict` | Near-zero command through the balance checkpoint |
| `gear_sonic/end_to_end_cli_loop` | Whole deployment loop, not just one inference call |

The current committed CPU package measures official and RoboWBC rows for all
canonical GEAR-Sonic and Decoupled cases. If a future rerun blocks an official
row, say that explicitly in the article. A blocked row is more credible than an
approximate comparison.

### Python API

For teams using LeRobot or other Python frameworks:

```python
import robowbc

policy = robowbc.load("gear_sonic", config_path="configs/sonic_g1.toml")
targets = policy.predict(observation)  # → JointPositionTargets at 50 Hz
```

`pip install robowbc` installs the Rust core as a compiled extension — no Rust toolchain
needed at runtime.

### What's next

robowbc is in early development. The current priorities:

1. **GEAR-SONIC real inference** — validate against NVIDIA's reference stack on real G1
2. **Python SDK** — `pip install robowbc` on PyPI
3. **LeRobot backend** — robowbc as an inference backend for `GrootLocomotionController`
4. **Multi-embodiment** — Booster T1, Fourier GR-1, AGIBOT X2 via config-driven robot profiles

The goal is for robowbc to be to WBC what LeRobot is to manipulation: one interface, every
policy, config-driven switching.

**GitHub:** https://github.com/MiaoDX/robowbc

---

## Chinese Draft

### 标题

> **为什么每个人形机器人团队都在重复造全身控制部署的轮子——以及如何终结这一现状**

### 副标题

> 2025 年超过 30 篇论文在 Unitree G1/H1 上做了全身控制实验，每篇都自己写了 C++ 部署代码。
> NVIDIA 有四套 WBC 实现，至今没有统一。这是一个可以解决的问题。

---

我们调研了 2025 年 30 多篇在真实 Unitree G1 或 H1 上验证的全身控制（WBC）论文。每一篇都
自带了一套部署栈：自定义 C++ 推理代码、自定义 ZMQ 或 ROS 封装、自定义配置文件格式。没有任
何两篇论文复用同一套运行时。

这不是研究者的问题。是结构性的空缺：**根本没有可以直接拿来用的统一 WBC 推理运行时**。

### 问题的规模

| 指标 | 2025 年 | 2026 年（预测） |
|------|---------|----------------|
| 人形机器人出货量 | 14,500+ | 30,000+ |
| G1/H1 WBC 论文 | 30+ | 仅 ICLR 2026 就有 8+ |
| 统一开源 WBC 运行时 | **0** | — |

NVIDIA 自己就有四套 WBC 实现：GEAR-SONIC、Decoupled WBC、HOVER、WBC-AGILE。它们之间
没有统一。从 GEAR-SONIC 切换到 HOVER，需要改推理代码、通信代码和配置结构。每个需要
同时部署两个模型的团队都在重复同样的工作。

### 这些模型的共同点

我们分析了 10 个经过真机验证的 WBC 实现的输出契约：

| 属性 | 共识 |
|------|------|
| 输出类型 | 关节位置 PD 目标 |
| 控制频率 | 50 Hz（SoFTA：100/50 Hz 上下肢异步） |
| 输入 | 本体感知 + SE3/速度指令 |
| 推理后端 | ONNX Runtime 或 PyTorch |

所有模型的输入输出形状完全一致。唯一的差异是内部模型架构。这种一致性使得一个薄抽象层
既**可行**又**自然**。

### 抽象层：一个 Trait，支持所有策略

```rust
pub trait WbcPolicy: Send + Sync {
    fn predict(&self, obs: &Observation) -> Result<JointPositionTargets>;
}
```

`Observation` 包含关节状态 + IMU + 速度指令。`JointPositionTargets` 是 50 Hz 的位置
目标向量。我们见过的每个 WBC 模型都能映射到这个契约。

robowbc 为 GEAR-SONIC、Decoupled WBC、HOVER、BFM-Zero、WholeBodyVLA 和 WBC-AGILE
实现了这个 trait。切换模型只需改配置文件：

```bash
# G1 上运行 GEAR-SONIC
cargo run --release --bin robowbc -- run --config configs/sonic_g1.toml

# H1 上运行 HOVER —— 同一个二进制，换一个 TOML
cargo run --release --bin robowbc -- run --config configs/hover_h1.toml
```

不改代码。不重新编译。

### 性能基准

不要在这份草稿里手工维护延迟数字表。直接使用
`artifacts/benchmarks/nvidia/SUMMARY.md` 和配套 JSON artifact 中的最新数字。

即使现在已经有测量结果，这一节也仍然要围绕标准 case 语义来写：

| Case ID | 面向读者的含义 |
|------|----------------|
| `gear_sonic_velocity/replan_tick` | locomotion 重规划边界上的控制环路径 |
| `gear_sonic_tracking/standing_placeholder_tick` | 编码器+解码器跟踪路径 |
| `decoupled_wbc/walk_predict` | 行走指令命中 walk checkpoint |
| `decoupled_wbc/balance_predict` | 零速度附近命中 balance checkpoint |
| `gear_sonic/end_to_end_cli_loop` | 真正的部署控制环，而不是单次推理 |

当前提交的 CPU 对比包已经包含 GEAR-Sonic 和 Decoupled 的官方 / RoboWBC
测量行。如果未来某次重跑里 NVIDIA 官方路径被阻塞，就在文章里明确写出阻塞
原因。比起模糊的近似对比，明确的 blocked row 更可信。

### Python API

对于使用 LeRobot 或其他 Python 框架的团队：

```python
import robowbc

policy = robowbc.load("gear_sonic", config_path="configs/sonic_g1.toml")
targets = policy.predict(observation)  # → 50 Hz 关节位置目标
```

`pip install robowbc` 将 Rust 核心安装为编译好的扩展模块——运行时不需要 Rust 工具链。

### 下一步

robowbc 目前处于早期开发阶段。当前优先级：

1. **GEAR-SONIC 真实推理** — 在真实 G1 上与 NVIDIA 参考栈对比验证
2. **Python SDK** — `pip install robowbc` 发布到 PyPI
3. **LeRobot 后端** — 将 robowbc 作为 `GrootLocomotionController` 的推理后端
4. **多机体支持** — 通过配置驱动的机器人配置文件支持 Booster T1、Fourier GR-1、AGIBOT X2

目标是让 robowbc 成为 WBC 领域的 LeRobot：一个接口，所有策略，配置切换。

**GitHub:** https://github.com/MiaoDX/robowbc

---

## Publishing checklist

When ready to publish (after GEAR-SONIC demo + benchmarks):

- [ ] Replace the case-registry table above with the latest rows from `artifacts/benchmarks/nvidia/SUMMARY.md` before publishing
- [ ] Add screenshots or terminal output showing policy running
- [ ] English: publish on Medium, then cross-post to HuggingFace blog
- [ ] English: submit to Hacker News (`Show HN: robowbc — unified WBC inference runtime for humanoid robots`)
- [ ] English: post on ROS Discourse (Rust/humanoid threads)
- [ ] English: post on r/robotics and r/rust
- [ ] Chinese: publish on Zhihu
- [ ] Chinese: post in WeChat robotics communities (机器人学 公众号 groups)
- [ ] Tag GEAR-SONIC / HOVER / BFM-Zero authors on social media
