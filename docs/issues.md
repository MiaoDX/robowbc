# RoboWBC — Issue Tracker (Issue-Driven Development)

> 按 Phase 分组，每个 issue 独立可执行，标注依赖关系。
> 在 GitHub 上创建时建议使用对应的 label: `phase-1`, `phase-2`, `phase-3`, `infra`, `model`, `docs`.

---

## Phase 0: 项目基础设施

### Issue #1: 初始化 Rust cargo workspace 项目骨架

**Labels**: `phase-1`, `infra`

**Description**:
创建 Rust cargo workspace，包含以下 crate 结构：

```
robowbc/
├── Cargo.toml              # workspace root
├── crates/
│   ├── robowbc-core/       # WbcPolicy trait + Observation/Command/Output 类型定义
│   ├── robowbc-ort/        # ort (ONNX Runtime) 推理后端
│   ├── robowbc-registry/   # inventory-based 策略注册与工厂
│   └── robowbc-cli/        # 命令行入口 (加载配置 → 运行策略)
```

**Acceptance Criteria**:
- [ ] `cargo build` 通过
- [ ] workspace 成员正确配置
- [ ] `.github/workflows/ci.yml` 包含 `cargo check`, `cargo test`, `cargo clippy`, `cargo fmt --check`
- [ ] `rustfmt.toml` 和 `clippy.toml` 配置完成

**Dependencies**: None

---

### Issue #2: 定义 `WbcPolicy` trait 和核心数据类型

**Labels**: `phase-1`, `infra`

**Description**:
在 `robowbc-core` 中定义核心抽象：

```rust
pub trait WbcPolicy: Send + Sync {
    fn predict(&self, obs: &Observation) -> Result<JointPositionTargets>;
    fn control_frequency_hz(&self) -> u32;
    fn supported_robots(&self) -> &[RobotConfig];
}

pub struct Observation { ... }
pub enum WbcCommand { ... }
pub struct JointPositionTargets { ... }
pub struct RobotConfig { ... }
```

需要覆盖 founding document Part 5.1 中定义的所有类型，包括：
- `Observation`: joint_positions, joint_velocities, gravity_vector, command, timestamp
- `WbcCommand`: Velocity, EndEffectorPoses, MotionTokens, JointTargets, KinematicPose
- `JointPositionTargets`: positions + timestamp
- `RobotConfig`: joint_count, joint_names, pd_gains 等

**Acceptance Criteria**:
- [ ] 类型定义完整，包含文档注释
- [ ] `cargo doc` 生成无警告
- [ ] 单元测试验证类型创建和基本操作

**Dependencies**: #1

---

### Issue #3: 实现 `inventory`-based 策略注册表 (Registry)

**Labels**: `phase-1`, `infra`

**Description**:
在 `robowbc-registry` 中实现策略注册与发现机制：

```rust
// 注册
inventory::submit! { WbcRegistration::new::<GearSonicPolicy>("gear_sonic") }

// 发现与实例化
let policy = WbcRegistry::build("gear_sonic", &config)?;
```

参考 StarVLA 的 registry + factory 模式，但利用 Rust 的 `inventory` crate 实现编译期自动发现。

**Acceptance Criteria**:
- [ ] 支持策略注册和按名称查找
- [ ] 支持从 TOML 配置文件驱动实例化
- [ ] 错误处理：未知策略名称返回明确错误
- [ ] 单元测试使用 mock policy 验证注册/发现流程

**Dependencies**: #2

---

### Issue #4: 定义 Unitree G1 的 `RobotConfig`

**Labels**: `phase-1`, `infra`

**Description**:
创建 Unitree G1 的硬件配置，包含：
- 关节数量、名称、顺序
- PD 增益默认值
- 关节角度限位
- 默认站立姿态 (default pose)

参考 GEAR-SONIC C++ 部署代码中的 G1 配置。以 TOML 文件 + Rust struct 双重形式存在。

**Acceptance Criteria**:
- [ ] `configs/robots/unitree_g1.toml` 配置文件
- [ ] `RobotConfig` 可从 TOML 反序列化
- [ ] 关节名称和顺序与 GEAR-SONIC 部署代码一致

**Dependencies**: #2

---

## Phase 1: GEAR-SONIC 端到端集成

### Issue #5: 实现 ort (ONNX Runtime) 推理后端

**Labels**: `phase-1`, `model`

**Description**:
在 `robowbc-ort` 中封装 `ort` crate，提供通用的 ONNX 模型加载和推理能力：

- 加载 `.onnx` 模型文件
- 支持 CUDA EP 和 TensorRT EP
- 输入/输出张量的 `Vec<f32>` ↔ ort tensor 转换
- Session 管理和线程安全

**Acceptance Criteria**:
- [ ] 可加载任意 ONNX 模型并执行推理
- [ ] 支持 CPU / CUDA / TensorRT execution provider 配置
- [ ] 错误处理：模型文件缺失、shape 不匹配等
- [ ] 基准测试：记录单次推理延迟

**Dependencies**: #1

---

### Issue #6: 实现 GEAR-SONIC `WbcPolicy` 封装

**Labels**: `phase-1`, `model`

**Description**:
实现 `GearSonicPolicy` 结构体，封装 SONIC 的三个 ONNX 模型：

1. `model_encoder.onnx` — 将运动参考编码为 latent tokens
2. `model_decoder.onnx` — 将 latent 解码为关节目标
3. `planner_sonic.onnx` — 实时运动规划器

需要：
- 从 HuggingFace (`nvidia/GEAR-SONIC`) 下载模型权重
- 理解三个模型的输入/输出张量 shape 和数据流
- 实现 `WbcPolicy` trait
- 处理 SONIC 特有的内部状态管理（encoder 输出缓存等）

**Acceptance Criteria**:
- [ ] 三个 ONNX 模型成功加载并链式推理
- [ ] `predict()` 输入 `Observation`，输出 `JointPositionTargets`
- [ ] 50 Hz 推理频率下延迟 < 5ms (GPU)
- [ ] 通过 `inventory` 自动注册
- [ ] 集成测试：使用录制的观测数据验证输出合理性

**Dependencies**: #2, #3, #5

---

### Issue #7: 实现 zenoh 通信层 — 连接 Unitree G1

**Labels**: `phase-1`, `infra`

**Description**:
创建 `robowbc-comm` crate（或在 cli 中实现），通过 zenoh 与 Unitree G1 通信：

- 订阅关节状态 topic（joint positions, velocities）
- 订阅 IMU 数据
- 发布关节位置目标命令
- 维持 50 Hz 控制循环的精确时序

需要与 Unitree SDK 的 DDS topic 格式兼容。

**Acceptance Criteria**:
- [ ] 可连接真实 G1（或模拟器）并读取关节状态
- [ ] 可发送关节位置目标命令
- [ ] 控制循环频率稳定在 50 Hz ± 1 Hz
- [ ] 通信延迟 < 2ms (同机)

**Dependencies**: #1

---

### Issue #8: 端到端集成 — SONIC on G1 完整运行

**Labels**: `phase-1`

**Description**:
将所有 Phase 1 组件集成：

```
CLI 加载配置 → Registry 创建 GearSonicPolicy → zenoh 接收观测 → predict() → zenoh 发送命令
```

实现 `robowbc-cli`：
- 从 TOML 配置文件读取：模型名称、模型路径、机器人配置、通信配置
- 启动控制循环
- Graceful shutdown (Ctrl+C)
- 运行时指标日志（推理延迟、循环频率、丢帧数）

**Acceptance Criteria**:
- [ ] `robowbc run --config sonic_g1.toml` 一条命令启动
- [ ] G1 可以执行 SONIC 的运动控制
- [ ] 性能对标 NVIDIA 原生 C++ 部署（延迟和稳定性）
- [ ] 连续运行 10 分钟无崩溃

**Dependencies**: #4, #6, #7

---

### Issue #9: 性能基准测试 — 对标 NVIDIA C++ 部署

**Labels**: `phase-1`, `docs`

**Description**:
建立性能基准测试框架，对比 robowbc (Rust + ort) vs NVIDIA 原生 C++ 部署：

- 单次推理延迟（P50, P99, P999）
- 控制循环频率稳定性
- 内存占用
- GPU 利用率
- 端到端延迟（传感器读取 → 命令发出）

输出为可复现的 benchmark 脚本 + 结果报告。

**Acceptance Criteria**:
- [ ] `cargo bench` 可运行推理基准
- [ ] 与 NVIDIA C++ 部署的对比数据
- [ ] 结果记录在 `docs/benchmarks/` 中

**Dependencies**: #8

---

## Phase 2: 多模型抽象验证

### Issue #10: 实现 Decoupled WBC 策略封装

**Labels**: `phase-2`, `model`

**Description**:
集成 NVIDIA Decoupled WBC（GR00T N1.5/N1.6 的生产默认控制器）：

- 从 PyTorch 导出为 ONNX
- 实现 `DecoupledWbcPolicy`，封装 RL 下半身 + 解析 IK 上半身
- 输入：velocity commands + end-effector SE3 poses
- 注册到 registry

验证「同一框架、同一接口、不同模型」的核心命题。

**Acceptance Criteria**:
- [ ] ONNX 导出成功
- [ ] `WbcPolicy` trait 实现完整
- [ ] 配置切换：仅改 TOML 即可从 SONIC 切到 Decoupled WBC
- [ ] G1 上实际运行验证

**Dependencies**: #3, #5

---

### Issue #11: 实现 PyO3 Python 推理后端

**Labels**: `phase-2`, `infra`

**Description**:
创建 `robowbc-pyo3` crate，支持直接调用 Python PyTorch 模型：

- PyO3 + numpy zero-copy 数据交换
- 加载 `.pt` / `.pth` 模型文件
- 在 Python 子解释器中执行 `model.forward()`
- 线程安全：GIL 管理策略

这是支持 HOVER、OmniH2O 等未导出 ONNX 的模型的关键路径。

**Acceptance Criteria**:
- [ ] 可加载并执行任意 PyTorch 模型
- [ ] 推理延迟开销相比纯 Python < 20%
- [ ] 与 ort 后端实现相同的 trait 接口
- [ ] 单元测试覆盖基本推理流程

**Dependencies**: #2

---

### Issue #12: 实现 HOVER 策略封装

**Labels**: `phase-2`, `model`

**Description**:
集成 NVIDIA HOVER（多模态命令空间，15+ 控制模式）：

- 需要先在 Isaac Lab 中训练并导出模型
- 实现 `HoverPolicy`
- 支持 HOVER 的多模式 sparsity mask 机制
- 硬件适配：Unitree H1（19 DOF）

HOVER 的多模式输入对 `WbcCommand` 枚举设计是重要验证。

**Acceptance Criteria**:
- [ ] 模型训练 + 导出流程文档化
- [ ] `WbcPolicy` trait 实现
- [ ] 多模式命令切换正常工作
- [ ] 仿真环境验证（H1 sim）

**Dependencies**: #3, #11 (PyO3 后端) 或 #5 (若导出 ONNX)

---

### Issue #13: TOML 配置驱动的模型切换

**Labels**: `phase-2`, `infra`

**Description**:
设计并实现完整的配置系统：

```toml
[policy]
name = "gear_sonic"
model_dir = "./models/sonic/"

[robot]
config = "unitree_g1"

[communication]
backend = "zenoh"
frequency_hz = 50

[inference]
backend = "ort"
device = "cuda:0"
```

支持：
- 仅修改配置文件即可切换模型、机器人、通信后端
- 配置验证和友好的错误提示
- 配置文件模板生成 (`robowbc init`)

**Acceptance Criteria**:
- [ ] 修改 `policy.name` 即可切换 SONIC ↔ Decoupled WBC
- [ ] 配置文件 schema 文档化
- [ ] `robowbc init` 生成带注释的模板配置
- [ ] 无效配置给出清晰错误信息

**Dependencies**: #3, #6, #10

---

### Issue #14: 第二硬件平台支持 — Unitree H1 或 Booster T1

**Labels**: `phase-2`, `infra`

**Description**:
添加第二个机器人硬件配置，验证 `RobotConfig` 抽象的通用性：

- 创建 `configs/robots/unitree_h1.toml`（或 `booster_t1.toml`）
- 关节映射、PD 增益适配
- 验证同一策略在不同硬件上运行（需要策略支持该硬件）

**Acceptance Criteria**:
- [ ] 新硬件配置文件完整
- [ ] 至少一个策略在新硬件上运行（仿真或实机）
- [ ] `RobotConfig` 抽象无需修改即可支持新硬件

**Dependencies**: #4

---

## Phase 3: 社区与文档

### Issue #15: Python SDK — PyO3 绑定发布

**Labels**: `phase-3`, `infra`

**Description**:
将 robowbc 封装为 Python 包，发布到 PyPI：

```python
import robowbc

policy = robowbc.load("gear_sonic", config_path="sonic_g1.toml")
targets = policy.predict(observation)
```

用户体验应与纯 Python 库无异。参考 HuggingFace `tokenizers` 的 PyO3 封装策略。

**Acceptance Criteria**:
- [ ] `pip install robowbc` 可安装
- [ ] Python API 文档完整
- [ ] 示例 notebook / script
- [ ] CI 构建 manylinux wheel

**Dependencies**: #11, #13

---

### Issue #16: 文档、教程与示例

**Labels**: `phase-3`, `docs`

**Description**:
建立完整的文档体系：

- **Getting Started**: 从安装到运行第一个策略
- **Architecture Guide**: 核心概念、trait 设计、推理后端
- **Model Integration Guide**: 如何添加新的 WBC 模型
- **Hardware Guide**: 如何添加新的机器人硬件
- **API Reference**: Rust doc + Python doc
- **示例集**: 各模型 × 各硬件的配置示例

**Acceptance Criteria**:
- [ ] mdBook 或类似工具构建文档站
- [ ] 至少 3 个端到端示例
- [ ] 新模型集成教程（以一个具体模型为例）

**Dependencies**: #8, #13

---

### Issue #17: 向上游提交集成 PR

**Labels**: `phase-3`, `docs`

**Description**:
- **GR00T-WholeBodyControl**: 提交 robowbc 集成指南作为文档 PR
- **LeRobot**: 提交作为 WBC 执行后端的集成方案

扩大项目影响力，与上游社区建立联系。

**Acceptance Criteria**:
- [ ] GR00T-WholeBodyControl PR 提交
- [ ] LeRobot PR 或 RFC 提交
- [ ] 技术博客文章（中英文各一篇）

**Dependencies**: #16

---

## 总览

| Issue | 标题 | Phase | 依赖 | 优先级 |
|-------|------|-------|------|--------|
| #1 | Rust cargo workspace 骨架 | 0 | - | 🔴 P0 |
| #2 | WbcPolicy trait + 核心类型 | 0 | #1 | 🔴 P0 |
| #3 | inventory 策略注册表 | 0 | #2 | 🔴 P0 |
| #4 | Unitree G1 RobotConfig | 0 | #2 | 🔴 P0 |
| #5 | ort ONNX 推理后端 | 1 | #1 | 🔴 P0 |
| #6 | GEAR-SONIC WbcPolicy 封装 | 1 | #2,#3,#5 | 🔴 P0 |
| #7 | zenoh 通信层 | 1 | #1 | 🟡 P1 |
| #8 | 端到端集成 (SONIC on G1) | 1 | #4,#6,#7 | 🔴 P0 |
| #9 | 性能基准测试 | 1 | #8 | 🟡 P1 |
| #10 | Decoupled WBC 封装 | 2 | #3,#5 | 🟡 P1 |
| #11 | PyO3 Python 推理后端 | 2 | #2 | 🟡 P1 |
| #12 | HOVER 策略封装 | 2 | #3,#11/#5 | 🟢 P2 |
| #13 | TOML 配置驱动模型切换 | 2 | #3,#6,#10 | 🟡 P1 |
| #14 | 第二硬件平台支持 | 2 | #4 | 🟢 P2 |
| #15 | Python SDK (PyPI) | 3 | #11,#13 | 🟢 P2 |
| #16 | 文档与教程 | 3 | #8,#13 | 🟢 P2 |
| #17 | 上游 PR | 3 | #16 | 🟢 P2 |

### 建议执行顺序（关键路径）

```
#1 → #2 → #3 → #5 → #6 → #7 → #8 → #9
         ↘ #4 ↗              ↘ #10 → #13
                               ↘ #11 → #12
```

Phase 1 的关键路径是 **#1 → #2 → #5 → #6 → #8**，建议优先推进。
