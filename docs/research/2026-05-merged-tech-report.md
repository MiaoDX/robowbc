# robowbc 技术调研报告（合并版 v2）

> 本报告基于 2026-05 的英文版调研整理为中文,并合入两项修订:(1) 删除"runtime 里 policy 热切"的讨论(明确不作为支持目标);(2) robowbc 本身代码采用 **MIT license**,第三方依赖与 model weight 协议通过独立 `LICENSES/` 目录承载。
>
> 关键技术名词(cyclors、cyclonedds-rs、dust-dds、CycloneDDS、Zenoh、unitree_sdk2、unitree_mujoco、ROS2、TensorRT、PyO3、ONNX、MuJoCo 等)保留英文不翻译。

---

## TL;DR

- **DDS Rust transport 选型**:在 Unitree 生态里,与 G1 在 wire 层兼容的事实标准是 **CycloneDDS 0.10.x + `unitree_hg` IDL**。在 Rust 侧最稳妥的路径是用 **`cyclors`**(ZettaScaleLabs 维护,被 zenoh-bridge-dds / zenoh-plugin-ros2dds 用作底层)或 **`cyclonedds-rs`**(Sojan James 个人维护、API 更地道但绑定 0.10.x 的旧 cyclonedds)。**`dust-dds` / `RustDDS` 纯 Rust 实现在标准上 OK,但与 Unitree 的 `unitree_hg` IDL/QoS 兼容性需要额外验证**,短期不推荐做主路径。Zenoh 适合做远程 / WAN / 多机编排和 teleop 入口,但要触达 G1 仍需要一座 DDS 桥。
- **Onboarding UX 必须复制 rl_sar 与 GR00T-WholeBodyControl 这两个项目的核心动作**:(a) `policy/<robot>/<config>/` 文件夹里放 `.pt` / `.onnx` + `config.yaml` + `base.yaml`;(b) 仿真启动 = 一行命令 `./bin/rl_sim_mujoco g1 scene_29dof`,真机启动 = 一行命令 `./bin/rl_real_g1 <iface>`;(c) 仿真先跑 unitree_mujoco,再跑真 G1;(d) emergency stop(GR00T 用 `O` 键,Unitree 推荐 L2+B/L2+R2)必须显式可用。
- **第一版应当 ship 的最小可行 demo**:用户 `git clone robowbc` → `docker compose up unitree_mujoco` → `cargo run --bin robowbc-run -- --policy gear_sonic --robot g1 --transport cyclonedds-sim`,机器人在 MuJoCo 里走起来;teleop 第一版只发**键盘**一种输入,复刻 GR00T 的 `O = e-stop / ] = engage / T = play motion` 模式。Path A(人形 pipeline)排序后的 4–8 周里程碑见第 9 章。

---

## 第 1 章:DDS Rust transport 选型(最高优先级)

### 1.1 背景与约束

robowbc 必须能跟两类 endpoint 通信:

1. **unitree_mujoco**:基于 `unitree_sdk2` C++ + MuJoCo 的本地仿真器,DDS 层用的是 **CycloneDDS 0.10.x**,G1 走 `unitree_hg` IDL(`unitree_go` IDL 是 Go2/B2/H1 系列用的)。topic 默认在 `domain_id=1` 的 `lo` 接口上,发布 `rt/lowstate`、订阅 `rt/lowcmd`。
2. **真 Unitree G1**:onboard CycloneDDS 0.10.x(默认装的版本对 G1 的新 `unitree_hg` IDL 有 bug,需要重装到 unitree_ros2 fork 的版本),网络接口典型为 `enp3s0` / `eth0`,通过 192.168.123.x 网段对外可见。

任何 robowbc 的 Rust transport 要想跑得动,wire 协议必须是 RTPS,IDL 必须能解 `unitree_hg::msg::dds_::LowCmd_` / `LowState_` / `HandCmd_` / `HandState_`,QoS 要跟 `unitree_sdk2` 默认一致(reliable,KeepLast=1,500 Hz 控制循环)。

### 1.2 候选 (a):基于 CycloneDDS 的三个 Rust binding

| Crate | 性质 | 维护方 | 维护活跃度 | 最契合的使用场景 |
|---|---|---|---|---|
| **cyclors** | `libddsc-sys` 风格的低层 C 绑定,Apache-2.0 | ZettaScaleLabs(zenoh 团队) | 持续更新,跟随 cyclonedds 主线 | zenoh-bridge-dds / zenoh-plugin-ros2dds 的底盘;任何要在 Rust 里"原汁原味"调 cyclonedds C API 的项目 |
| **cyclonedds-rs** | 安全 Rust 包装 + derive 宏,免 IDL 代码生成 | sjames(Sojan James 个人) | 缓慢;锁在 cyclonedds **0.10.x** 分支 | 偏向 idiomatic Rust 的应用层用户,不需要最新 cyclonedds feature |
| **dust-dds** | 纯 Rust 重新实现 DDS + DDSI-RTPS(含 minimum profile) | s2e-systems | 活跃,有 sync/async 双 API、商业支持 | 不需要 C 依赖、想要 100% safe Rust 的应用;但要验证与 cyclonedds 的 `unitree_hg` IDL wire-level 互通 |

补充候选:**RustDDS**(Atostek Oy)也是纯 Rust,跟 ROS 2 实测互通但对 Unitree IDL 的支持没有公开案例。

#### 1.2.1 cyclors 详细评估

- **构建复杂度**:`cyclors` 自带 vendored `cyclonedds` C 源码(crate 体积 ~16 MB / 319K SLoC),cargo build 时由 cmake/cc 在内部编译 libcyclonedds。**不需要**用户预装系统级 libcyclonedds。可选 feature:
  - `iceoryx`:开启 SHM 共享内存(仅 Linux/macOS)。
  - `prefix_symbols`:把 cyclone 符号加上 crate 版本前缀,避免和别的 cyclors(如 zenoh-plugin-dds 里的)静态链接冲突;macOS/Windows 需要 `llvm-nm` + `llvm-objcopy`。
  - `dds_security`:开启 DDS Security(仅 Linux/macOS)。
- **API 易用性**:本质是 `unsafe extern "C"` + `bindgen` 生成的常量/函数,要自己包薄一层 Rust 安全 wrapper。对 `LowCmd_`/`LowState_` 这种结构体,需要自己根据 `unitree_hg` IDL 写 CDR 序列化(或者 link 进 unitree_sdk2 的 IDL 编译产物)。
- **交叉编译**:和 cyclonedds 本身一样需要 cmake 工具链,但 zenoh-plugin-ros2dds 团队已经验证过 zig + colcon 的 aarch64 交叉编译路径。
- **机器人项目里的真实使用情况**:Eclipse Zenoh 全家桶(`zenoh-plugin-dds`、`zenoh-plugin-ros2dds`、`zenoh-bridge-ros2dds`)都把 cyclors 作为 DDS 端实现;这是 Rust 生态里和 ROS 2 / Unitree DDS 兼容性验证最充分的路径。
- **License**:Apache-2.0。

#### 1.2.2 cyclonedds-rs 详细评估

- **构建复杂度**:依赖外部预装的 `cyclonedds 0.10.x` + `iceoryx 2.0.2`(README 明确锁版本,连 commit hash 都写死了)。比 cyclors 更"麻烦",因为不会自动 vendor。
- **API 易用性**:提供 `#[derive(Topic)]` 宏,Reader/Writer/Listener、async reader、嵌套 key、SHM 都封装好了,比 cyclors 上手快得多。
- **维护风险**:仓库 35 stars / 8 forks,作者一人维护,README 直接说"current release only supports 0.10.X release branch"。如果 Unitree 后续把 cyclonedds 升级到 0.11+,可能要自己 fork 修。
- **License**:Apache-2.0。

#### 1.2.3 dust-dds 详细评估

- **构建复杂度**:纯 Rust,stable Rust,**no_unsafe**,cargo install 一行搞定,无外部 C 依赖。
- **API 易用性**:`#[derive(DdsType)]`,sync + async 双 API。可以从 IDL 用 `dust_dds_gen` 生成类型,所以理论上能解 `unitree_hg` 的 IDL。
- **互操作性风险**:dust-dds 的 README 自述 "minimum DDS profile + DDSI-RTPS",靠的是 OMG Shapes Demo 互测过。**没有公开证据表明它和 unitree_sdk2 的 cyclonedds + unitree_hg 跑通过**——这是这条路径的最大不确定性。**robowbc 第一版不应把 dust-dds 当主路径,但值得做一周的 spike**:写个最小 publisher/subscriber 跟 unitree_mujoco 的 `rt/lowstate` 互通试试。
- **License**:Apache-2.0。

### 1.3 候选 (b):基于 Zenoh 的 transport

Zenoh 本身是 Apache-2.0 的 Rust 原生 pub/sub + storage + queryable 协议栈,性能/discovery 开销在 ROS 2 over WiFi 场景比 vanilla DDS 好很多(zenoh.io 自己的 benchmark 声称把 discovery 流量降低 97%–99.9%)。但**Zenoh 不是 DDS 的 wire-level 替代品**:要跟 unitree_mujoco / 真 G1 通信,必须经过 DDS bridge:

- **`zenoh-bridge-dds`**:通用 DDS 路由桥,底层用 cyclors 调 cyclonedds。新项目官方推荐 `zenoh-bridge-ros2dds`,但 ROS 2 graph 概念在 G1 上意义不大(G1 是裸 DDS 不是 ROS 2)。
- **`zenoh-bridge-ros2dds`**:ROS 2 专用桥,更好地支持 namespace、QoS 推断、action/service。如果 robowbc 后续要兼容 ROS 2 节点(比如 nav2、rviz2),用这个。
- **混合架构(推荐 long-term)**:Rust runtime 直接发 zenoh key(如 `robowbc/g1/lowcmd`),由本地 zenoh-bridge-dds 桥成 `rt/lowcmd` DDS topic。teleop / 远程 dashboard 走原生 zenoh,机器人控制循环里 bridge → DDS。

### 1.4 unitree_mujoco 兼容性、QoS 和 topic 表

| Topic | 方向(host → robot 视角) | IDL(G1 = unitree_hg) | 频率 | 用途 |
|---|---|---|---|---|
| `rt/lowcmd` | publish | `unitree_hg::msg::dds_::LowCmd_` | 500 Hz | 35 个电机的 q/dq/tau/kp/kd 命令(带 CRC32) |
| `rt/lowstate` | subscribe | `unitree_hg::msg::dds_::LowState_` | 500 Hz | 关节状态、IMU、底层错误码 |
| `rt/arm_sdk` | publish | `unitree_hg::msg::dds_::LowCmd_` | 通常 50–500 Hz | 上半身运动控制接管,`motor_cmd[29].q` 是 weight ∈ [0,1] |
| `rt/dex3/left/cmd` / `rt/dex3/right/cmd` | publish | `unitree_hg::msg::dds_::HandCmd_` | 50–100 Hz | Dex3 灵巧手 7 motor 控制 |
| `rt/dex3/{left,right}/state` | subscribe | `HandState_` | — | 7 motor + 9 压力传感器 |
| `rt/wirelesscontroller` | subscribe | `WirelessController_` | — | 遥控器键值(仿真里通过 USB joystick 模拟) |
| `rt/secondary` (G1 only) | subscribe | `IMUState_` | — | 胸部 IMU |
| `rt/api/loco/{request,response}` | request/response | `unitree_api::msg::dds_::Request_/Response_` | 事件 | LocoClient JSON-RPC(StandUp、Move、Damp、WaveHand…) |
| `rt/api/motion_switcher/{request,response}` | request/response | 同上 | 事件 | 切换 sport mode / debug mode |

**QoS**:reliable + KeepLast=1,CycloneDDS 默认。仿真器 `unitree_mujoco` 在 `simulate/config.yaml` 里默认 `domain_id: 1` + `interface: lo`(真机默认 `domain_id: 0` + `enp3s0` 之类),要小心用 domain id 区分仿真和真机。

### 1.5 unitree_sdk2_python (PyO3 fallback)

如果 Rust DDS 写起来太痛苦,可以用 **PyO3** 在 Rust runtime 进程里嵌一个 Python 解释器、`import unitree_sdk2py` 跑 channel publisher/subscriber,把 numpy 数组在 Rust ↔ Python 之间零拷贝传。

- **可行性**:技术上完全可行,PyO3 0.23+ 的 `Python::with_gil` + `PyModule::import` 已经标准化;NumPy ndarray 互转有 `numpy` crate。
- **缺点**:(1) GIL 在 500 Hz 控制循环里是隐患,但只要把 DDS write/read 放进 Python 短临界区可以接受;(2) 部署要带 cpython + 完整 unitree_sdk2_python venv,比纯 Rust binary 重;(3) free-threaded Python(3.13+)下 GIL 行为有变化,需要测。
- **建议**:**作为 v0 的 escape hatch 可以保留**,但不要做成默认。Path A 的 4 周里程碑应当能完全脱离 Python SDK。

### 1.6 决策矩阵

| 优先级 | 推荐路径 | 理由 |
|---|---|---|
| **(i) 风险最低、最快跑通 sim** | **PyO3 调 unitree_sdk2_python**(v0),并行做 cyclors-based Rust transport spike | unitree_sdk2_python 是上游官方支持,跟 unitree_mujoco 100% 兼容,1 天就能 sim 跑起来;同时不阻塞 Rust 路径 |
| **(ii) 长期最佳的 Rust-native 架构** | **cyclors(直接)或 cyclonedds-rs(薄封装层)**,自己生成 `unitree_hg` IDL 的 Rust 类型 | wire-level 100% 跟 cyclonedds 兼容,跟 zenoh 生态可融合;比 dust-dds 风险低 |
| **(iii) 跟 ROS 2 生态共存最容易** | Rust runtime 直接用 zenoh API + 本地起 `zenoh-bridge-ros2dds` | 既能用原生 zenoh 编排远程 teleop / 多机,又能让 ROS 2 node(rviz2、nav2、teleop_twist_keyboard)零改动接入 |

> **Recommended for robowbc v1**: 主路径走 **cyclors + 自写 `unitree_hg` IDL Rust crate**;teleop / 远程入口预留一个 **zenoh feature flag**(默认关闭),把"DDS-only"和"DDS+Zenoh"两种部署都做成 cargo feature。dust-dds 留作 spike,验证通过后可以做成第二种 transport backend。

---

## 第 2 章:Unitree G1 控制接口的碎片化

### 2.1 控制面的三条线

G1 的控制接口在物理上分成三套相对独立但共享 DDS domain 的子系统:

1. **Locomotion**:`g1::LocoClient` → `rt/api/loco/{request,response}`(高层 JSON-RPC,对应 `Damp()` / `StandUp()` / `Move(vx,vy,wz)` / `Squat2StandUp` / `WaveHand` 等 22 个动作);以及底层 `rt/lowcmd` / `rt/lowstate`(35 motor 直接 PD + feedforward torque)。
2. **Arm**:`rt/arm_sdk`(共用 `LowCmd_` 类型,但只在 motor index 15–28 有效,并且 `motor_cmd[29].q` 是接管 weight ∈ [0,1],0 = 内置上身控制,1 = 用户完全接管);高层 `G1ArmActionClient` → `rt/api/{arm_action}` 走预定义 gesture 库。
3. **Hand (Dex3)**:`rt/dex3/{left,right}/cmd` / `rt/dex3/{left,right}/state`,每只手 7 motor、9 压力传感器,`HandCmd_` 用 `RIS_Mode_t` 结构。

### 2.2 操作模式与切换

- **sport mode**(出厂高层运动服务,FSM 驱动):内置上身/下身控制,用户只能发 LocoClient 高层命令。
- **debug mode / custom mode**:内置 sport_mode 完全退出,关节进 damping,用户接管 `rt/lowcmd`。**进入 debug mode 后 `SportModeState` 在真机上不可读**,但 unitree_mujoco 仿真里仍然保留这个 topic(方便分析)。
- **切换方式**:(a) 遥控器:`L2+B` 进 damping,`L2+UP` 锁定站立,`R1+X` 进主控;旧版本是 `L1+A` / `L1+UP` / `R1+X`。`L2+R2` 进零力矩调试模式。(b) 程序:通过 `rt/api/motion_switcher/{request,response}` 发请求。
- **arm_sdk 接管的渐进 weight**:xr_teleoperate wiki 明确建议 weight 从 0 缓慢线性插到 1,不要瞬间设 1.0(除非启动时 q_target == q_current),否则上身可能高速摆动。

### 2.3 各 stack 的拼装方式

#### 2.3.1 GR00T-WholeBodyControl 的 C++ 部署栈(`gear_sonic_deploy`)

GR00T-WBC 的 deploy 栈有以下特征(来自 DeepWiki 索引):

- **PolicyEngine** + **TRTInferenceEngine**:policy 推理在 dedicated CUDA stream 上跑,避免阻塞主控/安全线程。50 Hz 主控循环,policy 推理也是 50 Hz;**KinematicPlanner** 在 10 Hz 跑(生成参考轨迹)。
- **Memory pinning**:用 `cudaMallocHost` 给 input/output buffer 做 pinned memory 提高 DMA 速度。
- **deploy 入口**:`deploy.sh sim`(连 unitree_mujoco)/ `deploy.sh real`(连真 G1,自动检测 192.168.123.x 网段)。`O` 键紧急停止;`]` 启动控制系统;`T` 播放当前参考动作。
- **多输入模式**:`--input-type keyboard` / `--input-type zmq` / `--input-type zmq_manager`(PICO VR teleop)。motion 数据通过 ZMQ(不是 DDS)从 teleop / planner 进 deploy stack。
- **协议演进**:3 月 24 日 update 把 ZMQ header 升到 1280 bytes,加了 motor error monitoring 和 TTS alerts。

#### 2.3.2 rl_sar 怎么处理 G1

rl_sar (`fan-ziqi/rl_sar`) 是仿真验证 + 真机部署框架,和 GR00T 的设计哲学不一样——它把每个机器人的入口写成单独的 binary(`rl_real_a1` / `rl_real_go2` / `rl_real_g1`),全 C++(PyTorch 通过 LibTorch 加载 `.pt` model)。G1 和 Go2/Go2W 都通过同一个 USB Ethernet → `192.168.123.x` 网段连过去,传输直接用 `unitree_sdk2`(CycloneDDS)。

源码层面:`src/rl_sar/include/rl_real_*.hpp` 里继承 `RLReal` 基类,每个机器人实现自己的 `SetCommand()` / `GetState()` 把 RL SDK 的内部数据结构和 Unitree SDK 的 `LowCmd_` / `LowState_` 互转。

#### 2.3.3 unitree_rl_gym 的真机 deploy 栈

`unitree_rl_gym` 的 `deploy/deploy_real/` 是一个独立的 Python 部署器,输入参数是 `<net_interface> <config_name>`(如 `enp3s0 g1.yaml`)。启动序列固定:

1. 机器人吊挂 → 等进零力矩模式
2. 遥控器 `L2+R2` → debug mode(damping)
3. 启动部署脚本:`python deploy_real.py enp3s0 g1.yaml`
4. 关节进默认位姿(程序控制插值)
5. 操作员按遥控器 `start` → 站立到默认 dof pos
6. 慢慢放下吊带让脚着地
7. 按遥控器 `A` → 原地踏步
8. 用左/右摇杆走

### 2.4 安全层模式

- **关节限位**:每个 joint 有硬件限位,但 RL policy 输出在被发到 `rt/lowcmd` 之前必须做 software clip。`unitree_rl_gym` 的 deploy 代码里 PD gain 也由 `g1.yaml` 配置中心化定义,避免在多个地方调。
- **速度限制**:通常通过 RL 训练时的 reward + termination 条件保证;deploy 时无显式 dq 限位(推荐增加)。
- **急停**:(a) 遥控器 `L2+B` 立即进 damping mode;(b) GR00T 用键盘 `O`;(c) PICO VR `A+B+X+Y` 同时按。
- **Watchdog**:unitree_sdk2 的 SDK 内部有连接超时,但 policy 失控(比如 NaN 输出)的检测**不是 SDK 提供的**——必须在 robowbc 自己加(见第 7 章)。
- **CRC**:`rt/lowcmd` 每帧必须算 CRC32 否则机器人会忽略;`g1::publisher::LowCmd` 的 wrapper 自动算。Rust transport 也必须实现这个 CRC,否则 wire 上看不到任何反应。

---

## 第 3 章:unitree_mujoco 的 sim-first 验证路径

### 3.1 进程模型与网络

`unitree_mujoco` 是**独立进程 + DDS over UDP**,**不是** in-process。具体表现为:

- 起 `simulate/build/unitree_mujoco`(C++ 版,推荐)或者 `simulate_python/unitree_mujoco.py`,一个独立可执行文件 + MuJoCo viewer 窗口。
- 默认 `domain_id=1`、`interface=lo`,避免和真机的 `domain_id=0` 冲突。
- 内部用 `unitree_sdk2_bridge` 把 MuJoCo 的关节状态转 `LowState_` 发出去,把订到的 `LowCmd_` 转回 MuJoCo 的 `data->ctrl`。
- 因为是真 DDS,所以**任何 unitree_sdk2 / unitree_sdk2_python 的客户端、unitree_ros2 节点、未来的 robowbc Rust runtime,都不需要改一行代码就能从仿真切到真机**——只要换 domain_id 和 interface。

### 3.2 跟生产 G1 的一致性

| 维度 | unitree_mujoco | 真 G1 | 备注 |
|---|---|---|---|
| `rt/lowstate` / `rt/lowcmd` topic 名 | ✅ 一致 | ✅ | |
| IDL(`unitree_hg`) | ✅ | ✅ | |
| 35 motor 顺序 | ✅ 一致(hardware-matching) | ✅ | |
| IMU `[w,x,y,z]` 四元数顺序 | ✅ | ✅ | |
| `SportModeState` topic | ✅ 始终发 | ⚠️ debug mode 下不可读 | 仿真里保留方便分析 |
| `rt/api/loco/*`(高层 LocoClient) | ❌ 仿真器**不支持**高层 sport service | ✅ | 仿真只支持 low-level,要用 LocoClient 必须真机 |
| `rt/wirelesscontroller` | ✅ 通过 USB joystick 模拟 | ✅ | |
| `rt/dex3/*` | ⚠️ 取决于 mjcf 是否带 hand | ✅ | G1_29dof_with_hand mjcf 有 |
| 关节摩擦/惯性 | ❌ 简化模型 | — | 这是最大的 sim2real gap 来源 |

### 3.3 G1 上典型的 sim2real gap

来自 arXiv 2503.01255 (Hu et al., "Impact of Static Friction on Sim2Real")、unitree_rl_gym GitHub issues #45/#65、IsaacLab #1784 等公开讨论:

1. **静摩擦 / Coulomb friction**:传统 domain randomization 通常不包括静摩擦,导致 policy 在仿真里学会"拖脚走",到真机走不动。缓解方案:(a) 把静摩擦放进 DR 参数空间;(b) 用 actuator net;(c) 训练时加 feet-air-time reward。
2. **执行器延迟 / 通信延迟**:500 Hz 控制循环里 `rt/lowcmd` 到电机执行有几个 ms 抖动,仿真里默认是 0。
3. **PD gain 不匹配**:`g1.yaml` 里写的 kp/kd 必须和真机 firmware 默认匹配,否则 policy 输出会被电机控制器二次塑形。
4. **观测维度对齐**:GR00T-WBC 的 mujoco_sim_g1 README 明确警告:"Observations for missing joints are zeroed out (indices 12, 14, 20, 21, 27, 28)",这种"sim 模型 dof 数 vs policy obs 维度对齐"问题在切机器人配置时最容易踩坑。
5. **IMU drift**:仿真里 IMU 是 ground truth,真机有 bias / noise。

### 3.4 CI 模式

公开仓库里没有看到现成的"unitree_mujoco headless CI" 模板,但模式是清晰的:

```bash
# robowbc 推荐的 CI smoke test(伪代码)
unitree_mujoco --headless --robot g1 --scene scene_29dof.xml &
SIM_PID=$!
sleep 3  # 等 DDS discovery 完成
timeout 30s cargo run --release --bin robowbc-run -- \
    --policy gear_sonic --robot g1 \
    --scene scene_29dof --transport cyclonedds-sim \
    --assert-no-nan --assert-state-evolves --duration 25
kill $SIM_PID
```

`unitree_mujoco` 的 C++ simulate 进程支持去掉 viewer 编译选项跑 headless(自己改 cmake),Python 版用 `mujoco.MjData` 不开 viewer 即可。

---

## 第 4 章:rl_sar / GR00T-WholeBodyControl 的 onboarding UX 模式

### 4.1 rl_sar 的目录与启动 UX

```
rl_sar/
├── src/rl_sar/
│   ├── policy/                    # 用户把 .pt 放这里
│   │   ├── g1/
│   │   │   ├── base.yaml          # G1 基础参数(PD gains、joint names、default_dof_pos)
│   │   │   ├── scene_29dof/       # 一个 scene/config
│   │   │   │   ├── config.yaml    # 这个 scene 特有的参数
│   │   │   │   ├── policy.pt      # 用户放进来的 model
│   │   │   │   └── fsm.hpp        # 这个 scene 的 FSM 实现
│   │   │   └── scene_23dof/...
│   │   └── go2/...
│   ├── include/
│   │   ├── rl_real_g1.hpp         # G1 真机入口
│   │   ├── rl_real_go2.hpp
│   │   └── rl_sim_mujoco.hpp      # MuJoCo sim 入口
│   ├── src/
│   │   ├── rl_real_g1.cpp
│   │   └── rl_sim_mujoco.cpp
│   ├── launch/                    # ROS1 / ROS2 launch
│   └── scripts/                   # Python 版(v2.3 之后暂停维护)
├── docker/                        # Docker compose + xhost 转发
├── build.sh
└── cmake_build/bin/               # 编译产物
    ├── rl_real_g1
    ├── rl_real_go2
    └── rl_sim_mujoco
```

**核心抽象**:

- **policy 文件夹** = `policy/<ROBOT>/<CONFIG>/`。`<ROBOT>` 是机型(g1, go2, a1...),`<CONFIG>` 是训练的 task / scene 配置(如 `scene_29dof`、`isaacgym`、`himloco`)。
- **`base.yaml`**:machine-level 不变量,包括关节顺序、`default_dof_pos`、PD gain 默认值、IMU 安装方位等。
- **`config.yaml`**:scene/policy-level 参数,包括 obs scale、cmd scale、policy frequency、observation history length。

**启动命令**(一行):

```bash
# Sim
./cmake_build/bin/rl_sim_mujoco g1 scene_29dof

# Real
./cmake_build/bin/rl_real_g1 enp3s0          # CMake build
rosrun rl_sar rl_real_g1                     # ROS1
ros2 run rl_sar rl_real_g1                   # ROS2
```

**底下做了什么**:

1. 解析 `<ROBOT>/base.yaml` + `<ROBOT>/<CONFIG>/config.yaml`
2. 用 LibTorch 加载 `policy.pt`
3. 通过 `unitree_sdk2`(cyclonedds)订 `rt/lowstate` / 发 `rt/lowcmd`
4. 跑 FSM:`StateInit` → `StateRL_Init`(关节插值到 default_dof_pos)→ `StateRL_Running`(policy 推理 → CRC → 发命令)
5. 监听键盘(`R` 重置、`Space` 紧急停止、WASD/箭头给 cmd_vel);ROS2 可选 `cmd_vel` topic 输入

### 4.2 GR00T-WholeBodyControl 的 onboarding UX

```
GR00T-WholeBodyControl/
├── gear_sonic/               # 训练 + simulation Python 栈
│   └── scripts/
│       ├── run_sim_loop.py           # MuJoCo sim 入口
│       └── pico_manager_thread_server.py  # PICO VR teleop
├── gear_sonic_deploy/        # C++ 部署栈
│   ├── include/
│   │   ├── policy_engine.hpp
│   │   ├── trt_inference_engine.hpp
│   │   └── kinematic_planner.hpp
│   ├── src/
│   │   └── g1_deploy_onnx_ref.cpp
│   ├── policy/release/       # 用户从 HuggingFace 下载放这里
│   │   ├── model_encoder.onnx
│   │   ├── model_decoder.onnx
│   │   └── observation_config.yaml
│   ├── planner/target_vel/V2/
│   │   └── planner_sonic.onnx
│   ├── deploy.sh
│   └── scripts/setup_env.sh
├── motionbricks/             # Preview release 实时潜空间生成
├── decoupled_wbc/            # 老版本(GR00T N1.5/N1.6 用)
├── download_from_hf.py       # 一键下载 checkpoint
└── check_environment.py
```

**关键设计**:

- **多 venv 隔离**:每种用例(sim / training / deploy / teleop)有独立的 `.venv_*`,用 `uv` 自动管理。这避免了"装个 PyTorch 把 MuJoCo 搞坏"的问题。
- **policy 交付通过 HuggingFace**:`python download_from_hf.py` 从 `nvidia/GEAR-SONIC` 拉 ONNX,按 deployment binary 期望的目录结构落盘。
- **启动命令**(两个 terminal):

```bash
# Terminal 1 — MuJoCo simulator
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2 — C++ deployment
cd gear_sonic_deploy
bash deploy.sh sim --input-type keyboard      # 仿真
bash deploy.sh real --input-type keyboard     # 真机(auto-detect 192.168.123.x)
```

- **key binding**:`O` = e-stop;`]` = engage policy;`9` = drop robot to ground (in MuJoCo);`T` = play current ref motion;`N`/`P` = next/prev motion;`R` = restart motion;`Q`/`E` = nudge heading ±π/12。

### 4.3 Docker 模式(rl_sar)

rl_sar `docker/` 目录的核心套路(来自仓库结构推断和通用 ROS X11 forwarding 模式):

```bash
# 启动前在 host 执行
xhost +local:docker

# docker compose up
services:
  rl_sar:
    build: .
    runtime: nvidia
    network_mode: host           # 让 DDS 多播能走通
    environment:
      - DISPLAY=$DISPLAY
      - QT_X11_NO_MITSHM=1
      - NVIDIA_DRIVER_CAPABILITIES=all
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./policy:/workspace/policy
    devices:
      - /dev/input:/dev/input    # 让容器看到 USB joystick
```

**坑**:

1. **NVIDIA Container Toolkit**(已经替代 deprecated 的 `nvidia-docker2`)必须在 host 装好才能跑 GPU。
2. **DDS 多播**:默认 docker `bridge` 网络隔离 multicast,CycloneDDS discovery 会失败。解决方案:(a) `network_mode: host`;(b) 配置 CycloneDDS 用 unicast initial peers。
3. **X11 forwarding**:MuJoCo viewer 需要 X11 / OpenGL,要 `--volume /tmp/.X11-unix` + `xhost +local:docker`,并且 NVIDIA 驱动版本必须和容器里的 libgl 兼容。
4. **GUI 在 macOS / Windows 主机基本走不通**,得用 XQuartz / VcXsrv,延迟高、不推荐。

### 4.4 ROS1 / ROS2 / 无 ROS 三种支持

rl_sar 三种都支持,通过预处理宏(`#define USE_ROS`)和 CMake 选项切换。这个复杂度**不建议 robowbc 第一版抄**——理由:

- 维护成本:每加一个机器人,要在三种构建路径里都验证。
- ROS1 已经 EOL(Noetic 2025-05 EOL),新项目不应该往里走。
- ROS2 集成可以通过 zenoh-bridge-ros2dds **延迟到运行时**,编译时不耦合。

**推荐**:robowbc 第一版只发 **"无 ROS + Cargo binary"**,第二版加 **"zenoh feature flag"** 让 ROS2 节点可选接入。

### 4.5 测试 / CI

公开仓库里两个项目的 CI 都比较薄:

- rl_sar 主要靠 GitHub Actions 编译 sanity(Ubuntu 20.04 / 22.04 + ROS1/ROS2 矩阵)。
- GR00T-WBC 有 `check_environment.py` 做 import-level smoke test,但没有"真的跑 policy 看动作合不合理"的 CI(依赖 GPU + LFS,自托管 runner 才行)。

robowbc 应该**比这两个项目做得更激进一点**——参考第 3.4 节的 headless unitree_mujoco CI 模式。

---

## 第 5 章:Teleop 输入方式

| 方式 | Rust 集成复杂度 | 延迟 | 可靠性 | 安装阻力 |
|---|---|---|---|---|
| **键盘**(rl_sar、GR00T 默认) | 低,`crossterm` / `device_query` | <5 ms | 高 | 0(笔记本自带) |
| 手柄 / 摇杆(Logitech F710 / PS4) | 中,`gilrs` crate | <10 ms | 高(除非无线掉包) | 低(USB 即插即用) |
| ROS2 cmd_vel | 高(要 zenoh-bridge-ros2dds + ROS2 client lib) | 5–20 ms | 中(取决于网络) | 中(要装 ROS2) |
| Web teleop(浏览器 → WebSocket → DDS) | 中高,`axum` / `tokio-tungstenite` + JS frontend | 20–100 ms | 中 | 低(用户只要浏览器) |
| VR / 动捕(PICO 4 + XRoboToolKit) | 极高 | 10–30 ms | 中(依赖 WiFi) | 极高(要 VR 硬件 + 校准) |

**rl_sar 的键盘实现**:直接用 termios 风格的非阻塞读 stdin(C++ 里),WASD 给 vx/vy,QE 给 wz,R 重置 FSM,Space 急停。GR00T 是同样思路(`O` 急停、`]` engage、`T` play motion)。

**robowbc v1 推荐:只发键盘一种**。理由:

1. 复刻 rl_sar / GR00T 的肌肉记忆,用户上手成本最低。
2. Rust 生态里 `crossterm` 跨平台(Linux/macOS/Windows)成熟,500 行代码搞定。
3. 不引入硬件依赖,开箱即用。
4. 后续加手柄是非破坏性扩展(同一个 trait `TeleopSource`)。

不要做"输入设备菜单"——这是经典的过度设计。

---

## 第 6 章:Docker 和部署策略

### 6.1 现状对比

- **rl_sar**:Docker 是**主要**部署方式之一(README 推荐),有 `docker/Dockerfile` 和 compose 文件。底层 base image 是 `osrf/ros:noetic-desktop-full` / `osrf/ros:humble-desktop-full`。
- **GR00T-WholeBodyControl**:Docker **不是**主要部署方式,README 推 `uv` 多 venv 直接装在 host(因为 PICO VR / Isaac Lab / TensorRT 这些组件 Dockerize 起来收益不大、坑很多)。
- **unitree_sdk2**:仓库自带 `.devcontainer/Dockerfile`,主要给开发用不是 production。

### 6.2 robowbc 的建议方案

**双轨**:

1. **Native install** (推荐 default):`cargo install robowbc-cli` + 手动 `apt install` cyclonedds 系统库(如果 cyclors 静态 vendor 不能满足)。理由:跟 ROS 2 / unitree_sdk2 生态最自然,no GPU/X11/multicast 坑。
2. **Docker 镜像** `robowbc/runtime:latest`(兜底,给用户快速试 demo):
   - Base:`nvidia/cuda:12.4-runtime-ubuntu22.04`(带 CUDA + ONNX Runtime CUDA EP)
   - 装 cyclonedds 0.10.x、unitree_mujoco、robowbc binary
   - **`network_mode: host`** 是必须的(DDS 多播)
   - GUI 要 X11 转发(`-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY` + host 上 `xhost +local:docker`)
   - **预烤 checkpoint vs 运行时下载**:v1 推荐**运行时从 HuggingFace 下载**(学 GR00T 的 `download_from_hf.py`),理由:(a) image size 不爆炸(SONIC 几百 MB,bfm_zero/wbc_agile/decoupled_wbc 加起来上 GB);(b) license 隔离(NVIDIA Open Model License 的 weight 不能直接 redistribute 进自己的 docker image,见第 8.6 节)。

### 6.3 NVIDIA / TensorRT 集成

如果要用 TensorRT 做 ONNX 加速(GR00T-WBC 的 C++ deploy 栈在 G1 onboard Jetson Orin NX 上跑就是这条路),robowbc Rust 侧推荐:

- **`ort` crate**(ONNX Runtime Rust binding),支持 CUDA EP 和 TensorRT EP。
- 放弃 `tract`(纯 Rust,但 NVIDIA path 不支持)。
- Memory pinning:`ort` 的 `Session::run_with_iobinding` 可以指定 pinned host memory,对应 GR00T 的 `cudaMallocHost` 优化。

---

## 第 7 章:安全层常见模式

### 7.1 控制循环里的位置

```
                    [ teleop input ]
                          │
                          ▼
          ┌──────────────────────────────┐
          │  cmd processor (clip vx/vy)  │  ← 安全层 1:teleop 输入限位
          └──────────────────────────────┘
                          │
                          ▼
          ┌──────────────────────────────┐
          │     policy (ONNX inference)  │
          └──────────────────────────────┘
                          │
                          ▼
          ┌──────────────────────────────┐
          │  output validator            │  ← 安全层 2:检测 NaN / 突变 / 发散
          │   - NaN guard                │
          │   - rate limit (Δq, Δdq)     │
          │   - divergence vs lowstate   │
          └──────────────────────────────┘
                          │
                          ▼
          ┌──────────────────────────────┐
          │  joint clip (hard limits)    │  ← 安全层 3:关节硬限位
          └──────────────────────────────┘
                          │
                          ▼
          ┌──────────────────────────────┐
          │  watchdog                    │  ← 安全层 4:心跳/超时
          │   - last_lowstate_age < 50ms │
          │   - if e-stop, override      │
          └──────────────────────────────┘
                          │
                          ▼ rt/lowcmd
```

### 7.2 G1 特定考虑

- **arm_sdk weight 必须渐进**:从 0 到 1 至少给 1 秒,否则手臂瞬间高速摆动可能撞自己或人。
- **CRC32 必须算**:少了机器人忽略整帧。
- **mode_machine 校验**:`g1::publisher::LowCmd` 有 `check_mode_machine()` 校验当前 FSM 状态允不允许这个 lowcmd(PR mode vs AB mode 不能混)。
- **遥控器 watchdog**:`unitree_sdk2` 默认 3000 ms 没收到遥控器消息就报 timeout。robowbc 不一定要触发 timeout,但要至少**能感知**遥控器是否在线(用户按急停的物理 fallback)。

### 7.3 rl_sar / GR00T 怎么做

- **rl_sar**:FSM 里 `StateRL_Running` 检查 IMU roll/pitch 超阈值就转 `StateInit`(机器人翻倒自动停)。键盘 `Space` 触发软件 e-stop。
- **GR00T**:`O` 键直接在 C++ deploy 主循环里 `exit()`;motor error monitor + TTS alert 是 26-03-24 update 加的。
- **没有现成的"policy 失控检测器"**——这是 robowbc 可以做出差异化的地方。

### 7.4 robowbc 推荐的 policy 失控检测

```rust
// 简化伪代码
struct PolicyValidator {
    last_q: [f32; 35],
    max_dq_per_step: f32,    // 比如 5 rad/s × dt = 0.01 rad
    divergence_threshold: f32,
}

impl PolicyValidator {
    fn validate(&mut self, q_target: &[f32; 35], q_current: &[f32; 35]) -> Result<(), Fault> {
        // 1. NaN guard
        if q_target.iter().any(|x| !x.is_finite()) {
            return Err(Fault::Nan);
        }
        // 2. Rate limit (跟上一帧 q_target 比)
        for i in 0..35 {
            if (q_target[i] - self.last_q[i]).abs() > self.max_dq_per_step {
                return Err(Fault::RateLimit(i));
            }
        }
        // 3. Divergence (跟当前 q 比)
        for i in 0..35 {
            if (q_target[i] - q_current[i]).abs() > self.divergence_threshold {
                return Err(Fault::Divergence(i));
            }
        }
        self.last_q = *q_target;
        Ok(())
    }
}
```

触发 `Fault` 后转 damping mode,可选向 host 报警 / log 到 rerun。

---

## 第 8 章:可能被忽略的维度(更广泛的探索)

### 8.1 Teleop 录制 / 回放

- **场景**:用户用键盘走了 30 秒,觉得"刚才那个转弯不错",想保存这段做 benchmark 或当 demo 数据。
- **建议实现**:runtime 的 cmd 总线挂一个 `Tee`,把 `(timestamp, cmd_vel, button_events)` 序列化成 MCAP / parquet / .rrd(rerun 原生格式)。回放时再加一个 `TeleopSource::Replay(file)` 实现,跟 `TeleopSource::Keyboard` 等价替换。
- **跟 rerun.io 配合**:rerun 0.25+ 实验性支持 MCAP reading,可以直接拖 `.mcap` 文件进 viewer 看回放。

> **注**:**runtime 里 policy 热切**(在不重启进程的情况下从 gear_sonic 切到 decoupled_wbc)已被明确**排除在支持目标之外**。policy 在 `RobowbcRuntime::new()` 时一次性 load,切换 policy 重启进程即可。这简化了 runtime 生命周期,避免了 crossfade buffer / dual policy state 这些复杂度。如果未来有强需求再独立设计。

### 8.2 跨机型 config 抽象

目标:加一个新机器人(Booster T1、Fourier GR-2)只是加一个 TOML 不是新代码。

```toml
# robots/booster_t1/robowbc.toml
[robot]
name = "booster_t1"
n_dof = 27
idl = "booster_proto"            # 需要新 IDL crate

[transport]
type = "cyclonedds"
domain_id = 0
lowcmd_topic = "rt/lowcmd"
lowstate_topic = "rt/lowstate"
state_freq_hz = 500

[joint_limits]
# 27 joints worth of (lower, upper, vel_max)
limits = [...]

[default_pose]
q = [0.0, 0.0, ...]  # 27 elements

[policy_compat]
# 每种 policy 输入的 obs 顺序、action 顺序映射
gear_sonic = { obs_map = "policy_g1_to_t1.json", action_map = "..." }
```

需要的代码层抽象:(a) `RobotSpec` trait;(b) `IdlCodec` trait(每个机型一个 IDL Rust crate);(c) `PolicyAdapter` trait(处理 obs/action 维度对齐)。**这套抽象在 v0/v1 不要急着做,等加第二个机器人时再 refactor**——过早抽象比 copy-paste 危险。

### 8.3 给未来 LeRobot 集成(Path B)留的接口

LeRobot(Hugging Face)是 manipulation-centric 的 Python 框架,跟 robowbc 的 locomotion-WBC 重心不同,但接口面有重叠:

- LeRobot 用 `LeRobotDataset` + parquet + episodes 的格式存数据。
- LeRobot 的 policy 通常输出 7-DoF / 14-DoF arm + gripper,不是全身 35 motor。

**最小接口面**:robowbc 提供一个 `RobotIO` Python wrapper(PyO3 bindings 暴露 `read_state() -> dict, write_action(dict)`),让 LeRobot 端能把 robowbc 当成一个 robot driver 用。这样 Path A 不堵 Path B:未来想做 manipulation 用 LeRobot 训,再用 robowbc 跑就行。

### 8.4 日志 / 可观测性

机器人工程师跑的时候真正想看:

- 关节的 q_target vs q_actual 时间序列(哪只腿没跟上?)
- IMU roll/pitch(要翻了吗?)
- Policy 输出的 raw value(NaN 预警)
- DDS 通信延迟(比如 lowstate_age)

**推荐 stack**:

- **rerun.io**:原生 Rust SDK,主线程 `rec.log("joints/q_target", ...)`,免费 viewer 可以本地看也可以 web stream。比 ROS2 bag + rqt 现代得多。
- **可选 fallback**:parquet 落盘 + jupyter 离线分析。
- **不推荐**:Prometheus / Grafana(不擅长高频时序)、ROS2 bag(绑死 ROS 生态)。

### 8.5 版本化 policy artifact 格式

一个"policy bundle"该包含什么,才能让别人能复现?

```
gear_sonic_v1.2/
├── manifest.toml              # checksum、版本、训练 git rev、依赖 robowbc 版本
├── encoder.onnx
├── decoder.onnx
├── observation_config.yaml    # obs 字段定义
├── action_config.yaml         # action 字段定义
├── pd_gains.yaml              # 推荐 PD(policy 训练时用的)
├── safety_limits.yaml         # 关节限位(policy author 标定的)
├── scene/
│   └── g1_29dof_with_hand.xml # 关联的 mjcf
├── LICENSE                    # NVIDIA Open Model License 或 Apache-2.0
└── README.md
```

打包成 `.tar.zst` 上 HuggingFace。robowbc 解析 `manifest.toml` 自动校验 checksum 和兼容性。

### 8.6 License 策略

robowbc **本身代码采用 MIT license**,是宽松度最高的常见 OSS 协议之一。但因为依赖栈和模型分发涉及多种不同协议,license 隔离需要做得清楚——直接放一个大 markdown 不够,要按依赖类型分文件管理。

#### 8.6.1 各组件 license 现状

| 组件 | License | 跟 MIT 共存 OK? | 备注 |
|---|---|---|---|
| **robowbc 自身代码** | **MIT** | — | 本仓库 |
| cyclors | Apache-2.0 | ✅ | dependency |
| cyclonedds-rs | Apache-2.0 | ✅ | alternative dependency |
| CycloneDDS (libcyclonedds) | EPL-2.0 OR BSD-3-Clause | ✅(走 BSD-3 路径) | cyclors / cyclonedds-rs 底层 |
| dust-dds | Apache-2.0 | ✅ | optional |
| zenoh / zenoh-bridge-* | Apache-2.0 | ✅ | optional |
| ort (ONNX Runtime Rust) | Apache-2.0 / MIT 双协议 | ✅ | dependency |
| crossterm | MIT | ✅ | dependency |
| rerun-rs | Apache-2.0 / MIT 双协议 | ✅ | dependency |
| Unitree SDK / unitree_sdk2 | BSD-3-Clause | ✅ | reference / IDL 来源 |
| **GEAR-SONIC weight** | **NVIDIA Open Model License** | ⚠️ 商用 OK 但需 attribution | **不打包进发行物** |
| decoupled_wbc / wbc_agile / bfm_zero weight | 各自不同(部分 CC BY-NC,部分 NVIDIA OML) | ⚠️ 部分非商用 | **不打包进发行物** |

#### 8.6.2 推荐的仓库结构

```
robowbc/
├── LICENSE                              # MIT,只覆盖本仓库代码
├── LICENSES/                            # 第三方协议汇总目录
│   ├── README.md                        # 索引:哪个组件用哪个协议、链接到原始声明
│   ├── cyclonedds-EPL-2.0.txt
│   ├── cyclonedds-BSD-3-Clause.txt
│   ├── cyclors-Apache-2.0.txt
│   ├── ort-Apache-2.0.txt
│   ├── ort-MIT.txt
│   ├── nvidia-open-model-license.txt
│   └── ...
├── docs/
│   └── third-party-notices.md           # 给最终用户看的简明声明
└── README.md                            # 顶部加一句 license 总结
```

**为什么用 `LICENSES/` 目录而不是单文件 `THIRD_PARTY_LICENSES.md`**:

- 工具链识别更好:`cargo about`、`licensee`、GitHub license detector 都按目录结构扫
- 未来某个第三方协议更新(比如 NVIDIA 出 OML v2),单独替换一个文件比改大 markdown 干净
- 跟 SPDX 风格一致,便于自动化生成 `cargo about generate licenses.html` 这类输出
- 添加新依赖时 PR 改动定位清晰(一个新文件,不是 10000 行 markdown 里加一段)

#### 8.6.3 关键执行策略

1. **robowbc binary / docker image 不打包任何 model weight**——一律运行时从 HuggingFace 下载(参考 GR00T 的 `download_from_hf.py` 风格),用户在第一次运行时显式接受对应 license。
2. README 顶部加一句简明声明:"robowbc itself is MIT-licensed; third-party dependencies and bundled policies retain their original licenses, see [LICENSES/](LICENSES/)."
3. 任何 PR 引入新依赖都需要在 `LICENSES/README.md` 里登记,CI 跑 `cargo about` 校验是否有 MIT-不兼容的协议偷偷溜进来(比如某个 dependency 升级到 GPL)。
4. `docs/third-party-notices.md` 用人话总结:哪些组件、哪些 license 类别(permissive / weak copyleft / model license)、用户在 redistribute 时要注意什么。

---

## 第 9 章:具体落地建议

### 9.1 4 周和 8 周的 Path A 排序里程碑

#### 4 周 milestone("sim 走起来")

| 周 | 任务 | 验收标准 |
|---|---|---|
| W1 | 搭 cyclors-based DDS transport spike,写 `unitree_hg::LowCmd_/LowState_` 的 Rust IDL crate(手写 CDR 序列化或调 `cyclonedds_idlc` 生成) | `cargo run --example pubsub` 跟 unitree_mujoco 能互通 `rt/lowstate` |
| W2 | 实现 `Policy` trait(gear_sonic ONNX 加载 + 推理,用 `ort` crate);实现 `PolicyValidator`(NaN/rate/divergence guard) | 仿真里 policy 推理 50 Hz,validator 触发 fault 能转 damping |
| W3 | 实现 keyboard teleop(`crossterm`)+ FSM(Init / RL_Init / RL_Running / Damping)+ rerun logging | 用户按 WASD 能让 G1 在 unitree_mujoco 里走、QE 转向、Space 急停 |
| W4 | 出 `cargo install robowbc-cli` / `docker compose up` 双路径,写 README + 录 demo gif;`LICENSES/` 目录就位 | 外部用户照着 README 5 分钟内 sim 跑起来 |

#### 8 周 milestone("真机走起来")

| 周 | 任务 | 验收标准 |
|---|---|---|
| W5 | unitree_mujoco headless CI(GitHub Actions self-hosted runner) | 每个 PR 自动跑 30 秒 sim smoke test |
| W6 | decoupled_wbc / wbc_agile / bfm_zero 适配(trait + config 切换) | 4 种 policy 在 sim 里都能 run,命令行切换 |
| W7 | 真 G1 第一次 bring-up(吊挂、debug mode、L2+R2、cyclonedds domain 0、enp3s0) | 真机原地踏步 + 慢速行走 |
| W8 | 真机安全性强化(watchdog 超时、CRC 校验、emergency stop 回归)+ docs | 团队外的 demo "1 行命令真机走起来" |

### 9.2 要在 MiaoDX/robowbc 里开的具体 GitHub issue

```
[issue]  1. DDS transport: cyclors-based Rust integration
   scope: 用 cyclors crate + 自写 unitree_hg IDL crate;包 publisher/subscriber;500 Hz 测延迟

[issue]  2. unitree_hg IDL Rust crate
   scope: 用 cyclonedds_idlc 从 unitree_sdk2 的 .idl 生成,或者手写 CDR;
          导出 LowCmd_ / LowState_ / HandCmd_ / HandState_

[issue]  3. ONNX inference via ort crate
   scope: gear_sonic encoder+decoder 双模型加载;CUDA EP / TensorRT EP 切换;pinned memory

[issue]  4. PolicyValidator: NaN / rate-limit / divergence guard
   scope: 见第 7.4 节代码草图;落 rerun log

[issue]  5. FSM (Init / RL_Init / RL_Running / Damping / Fault)
   scope: 复刻 rl_sar 的 FSMFactory 模式;每种 policy 注册自己的 FSM 状态。
          注:single policy per runtime instance,切换 policy 重启进程即可

[issue]  6. Keyboard teleop with crossterm
   scope: WASD/QE/Space/R 按键映射;30 ms 抖动以下

[issue]  7. policy/<robot>/<config>/ folder convention + base.yaml + config.yaml parser
   scope: 复刻 rl_sar 的 policy folder 思想;用 toml-rs / serde_yaml

[issue]  8. unitree_mujoco headless CI smoke test
   scope: GitHub Actions self-hosted runner;起 unitree_mujoco --headless;
          timeout 30s 跑 robowbc-run;assert no NaN + state evolves

[issue]  9. Real G1 bring-up runbook
   scope: 一份 SAFETY.md,吊挂步骤、L2+R2、网络配置、急停操作、第一次原地踏步的 SOP

[issue] 10. Docker image robowbc/runtime:latest
   scope: nvidia/cuda base + cyclonedds 0.10.x + robowbc binary;network_mode: host;
          X11 forward;明确不在 image 里 bake model weight,运行时从 HF 下载

[issue] 11. LICENSES/ 目录初始化 + 第三方协议登记
   scope: 创建 LICENSES/ 目录 + README 索引;填充当前依赖的协议文本
          (cyclors / cyclonedds / ort / crossterm / rerun / NVIDIA OML 等);
          CI 加 cargo about / licensee 检查;README 顶部加 license 声明;
          docs/third-party-notices.md 用人话写一份给最终用户看的总结

[issue] 12. rerun.io logging integration
   scope: q_target / q_actual / IMU / cmd_vel / policy_latency 5 个时间序列

[issue] 13. (stretch) zenoh feature flag + zenoh-bridge-ros2dds compose service
   scope: cargo feature "zenoh";预设 zenoh config 给远程 teleop 准备

[issue] 14. (stretch) dust-dds backend spike
   scope: 一周 timebox,验证能不能跟 unitree_mujoco 互通 LowState/LowCmd
```

### 9.3 先做哪个原型来消除最多风险

**第一个原型(Week 1,最多 5 个工作日)**:

> "cyclors + 手写 LowCmd_/LowState_ Rust struct + 一个 print-only 程序,启动 unitree_mujoco 后能 print 出 G1 的关节状态、并能往 rt/lowcmd 发一个 'hold default pose' 命令让仿真里的 G1 维持站立姿态。"

这个原型消除的风险(按重要性排序):

1. **wire-level DDS 兼容性**:cyclors 跟 unitree_mujoco 的 cyclonedds 能不能讲上。
2. **CRC32 算法是否正确**:发出去机器人不忽略。
3. **`unitree_hg` IDL CDR 序列化是否对齐**:35 个 motor、IMU、`mode_pr` / `mode_machine` 字段顺序。
4. **构建复杂度**:cyclors vendored cmake 在团队的开发机和 CI 上能不能编过。

如果这 5 天的原型跑通了,剩下所有工程都是机械活。如果跑不通,得切到 PyO3 fallback 路径。

### 9.4 最小可行的"机器人通过 robowbc 走起来"demo 规格

```bash
# 用户视角的完整体验
$ git clone https://github.com/MiaoDX/robowbc && cd robowbc
$ docker compose up -d unitree_mujoco           # 起仿真器(独立容器,DDS 在 lo:domain 1)
$ cargo run --release --bin robowbc-run -- \
      --policy gear_sonic \
      --robot g1 \
      --scene scene_29dof \
      --transport cyclonedds-sim \
      --teleop keyboard

[INFO] robowbc 0.1.0 (MIT license; see LICENSES/ for third-party notices)
[INFO] Loaded policy: gear_sonic v1.2 (NVIDIA Open Model License)
[INFO] DDS transport: cyclors → cyclonedds 0.10.x → domain=1, iface=lo
[INFO] Subscribing rt/lowstate (unitree_hg::LowState_)
[INFO] Publishing rt/lowcmd (unitree_hg::LowCmd_) @ 500 Hz
[INFO] Teleop: keyboard. Use WASD to move, QE to rotate, SPACE to e-stop.
[INFO] FSM: Init → RL_Init (interpolating to default_dof_pos)...
[INFO] FSM: RL_Init → RL_Running. Press SPACE to e-stop.

# 用户按 W:MuJoCo 窗口里 G1 向前走。按 SPACE:进 damping,原地不动。
```

**Done definition**:

1. 任何 Linux 用户从 git clone 到看到 G1 走,5 分钟内(不算 GPU 驱动安装)。
2. 不需要装 ROS。
3. 不需要装 Python venv(除了 unitree_mujoco 容器内部)。
4. 一个 cargo binary + 一个 docker compose,仅此而已。
5. README 里有动图截屏。
6. License 边界清晰:robowbc 是 MIT,SONIC weight 走 NVIDIA OML(运行时 banner 显式提示)。

---

## Caveats / 注意事项

1. **未亲自验证的关键互操作性**:本报告中"cyclors + unitree_mujoco wire-level 兼容"是基于 (a) cyclors 是 zenoh-bridge-dds 的底层并已被验证跟 ROS 2 / cyclonedds 互通,(b) unitree_mujoco 用的就是 cyclonedds 0.10.x 这两个事实**推断**的,**没有**找到一个公开 commit 或 issue 直接说"我用 cyclors 跑通了 unitree_hg::LowCmd_"。Week 1 的 spike 就是为了验证这个假设;如果证伪,要切到 cyclonedds-rs 或 PyO3 路径。
2. **dust-dds / RustDDS 与 Unitree IDL 的兼容性完全没有公开证据**。本报告推荐它们只作为 spike,不要当主路径。
3. **GR00T-WBC 是快速演进的项目**,本报告引用的若干 changelog 条目(如 2026-03-24 的 ZMQ protocol v4、2026-04-10 的 SONIC training code 公开、2026-04-27 的 MotionBricks preview)来自其文档站,可能在实际开发周期里又有 breaking change。绑版本时建议 pin 一个 git rev。
4. **Unitree G1 firmware 的"L1+A vs L2+B"按键组合在不同 firmware 版本里有变化**——上文给出的是当前主流文档说法,真机调试时务必参考机器人当前 firmware 对应的 user manual,否则按错键可能进入意外模式。
5. **真机验证有物理风险**。第 7 章给出的 4 层安全模式是必要不充分条件——bring-up 第一次必须吊挂、必须有人按急停、必须先在 unitree_mujoco 里把同一个 policy 验证 ≥30 分钟没有任何 fault 才能上真机。
6. **License 部分本人不是律师**,第 8.6 节的论述基于公开声明的常识理解,正式分发前请法务 review,特别是 NVIDIA Open Model License 的 attribution / commercial use 条款,以及 MIT 跟 EPL-2.0 / Apache-2.0 在 binary distribution 时的 notice 义务。
7. **本报告的"4 周 / 8 周里程碑"是基于 1–2 名熟悉 Rust + 机器人控制的工程师全职投入估算的**,如果团队还要并行做 policy 训练 / 数据采集,这个时间表会延长。
8. **报告中提到的若干性能数字**(如 zenoh 把 discovery 流量降低 97%–99.9%、SONIC 在 G1 上 100% success rate、SONIC + GR00T N1.5 95% pick-and-place 成功率)**全部来自项目方自己发的 blog / paper**,没有第三方独立复现的报告,看待时应当打折。
9. **ROS1 已经 EOL**,本报告推荐 robowbc 不支持 ROS1 主要基于这一点,但如果 MiaoDX 的客户端基础里有强 ROS1 依赖,要重新评估。
10. **键盘 teleop 在 Docker 容器里需要 TTY 透传**(`docker run -it`)才能拿到非阻塞键盘输入;compose 里要 `tty: true` + `stdin_open: true`。这是常见的小坑。
