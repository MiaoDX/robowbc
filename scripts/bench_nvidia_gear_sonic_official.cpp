#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "math_utils.hpp"
#include "policy_parameters.hpp"
#include "robot_parameters.hpp"

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kPlannerQposDim = 36;
constexpr std::size_t kPlannerJointOffset = 7;
constexpr std::size_t kPlannerContextLen = 4;
constexpr std::size_t kReplanIntervalTicks = 5;
constexpr std::size_t kAllowedPredNumTokens = 11;
constexpr std::array<std::int64_t, kAllowedPredNumTokens> kAllowedPredNumTokensMask = {
    0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
};
constexpr float kDefaultHeightMeters = 0.74f;
constexpr float kOfficialDefaultHeightMeters = 0.788740f;
constexpr float kDefaultHeightSentinel = -1.0f;
constexpr std::int64_t kDefaultModeWalk = 2;
constexpr std::int64_t kDefaultModeIdle = 0;
constexpr std::int64_t kDefaultModeSlowWalk = 1;
constexpr std::size_t kReferenceFutureFrames = 10;
constexpr std::size_t kReferenceFrameStep = 5;
constexpr float kControlDtSeconds = 1.0f / 50.0f;
constexpr float kPlannerInterpStep = 30.0f / 50.0f;
constexpr std::size_t kEncoderDim = 64;
constexpr std::size_t kEncoderObsDictDim = 1762;
constexpr std::size_t kDecoderObsDictDim = 994;
constexpr std::size_t kDecoderHistoryLen = 10;
constexpr std::size_t kLaterMotionProbeTick = 25;
constexpr std::size_t kEncoderModeOffset = 0;
constexpr std::size_t kEncoderMotionJointPositionsOffset = 4;
constexpr std::size_t kEncoderMotionJointVelocitiesOffset = 294;
constexpr std::size_t kEncoderMotionAnchorOrientationOffset = 601;

volatile float g_sink = 0.0f;

struct Options {
    std::string case_id;
    fs::path model_dir;
    fs::path output;
    std::optional<fs::path> dump_dir;
    int samples = 100;
    int ticks = 200;
    int control_frequency_hz = 50;
};

struct Observation {
    std::vector<float> joint_positions;
    std::vector<float> joint_velocities;
    std::array<float, 3> gravity_vector{};
    std::array<float, 3> angular_velocity{};
    std::array<double, 4> base_quat_wxyz{1.0, 0.0, 0.0, 0.0};
};

struct Twist {
    std::array<float, 3> linear{};
    std::array<float, 3> angular{};
};

struct PlannerCommand {
    std::int64_t mode = kDefaultModeIdle;
    float target_vel = kDefaultHeightSentinel;
    float height = kDefaultHeightSentinel;
    std::array<float, 3> movement_direction{0.0f, 0.0f, 0.0f};
    std::array<float, 3> facing_direction{1.0f, 0.0f, 0.0f};
};

float wrap_angle_rad(float angle) {
    while (angle > static_cast<float>(M_PI)) {
        angle -= 2.0f * static_cast<float>(M_PI);
    }
    while (angle < -static_cast<float>(M_PI)) {
        angle += 2.0f * static_cast<float>(M_PI);
    }
    return angle;
}

std::pair<float, float> bin_angle_to_8_directions(float angle) {
    constexpr float kBinSize = static_cast<float>(M_PI / 4.0);

    const float normalized = wrap_angle_rad(angle);
    int bin_index = static_cast<int>(std::lround(normalized / kBinSize));
    if (bin_index > 4) {
        bin_index -= 8;
    }
    if (bin_index < -4) {
        bin_index += 8;
    }

    float slow_walk_speed = 0.2f;
    switch (bin_index) {
        case 0:
        case 1:
        case -1:
            slow_walk_speed = 0.3f;
            break;
        case 2:
        case -2:
            slow_walk_speed = 0.35f;
            break;
        case 3:
        case -3:
            slow_walk_speed = 0.25f;
            break;
        case 4:
        case -4:
        default:
            slow_walk_speed = 0.2f;
            break;
    }

    return {static_cast<float>(bin_index) * kBinSize, slow_walk_speed};
}

PlannerCommand idle_planner_command() {
    return PlannerCommand{};
}

PlannerCommand derive_planner_command(float& facing_yaw_rad, const Twist& twist) {
    facing_yaw_rad = wrap_angle_rad(facing_yaw_rad + twist.angular[2] * kControlDtSeconds);
    const std::array<float, 3> facing_direction = {
        std::cos(facing_yaw_rad),
        std::sin(facing_yaw_rad),
        0.0f,
    };
    const float command_norm = std::sqrt(
        twist.linear[0] * twist.linear[0] + twist.linear[1] * twist.linear[1]
    );

    if (command_norm <= 0.01f) {
        return PlannerCommand{
            kDefaultModeIdle,
            kDefaultHeightSentinel,
            kDefaultHeightSentinel,
            {0.0f, 0.0f, 0.0f},
            facing_direction,
        };
    }

    const float local_movement_angle = std::atan2(twist.linear[1], twist.linear[0]);
    const auto [movement_angle, slow_walk_speed] =
        bin_angle_to_8_directions(facing_yaw_rad + local_movement_angle);

    return PlannerCommand{
        kDefaultModeSlowWalk,
        slow_walk_speed,
        kDefaultHeightSentinel,
        {std::cos(movement_angle), std::sin(movement_angle), 0.0f},
        facing_direction,
    };
}

std::vector<float> default_pose() {
    std::vector<float> pose;
    pose.reserve(G1_NUM_MOTOR);
    for (double angle : default_angles) {
        pose.push_back(static_cast<float>(angle));
    }
    return pose;
}

std::vector<float> mujoco_to_isaaclab_values(const std::vector<float>& values) {
    if (values.size() != G1_NUM_MOTOR) {
        throw std::runtime_error("values must match G1_NUM_MOTOR");
    }

    std::vector<float> remapped(G1_NUM_MOTOR, 0.0f);
    for (std::size_t mujoco_index = 0; mujoco_index < G1_NUM_MOTOR; ++mujoco_index) {
        remapped[static_cast<std::size_t>(isaaclab_to_mujoco[mujoco_index])] =
            values[mujoco_index];
    }
    return remapped;
}

std::vector<float> mujoco_to_isaaclab_positions(const std::vector<float>& joint_positions) {
    return mujoco_to_isaaclab_values(joint_positions);
}

std::vector<float> mujoco_to_isaaclab_joint_offsets(const std::vector<float>& joint_positions) {
    auto remapped = mujoco_to_isaaclab_values(joint_positions);
    for (std::size_t mujoco_index = 0; mujoco_index < G1_NUM_MOTOR; ++mujoco_index) {
        remapped[static_cast<std::size_t>(isaaclab_to_mujoco[mujoco_index])] -=
            static_cast<float>(default_angles[mujoco_index]);
    }
    return remapped;
}

std::vector<float> isaaclab_to_mujoco_values(const std::vector<float>& values) {
    if (values.size() != G1_NUM_MOTOR) {
        throw std::runtime_error("values must match G1_NUM_MOTOR");
    }

    std::vector<float> remapped(G1_NUM_MOTOR, 0.0f);
    for (std::size_t mujoco_index = 0; mujoco_index < G1_NUM_MOTOR; ++mujoco_index) {
        remapped[mujoco_index] = values[static_cast<std::size_t>(isaaclab_to_mujoco[mujoco_index])];
    }
    return remapped;
}

Observation zero_tracking_observation() {
    Observation obs;
    obs.joint_positions.assign(G1_NUM_MOTOR, 0.0f);
    obs.joint_velocities.assign(G1_NUM_MOTOR, 0.0f);
    obs.gravity_vector = {0.0f, 0.0f, -1.0f};
    obs.angular_velocity = {0.0f, 0.0f, 0.0f};
    return obs;
}

Observation standing_velocity_observation() {
    Observation obs;
    obs.joint_positions = default_pose();
    obs.joint_velocities.assign(G1_NUM_MOTOR, 0.0f);
    obs.gravity_vector = {0.0f, 0.0f, -1.0f};
    obs.angular_velocity = {0.0f, 0.0f, 0.0f};
    return obs;
}

std::string json_escape(const std::string& input) {
    std::string escaped;
    escaped.reserve(input.size());
    for (char ch : input) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }
    return escaped;
}

std::vector<std::string> session_names(Ort::Session& session, bool inputs) {
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
    std::vector<std::string> names;
    names.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        auto name = inputs ? session.GetInputNameAllocated(index, allocator)
                           : session.GetOutputNameAllocated(index, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

void require_name(const std::vector<std::string>& names, std::string_view needle, std::string_view kind) {
    if (std::find(names.begin(), names.end(), needle) == names.end()) {
        throw std::runtime_error(
            "missing required " + std::string(kind) + " tensor '" + std::string(needle) + "'"
        );
    }
}

Ort::Session make_session(Ort::Env& env, const fs::path& model_path) {
    if (!fs::is_regular_file(model_path)) {
        throw std::runtime_error("model file does not exist: " + model_path.string());
    }

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    const std::string model_path_str = model_path.string();
    return Ort::Session(env, model_path_str.c_str(), options);
}

struct PlannerState {
    std::deque<std::vector<float>> context;
    std::vector<std::vector<float>> trajectory;
    std::size_t traj_index = 0;
    float interp_phase = 0.0f;
    std::size_t steps_since_plan = kReplanIntervalTicks;
    std::vector<float> last_context_frame;

    PlannerState()
        : context(), trajectory(), last_context_frame()
    {
        reset();
    }

    void reset() {
        const auto standing = make_standing_qpos();
        context.clear();
        for (std::size_t index = 0; index < kPlannerContextLen; ++index) {
            context.push_back(standing);
        }
        trajectory.clear();
        traj_index = 0;
        interp_phase = 0.0f;
        steps_since_plan = kReplanIntervalTicks;
        last_context_frame = standing;
    }

    static std::vector<float> make_standing_qpos() {
        std::vector<float> qpos(kPlannerQposDim, 0.0f);
        qpos[2] = kDefaultHeightMeters;
        qpos[3] = 1.0f;
        const auto pose = default_pose();
        std::copy(pose.begin(), pose.end(), qpos.begin() + static_cast<std::ptrdiff_t>(kPlannerJointOffset));
        return qpos;
    }
};

struct TrackingState {
    std::deque<std::array<float, 3>> gravity;
    std::deque<std::array<float, 3>> angular_velocity;
    std::deque<std::vector<float>> joint_positions;
    std::deque<std::vector<float>> joint_velocities;
    std::deque<std::vector<float>> last_actions;

    TrackingState()
        : gravity(), angular_velocity(), joint_positions(), joint_velocities(), last_actions()
    {
        reset();
    }

    void reset() {
        gravity.clear();
        angular_velocity.clear();
        joint_positions.clear();
        joint_velocities.clear();
        last_actions.clear();
        for (std::size_t index = 0; index < kDecoderHistoryLen; ++index) {
            gravity.push_back({0.0f, 0.0f, 1.0f});
            angular_velocity.push_back({0.0f, 0.0f, 0.0f});
            joint_positions.emplace_back(G1_NUM_MOTOR, 0.0f);
            joint_velocities.emplace_back(G1_NUM_MOTOR, 0.0f);
            last_actions.emplace_back(G1_NUM_MOTOR, 0.0f);
        }
    }

    void push(const Observation& obs, const std::vector<float>& actions) {
        if (gravity.size() >= kDecoderHistoryLen) {
            gravity.pop_front();
            angular_velocity.pop_front();
            joint_positions.pop_front();
            joint_velocities.pop_front();
            last_actions.pop_front();
        }
        gravity.push_back(obs.gravity_vector);
        angular_velocity.push_back(obs.angular_velocity);
        joint_positions.push_back(mujoco_to_isaaclab_joint_offsets(obs.joint_positions));
        joint_velocities.push_back(mujoco_to_isaaclab_values(obs.joint_velocities));
        last_actions.push_back(actions);
    }
};

class GearSonicOfficialHarness {
public:
    explicit GearSonicOfficialHarness(
        const fs::path& model_dir,
        std::optional<fs::path> dump_dir = std::nullopt
    )
        : env_(ORT_LOGGING_LEVEL_WARNING, "gear_sonic_official"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          encoder_(make_session(env_, model_dir / "model_encoder.onnx")),
          decoder_(make_session(env_, model_dir / "model_decoder.onnx")),
          planner_(make_session(env_, model_dir / "planner_sonic.onnx")),
          planner_state_(),
          tracking_state_(),
          velocity_obs_(standing_velocity_observation()),
          tracking_obs_(zero_tracking_observation()),
          latest_action_(G1_NUM_MOTOR, 0.0f),
          dump_dir_(std::move(dump_dir))
    {
        validate_contracts();
    }

    void reset() {
        planner_state_.reset();
        tracking_state_.reset();
        velocity_obs_ = standing_velocity_observation();
        tracking_obs_ = zero_tracking_observation();
        latest_action_.assign(G1_NUM_MOTOR, 0.0f);
        dumped_tracking_tensors_ = false;
    }

    std::vector<float> velocity_tick() {
        const auto context_frame = planner_context_frame(planner_state_.last_context_frame, velocity_obs_);
        if (planner_state_.context.size() >= kPlannerContextLen) {
            planner_state_.context.pop_front();
        }
        planner_state_.context.push_back(context_frame);
        planner_state_.last_context_frame = context_frame;

        if (planner_state_.steps_since_plan >= kReplanIntervalTicks || planner_state_.trajectory.empty()) {
            planner_state_.trajectory = run_planner(planner_state_.context, velocity_command());
            planner_state_.traj_index = 0;
            planner_state_.interp_phase = 0.0f;
            planner_state_.steps_since_plan = 0;
        }

        planner_state_.steps_since_plan += 1;
        const auto frame = next_planner_frame(planner_state_);
        planner_state_.last_context_frame = frame;
        return std::vector<float>(
            frame.begin() + static_cast<std::ptrdiff_t>(kPlannerJointOffset),
            frame.end()
        );
    }

    std::vector<float> tracking_tick() {
        const auto encoder_obs = build_encoder_obs_dict();
        const auto tokens = run_single_f32(encoder_, "obs_dict", "encoded_tokens", encoder_obs, kEncoderObsDictDim);
        if (tokens.size() != kEncoderDim) {
            throw std::runtime_error(
                "encoder output dimension mismatch: expected " + std::to_string(kEncoderDim) +
                ", got " + std::to_string(tokens.size())
            );
        }

        tracking_state_.push(tracking_obs_, latest_action_);
        const auto decoder_obs = build_decoder_obs_dict(tokens);
        const auto raw_actions = run_single_f32(decoder_, "obs_dict", "action", decoder_obs, kDecoderObsDictDim);
        if (raw_actions.size() != G1_NUM_MOTOR) {
            throw std::runtime_error(
                "decoder output dimension mismatch: expected " + std::to_string(G1_NUM_MOTOR) +
                ", got " + std::to_string(raw_actions.size())
            );
        }
        maybe_dump_tracking_tensors(encoder_obs, tokens, decoder_obs, raw_actions);

        std::vector<float> positions(G1_NUM_MOTOR, 0.0f);
        for (int mujoco_index = 0; mujoco_index < G1_NUM_MOTOR; ++mujoco_index) {
            const int isaaclab_index = isaaclab_to_mujoco[static_cast<std::size_t>(mujoco_index)];
            const float action = raw_actions[static_cast<std::size_t>(isaaclab_index)];
            const float scaled = action * static_cast<float>(g1_action_scale[static_cast<std::size_t>(mujoco_index)]);
            positions[static_cast<std::size_t>(mujoco_index)] =
                static_cast<float>(default_angles[static_cast<std::size_t>(mujoco_index)]) + scaled;
        }

        latest_action_ = raw_actions;
        return positions;
    }

    std::vector<float> velocity_first_live_replan_dump() {
        if (!dump_dir_.has_value()) {
            throw std::runtime_error("--dump-dir is required for gear_sonic_velocity/first_live_replan_dump");
        }

        reset();
        tracking_state_.reset();
        latest_action_.assign(G1_NUM_MOTOR, 0.0f);

        std::deque<std::vector<float>> context;
        const auto standing = make_official_standing_qpos();
        for (std::size_t index = 0; index < kPlannerContextLen; ++index) {
            context.push_back(standing);
        }

        const auto idle_planned_30hz = run_planner_command(context, idle_planner_command());
        const auto bootstrap_motion_50hz = resample_planner_trajectory_to_50hz(idle_planned_30hz);

        constexpr std::size_t kFirstLiveReplanTick = kReplanIntervalTicks;
        const auto planner_context =
            rebuild_planner_context_from_motion(bootstrap_motion_50hz, kFirstLiveReplanTick);

        float facing_yaw_rad = 0.0f;
        Twist twist;
        twist.linear = {0.6f, 0.0f, 0.0f};
        twist.angular = {0.0f, 0.0f, 0.0f};
        const auto live_command = derive_planner_command(facing_yaw_rad, twist);
        const auto live_planned_30hz = run_planner_command(planner_context, live_command);
        const auto live_planned_50hz = resample_planner_trajectory_to_50hz(live_planned_30hz);
        const auto committed_motion_50hz = blend_planner_motion(
            bootstrap_motion_50hz,
            kFirstLiveReplanTick,
            kFirstLiveReplanTick,
            live_planned_50hz
        );
        const auto committed_joint_velocities =
            compute_motion_joint_velocities_isaaclab(committed_motion_50hz);

        const std::array<double, 4> init_base_quat_wxyz = {1.0, 0.0, 0.0, 0.0};
        const auto init_ref_root_quat_wxyz =
            planner_frame_root_quaternion(committed_motion_50hz.front());
        const auto encoder_obs = build_velocity_encoder_obs_dict(
            committed_motion_50hz,
            committed_joint_velocities,
            0,
            init_base_quat_wxyz,
            init_base_quat_wxyz,
            init_ref_root_quat_wxyz
        );
        const auto tokens = run_single_f32(
            encoder_,
            "obs_dict",
            "encoded_tokens",
            encoder_obs,
            kEncoderObsDictDim
        );
        if (tokens.size() != kEncoderDim) {
            throw std::runtime_error(
                "encoder output dimension mismatch: expected " + std::to_string(kEncoderDim) +
                ", got " + std::to_string(tokens.size())
            );
        }

        tracking_state_.push(tracking_obs_, latest_action_);
        const auto decoder_obs = build_decoder_obs_dict(tokens);
        const auto raw_actions = run_single_f32(
            decoder_,
            "obs_dict",
            "action",
            decoder_obs,
            kDecoderObsDictDim
        );
        if (raw_actions.size() != G1_NUM_MOTOR) {
            throw std::runtime_error(
                "decoder output dimension mismatch: expected " + std::to_string(G1_NUM_MOTOR) +
                ", got " + std::to_string(raw_actions.size())
            );
        }

        const std::vector<float> planner_command = {
            static_cast<float>(live_command.mode),
            live_command.target_vel,
            live_command.height,
            live_command.movement_direction[0],
            live_command.movement_direction[1],
            live_command.movement_direction[2],
            live_command.facing_direction[0],
            live_command.facing_direction[1],
            live_command.facing_direction[2],
        };

        write_matrix_json(*dump_dir_ / "bootstrap_motion_50hz.json", bootstrap_motion_50hz);
        write_matrix_json(*dump_dir_ / "planner_context.json", deque_to_rows(planner_context));
        write_vector_json(*dump_dir_ / "planner_command.json", planner_command);
        write_matrix_json(*dump_dir_ / "planner_motion_30hz.json", live_planned_30hz);
        write_matrix_json(*dump_dir_ / "planner_motion_50hz.json", live_planned_50hz);
        write_matrix_json(
            *dump_dir_ / "planner_motion_50hz_committed.json",
            committed_motion_50hz
        );
        write_matrix_json(
            *dump_dir_ / "planner_joint_velocities_50hz.json",
            committed_joint_velocities
        );
        write_vector_json(*dump_dir_ / "velocity_encoder_obs.json", encoder_obs);
        write_vector_json(*dump_dir_ / "velocity_tokens.json", tokens);
        write_vector_json(*dump_dir_ / "velocity_decoder_obs.json", decoder_obs);
        write_vector_json(*dump_dir_ / "velocity_raw_actions.json", raw_actions);

        std::vector<float> positions(G1_NUM_MOTOR, 0.0f);
        for (int mujoco_index = 0; mujoco_index < G1_NUM_MOTOR; ++mujoco_index) {
            const int isaaclab_index = isaaclab_to_mujoco[static_cast<std::size_t>(mujoco_index)];
            const float action = raw_actions[static_cast<std::size_t>(isaaclab_index)];
            const float scaled =
                action * static_cast<float>(g1_action_scale[static_cast<std::size_t>(mujoco_index)]);
            positions[static_cast<std::size_t>(mujoco_index)] =
                static_cast<float>(default_angles[static_cast<std::size_t>(mujoco_index)]) + scaled;
        }

        latest_action_ = raw_actions;
        return positions;
    }

    std::vector<float> velocity_later_motion_dump() {
        if (!dump_dir_.has_value()) {
            throw std::runtime_error("--dump-dir is required for gear_sonic_velocity/later_motion_dump");
        }

        reset();
        tracking_state_.reset();
        latest_action_.assign(G1_NUM_MOTOR, 0.0f);

        std::deque<std::vector<float>> context;
        const auto standing = make_official_standing_qpos();
        for (std::size_t index = 0; index < kPlannerContextLen; ++index) {
            context.push_back(standing);
        }

        const auto idle_planned_30hz = run_planner_command(context, idle_planner_command());
        const auto bootstrap_motion_50hz = resample_planner_trajectory_to_50hz(idle_planned_30hz);

        constexpr std::size_t kFirstLiveReplanTick = kReplanIntervalTicks;
        const auto planner_context =
            rebuild_planner_context_from_motion(bootstrap_motion_50hz, kFirstLiveReplanTick);

        float facing_yaw_rad = 0.0f;
        Twist twist;
        twist.linear = {0.6f, 0.0f, 0.0f};
        twist.angular = {0.0f, 0.0f, 0.0f};
        const auto live_command = derive_planner_command(facing_yaw_rad, twist);
        const auto live_planned_30hz = run_planner_command(planner_context, live_command);
        const auto live_planned_50hz = resample_planner_trajectory_to_50hz(live_planned_30hz);
        const auto committed_motion_50hz = blend_planner_motion(
            bootstrap_motion_50hz,
            kFirstLiveReplanTick,
            kFirstLiveReplanTick,
            live_planned_50hz
        );
        const auto committed_joint_velocities =
            compute_motion_joint_velocities_isaaclab(committed_motion_50hz);
        if (committed_motion_50hz.size() <= kLaterMotionProbeTick) {
            throw std::runtime_error("later motion probe tick exceeds committed motion length");
        }

        const std::array<double, 4> init_base_quat_wxyz = {1.0, 0.0, 0.0, 0.0};
        const auto init_ref_root_quat_wxyz =
            planner_frame_root_quaternion(committed_motion_50hz.front());

        Observation probe_obs;
        std::vector<float> encoder_obs;
        std::vector<float> tokens;
        std::vector<float> decoder_obs;
        std::vector<float> raw_actions;
        for (std::size_t tick = 0; tick <= kLaterMotionProbeTick; ++tick) {
            probe_obs = motion_observation_from_planner_frame(
                committed_motion_50hz,
                committed_joint_velocities,
                tick
            );
            encoder_obs = build_velocity_encoder_obs_dict(
                committed_motion_50hz,
                committed_joint_velocities,
                tick,
                probe_obs.base_quat_wxyz,
                init_base_quat_wxyz,
                init_ref_root_quat_wxyz
            );
            tokens = run_single_f32(
                encoder_,
                "obs_dict",
                "encoded_tokens",
                encoder_obs,
                kEncoderObsDictDim
            );
            if (tokens.size() != kEncoderDim) {
                throw std::runtime_error(
                    "encoder output dimension mismatch: expected " + std::to_string(kEncoderDim) +
                    ", got " + std::to_string(tokens.size())
                );
            }

            tracking_state_.push(probe_obs, latest_action_);
            decoder_obs = build_decoder_obs_dict(tokens);
            raw_actions = run_single_f32(
                decoder_,
                "obs_dict",
                "action",
                decoder_obs,
                kDecoderObsDictDim
            );
            if (raw_actions.size() != G1_NUM_MOTOR) {
                throw std::runtime_error(
                    "decoder output dimension mismatch: expected " + std::to_string(G1_NUM_MOTOR) +
                    ", got " + std::to_string(raw_actions.size())
                );
            }
            latest_action_ = raw_actions;
        }

        write_vector_json(
            *dump_dir_ / "velocity_probe_tick.json",
            std::vector<float>{static_cast<float>(kLaterMotionProbeTick)}
        );
        write_vector_json(*dump_dir_ / "current_joint_positions_mujoco.json", probe_obs.joint_positions);
        write_vector_json(*dump_dir_ / "current_joint_velocities_mujoco.json", probe_obs.joint_velocities);
        write_vector_json(
            *dump_dir_ / "current_base_quat_wxyz.json",
            std::vector<float>{
                static_cast<float>(probe_obs.base_quat_wxyz[0]),
                static_cast<float>(probe_obs.base_quat_wxyz[1]),
                static_cast<float>(probe_obs.base_quat_wxyz[2]),
                static_cast<float>(probe_obs.base_quat_wxyz[3]),
            }
        );
        write_vector_json(
            *dump_dir_ / "current_gravity.json",
            std::vector<float>{
                probe_obs.gravity_vector[0],
                probe_obs.gravity_vector[1],
                probe_obs.gravity_vector[2],
            }
        );
        write_vector_json(
            *dump_dir_ / "current_angular_velocity.json",
            std::vector<float>{
                probe_obs.angular_velocity[0],
                probe_obs.angular_velocity[1],
                probe_obs.angular_velocity[2],
            }
        );
        write_matrix_json(
            *dump_dir_ / "history_joint_positions_isaaclab_offsets.json",
            deque_to_rows(tracking_state_.joint_positions)
        );
        write_matrix_json(
            *dump_dir_ / "history_joint_velocities_isaaclab.json",
            deque_to_rows(tracking_state_.joint_velocities)
        );
        write_matrix_json(
            *dump_dir_ / "history_last_actions.json",
            deque_to_rows(tracking_state_.last_actions)
        );
        write_matrix_json(
            *dump_dir_ / "history_gravity.json",
            deque_to_rows(tracking_state_.gravity)
        );
        write_matrix_json(
            *dump_dir_ / "history_angular_velocity.json",
            deque_to_rows(tracking_state_.angular_velocity)
        );
        write_vector_json(*dump_dir_ / "velocity_encoder_obs.json", encoder_obs);
        write_vector_json(*dump_dir_ / "velocity_tokens.json", tokens);
        write_vector_json(*dump_dir_ / "velocity_decoder_obs.json", decoder_obs);
        write_vector_json(*dump_dir_ / "velocity_raw_actions.json", raw_actions);

        std::vector<float> positions(G1_NUM_MOTOR, 0.0f);
        for (int mujoco_index = 0; mujoco_index < G1_NUM_MOTOR; ++mujoco_index) {
            const int isaaclab_index = isaaclab_to_mujoco[static_cast<std::size_t>(mujoco_index)];
            const float action = raw_actions[static_cast<std::size_t>(isaaclab_index)];
            const float scaled =
                action * static_cast<float>(g1_action_scale[static_cast<std::size_t>(mujoco_index)]);
            positions[static_cast<std::size_t>(mujoco_index)] =
                static_cast<float>(default_angles[static_cast<std::size_t>(mujoco_index)]) + scaled;
        }

        return positions;
    }

private:
    Ort::Env env_;
    Ort::MemoryInfo memory_info_;
    Ort::Session encoder_;
    Ort::Session decoder_;
    Ort::Session planner_;
    PlannerState planner_state_;
    TrackingState tracking_state_;
    Observation velocity_obs_;
    Observation tracking_obs_;
    std::vector<float> latest_action_;
    std::optional<fs::path> dump_dir_;
    bool dumped_tracking_tensors_ = false;

    void validate_contracts() {
        const auto planner_inputs = session_names(planner_, true);
        const auto planner_outputs = session_names(planner_, false);
        for (const auto* name : {
                 "context_mujoco_qpos",
                 "target_vel",
                 "mode",
                 "movement_direction",
                 "facing_direction",
                 "random_seed",
                 "has_specific_target",
                 "specific_target_positions",
                 "specific_target_headings",
                 "allowed_pred_num_tokens",
                 "height",
             }) {
            require_name(planner_inputs, name, "planner input");
        }
        require_name(planner_outputs, "mujoco_qpos", "planner output");
        require_name(planner_outputs, "num_pred_frames", "planner output");

        const auto encoder_inputs = session_names(encoder_, true);
        const auto encoder_outputs = session_names(encoder_, false);
        require_name(encoder_inputs, "obs_dict", "encoder input");
        require_name(encoder_outputs, "encoded_tokens", "encoder output");

        const auto decoder_inputs = session_names(decoder_, true);
        const auto decoder_outputs = session_names(decoder_, false);
        require_name(decoder_inputs, "obs_dict", "decoder input");
        require_name(decoder_outputs, "action", "decoder output");
    }

    std::vector<float> planner_context_frame(
        const std::vector<float>& template_frame,
        const Observation& obs
    ) const {
        std::vector<float> frame =
            template_frame.size() == kPlannerQposDim ? template_frame : std::vector<float>(kPlannerQposDim, 0.0f);
        if (frame[2] == 0.0f) {
            frame[2] = kDefaultHeightMeters;
        }
        frame[3] = static_cast<float>(obs.base_quat_wxyz[0]);
        frame[4] = static_cast<float>(obs.base_quat_wxyz[1]);
        frame[5] = static_cast<float>(obs.base_quat_wxyz[2]);
        frame[6] = static_cast<float>(obs.base_quat_wxyz[3]);
        std::copy(obs.joint_positions.begin(), obs.joint_positions.end(), frame.begin() + static_cast<std::ptrdiff_t>(kPlannerJointOffset));
        return frame;
    }

    Twist velocity_command() const {
        Twist twist;
        twist.linear = {0.3f, 0.0f, 0.0f};
        twist.angular = {0.0f, 0.0f, 0.0f};
        return twist;
    }

    static std::vector<float> make_official_standing_qpos() {
        std::vector<float> qpos(kPlannerQposDim, 0.0f);
        qpos[2] = kOfficialDefaultHeightMeters;
        qpos[3] = 1.0f;
        const auto pose = default_pose();
        std::copy(
            pose.begin(),
            pose.end(),
            qpos.begin() + static_cast<std::ptrdiff_t>(kPlannerJointOffset)
        );
        return qpos;
    }

    std::vector<std::vector<float>> run_planner(
        const std::deque<std::vector<float>>& context,
        const Twist& twist
    ) {
        const float cmd_norm =
            std::sqrt(twist.linear[0] * twist.linear[0] + twist.linear[1] * twist.linear[1]);
        const std::array<float, 3> movement_direction =
            cmd_norm > 1e-6f
                ? std::array<float, 3>{
                      twist.linear[0] / cmd_norm,
                      twist.linear[1] / cmd_norm,
                      0.0f,
                  }
                : std::array<float, 3>{1.0f, 0.0f, 0.0f};
        const float yaw = twist.angular[2];
        const std::array<float, 3> facing_direction = {
            std::cos(yaw),
            std::sin(yaw),
            0.0f,
        };
        return run_planner_command(
            context,
            PlannerCommand{
                kDefaultModeWalk,
                cmd_norm,
                kDefaultHeightMeters,
                movement_direction,
                facing_direction,
            }
        );
    }

    std::vector<std::vector<float>> run_planner_command(
        const std::deque<std::vector<float>>& context,
        const PlannerCommand& command
    ) {
        std::vector<float> context_data;
        context_data.reserve(context.size() * kPlannerQposDim);
        for (const auto& frame : context) {
            context_data.insert(context_data.end(), frame.begin(), frame.end());
        }

        const std::array<float, 1> target_vel = {command.target_vel};
        const std::array<std::int64_t, 1> mode = {command.mode};
        const std::array<float, 1> height = {command.height};
        const std::array<std::int64_t, 1> random_seed = {0};
        const std::array<std::int64_t, 1> has_specific_target = {0};
        const std::vector<float> specific_target_positions(12, 0.0f);
        const std::vector<float> specific_target_headings(4, 0.0f);
        const auto allowed_pred_num_tokens = kAllowedPredNumTokensMask;

        const std::array<std::int64_t, 3> context_shape = {
            1,
            static_cast<std::int64_t>(kPlannerContextLen),
            static_cast<std::int64_t>(kPlannerQposDim),
        };
        const std::array<std::int64_t, 2> vec3_shape = {1, 3};
        const std::array<std::int64_t, 1> scalar_shape = {1};
        const std::array<std::int64_t, 2> has_target_shape = {1, 1};
        const std::array<std::int64_t, 3> specific_positions_shape = {1, 4, 3};
        const std::array<std::int64_t, 2> specific_headings_shape = {1, 4};
        const std::array<std::int64_t, 2> allowed_tokens_shape = {
            1,
            static_cast<std::int64_t>(kAllowedPredNumTokens),
        };

        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(11);
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            context_data.data(),
            context_data.size(),
            context_shape.data(),
            context_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(target_vel.data()),
            target_vel.size(),
            scalar_shape.data(),
            scalar_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(
            memory_info_,
            const_cast<std::int64_t*>(mode.data()),
            mode.size(),
            scalar_shape.data(),
            scalar_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(command.movement_direction.data()),
            command.movement_direction.size(),
            vec3_shape.data(),
            vec3_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(command.facing_direction.data()),
            command.facing_direction.size(),
            vec3_shape.data(),
            vec3_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(
            memory_info_,
            const_cast<std::int64_t*>(random_seed.data()),
            random_seed.size(),
            scalar_shape.data(),
            scalar_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(
            memory_info_,
            const_cast<std::int64_t*>(has_specific_target.data()),
            has_specific_target.size(),
            has_target_shape.data(),
            has_target_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(specific_target_positions.data()),
            specific_target_positions.size(),
            specific_positions_shape.data(),
            specific_positions_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(specific_target_headings.data()),
            specific_target_headings.size(),
            specific_headings_shape.data(),
            specific_headings_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(
            memory_info_,
            const_cast<std::int64_t*>(allowed_pred_num_tokens.data()),
            allowed_pred_num_tokens.size(),
            allowed_tokens_shape.data(),
            allowed_tokens_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(height.data()),
            height.size(),
            scalar_shape.data(),
            scalar_shape.size()
        ));

        static constexpr const char* kPlannerInputNames[] = {
            "context_mujoco_qpos",
            "target_vel",
            "mode",
            "movement_direction",
            "facing_direction",
            "random_seed",
            "has_specific_target",
            "specific_target_positions",
            "specific_target_headings",
            "allowed_pred_num_tokens",
            "height",
        };
        static constexpr const char* kPlannerOutputNames[] = {
            "mujoco_qpos",
            "num_pred_frames",
        };

        auto outputs = planner_.Run(
            Ort::RunOptions{nullptr},
            kPlannerInputNames,
            input_tensors.data(),
            input_tensors.size(),
            kPlannerOutputNames,
            std::size(kPlannerOutputNames)
        );

        const Ort::Value& mujoco_qpos = outputs[0];
        const Ort::Value& num_pred_frames = outputs[1];
        const auto qpos_info = mujoco_qpos.GetTensorTypeAndShapeInfo();
        const auto qpos_shape = qpos_info.GetShape();
        const std::size_t available_frames = qpos_info.GetElementCount() / kPlannerQposDim;
        const float* qpos_data = mujoco_qpos.GetTensorData<float>();

        std::int64_t predicted_frames = 1;
        const auto frames_info = num_pred_frames.GetTensorTypeAndShapeInfo();
        if (frames_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            predicted_frames =
                static_cast<std::int64_t>(num_pred_frames.GetTensorData<std::int32_t>()[0]);
        } else if (frames_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            predicted_frames = num_pred_frames.GetTensorData<std::int64_t>()[0];
        } else {
            throw std::runtime_error("planner num_pred_frames output must be int32 or int64");
        }
        predicted_frames = std::max<std::int64_t>(predicted_frames, 1);
        const std::size_t frame_count =
            std::min<std::size_t>(static_cast<std::size_t>(predicted_frames), available_frames);

        if (!(qpos_shape.size() == 3 || qpos_shape.size() == 2)) {
            throw std::runtime_error("unexpected planner mujoco_qpos rank");
        }

        std::vector<std::vector<float>> trajectory;
        trajectory.reserve(frame_count);
        for (std::size_t frame = 0; frame < frame_count; ++frame) {
            const float* begin = qpos_data + frame * kPlannerQposDim;
            trajectory.emplace_back(
                begin,
                begin + static_cast<std::ptrdiff_t>(kPlannerQposDim)
            );
        }
        return trajectory;
    }

    static std::vector<float> next_planner_frame(PlannerState& state) {
        if (state.trajectory.empty()) {
            return state.last_context_frame;
        }
        if (state.trajectory.size() < 2) {
            return state.trajectory.front();
        }

        const std::size_t index = std::min(state.traj_index, state.trajectory.size() - 2);
        const auto& frame_a = state.trajectory[index];
        const auto& frame_b = state.trajectory[std::min(index + 1, state.trajectory.size() - 1)];
        std::vector<float> interpolated(frame_a.size(), 0.0f);
        for (std::size_t i = 0; i < frame_a.size(); ++i) {
            interpolated[i] = frame_a[i] + state.interp_phase * (frame_b[i] - frame_a[i]);
        }

        state.interp_phase += kPlannerInterpStep;
        while (state.interp_phase >= 1.0f && state.traj_index < state.trajectory.size() - 2) {
            state.interp_phase -= 1.0f;
            state.traj_index += 1;
        }
        return interpolated;
    }

    static std::array<double, 4> planner_frame_root_quaternion(const std::vector<float>& frame) {
        return quat_unit_d({
            static_cast<double>(frame[3]),
            static_cast<double>(frame[4]),
            static_cast<double>(frame[5]),
            static_cast<double>(frame[6]),
        });
    }

    static std::vector<float> interpolate_planner_qpos(
        const std::vector<float>& frame_a,
        const std::vector<float>& frame_b,
        float alpha
    ) {
        std::vector<float> frame(frame_a.size(), 0.0f);
        for (std::size_t index = 0; index < frame.size(); ++index) {
            frame[index] = frame_a[index] + alpha * (frame_b[index] - frame_a[index]);
        }

        const auto quat = quat_slerp_d(
            planner_frame_root_quaternion(frame_a),
            planner_frame_root_quaternion(frame_b),
            static_cast<double>(alpha)
        );
        frame[3] = static_cast<float>(quat[0]);
        frame[4] = static_cast<float>(quat[1]);
        frame[5] = static_cast<float>(quat[2]);
        frame[6] = static_cast<float>(quat[3]);
        return frame;
    }

    static std::vector<float> sample_motion_qpos_50hz(
        const std::vector<std::vector<float>>& motion_qpos,
        float frame_idx
    ) {
        if (motion_qpos.empty()) {
            return {};
        }
        if (motion_qpos.size() == 1) {
            return motion_qpos.front();
        }

        const float clamped = std::clamp(
            frame_idx,
            0.0f,
            static_cast<float>(motion_qpos.size() - 1)
        );
        const auto frame_a_idx = static_cast<std::size_t>(std::floor(clamped));
        const auto frame_b_idx = std::min(frame_a_idx + 1, motion_qpos.size() - 1);
        const float alpha = clamped - static_cast<float>(frame_a_idx);
        return interpolate_planner_qpos(
            motion_qpos[frame_a_idx],
            motion_qpos[frame_b_idx],
            alpha
        );
    }

    static std::vector<std::vector<float>> resample_planner_trajectory_to_50hz(
        const std::vector<std::vector<float>>& trajectory
    ) {
        if (trajectory.empty()) {
            return {};
        }

        const float motion_seconds = static_cast<float>(trajectory.size()) / 30.0f;
        const auto frame_count = std::max(
            static_cast<std::size_t>(std::floor(motion_seconds * 50.0f)),
            static_cast<std::size_t>(1)
        );
        std::vector<std::vector<float>> motion_qpos;
        motion_qpos.reserve(frame_count);
        for (std::size_t frame_50hz = 0; frame_50hz < frame_count; ++frame_50hz) {
            const float frame_30hz = static_cast<float>(frame_50hz) * 30.0f / 50.0f;
            const auto frame_a_idx = static_cast<std::size_t>(std::floor(frame_30hz));
            const auto frame_b_idx = std::min(frame_a_idx + 1, trajectory.size() - 1);
            const float alpha = frame_30hz - static_cast<float>(frame_a_idx);
            motion_qpos.push_back(interpolate_planner_qpos(
                trajectory[frame_a_idx],
                trajectory[frame_b_idx],
                alpha
            ));
        }
        return motion_qpos;
    }

    static std::deque<std::vector<float>> rebuild_planner_context_from_motion(
        const std::vector<std::vector<float>>& motion_qpos_50hz,
        std::size_t current_motion_frame
    ) {
        std::deque<std::vector<float>> context;
        if (motion_qpos_50hz.empty()) {
            return context;
        }

        const float gen_time = static_cast<float>(current_motion_frame + 2) * kControlDtSeconds;
        for (std::size_t frame_idx = 0; frame_idx < kPlannerContextLen; ++frame_idx) {
            const float sample_time = gen_time + static_cast<float>(frame_idx) / 30.0f;
            context.push_back(sample_motion_qpos_50hz(
                motion_qpos_50hz,
                sample_time * 50.0f
            ));
        }
        return context;
    }

    static std::vector<float> planner_joint_positions_isaaclab(const std::vector<float>& frame) {
        std::vector<float> positions(G1_NUM_MOTOR, 0.0f);
        for (std::size_t mujoco_idx = 0; mujoco_idx < G1_NUM_MOTOR; ++mujoco_idx) {
            const auto isaaclab_idx = static_cast<std::size_t>(isaaclab_to_mujoco[mujoco_idx]);
            positions[isaaclab_idx] = frame[kPlannerJointOffset + mujoco_idx];
        }
        return positions;
    }

    static std::vector<std::vector<float>> compute_motion_joint_velocities_isaaclab(
        const std::vector<std::vector<float>>& motion_qpos
    ) {
        if (motion_qpos.empty()) {
            return {};
        }

        std::vector<std::vector<float>> positions;
        positions.reserve(motion_qpos.size());
        for (const auto& frame : motion_qpos) {
            positions.push_back(planner_joint_positions_isaaclab(frame));
        }

        std::vector<std::vector<float>> velocities(
            positions.size(),
            std::vector<float>(G1_NUM_MOTOR, 0.0f)
        );
        for (std::size_t frame_idx = 0; frame_idx + 1 < positions.size(); ++frame_idx) {
            for (std::size_t joint_idx = 0; joint_idx < G1_NUM_MOTOR; ++joint_idx) {
                velocities[frame_idx][joint_idx] =
                    (positions[frame_idx + 1][joint_idx] - positions[frame_idx][joint_idx]) * 50.0f;
            }
        }
        if (positions.size() > 1) {
            velocities.back() = velocities[velocities.size() - 2];
        }
        return velocities;
    }

    static std::vector<std::vector<float>> blend_planner_motion(
        const std::vector<std::vector<float>>& existing_motion_qpos,
        std::size_t current_motion_frame,
        std::size_t request_motion_frame,
        const std::vector<std::vector<float>>& new_motion_qpos
    ) {
        if (existing_motion_qpos.empty()) {
            return new_motion_qpos;
        }

        constexpr std::size_t kPlannerLookAheadSteps = 2;
        constexpr std::size_t kPlannerBlendFrames = 8;

        const auto gen_frame = request_motion_frame + kPlannerLookAheadSteps;
        const auto lead_frames =
            gen_frame > current_motion_frame ? gen_frame - current_motion_frame : 0;
        const auto new_anim_length = lead_frames + new_motion_qpos.size();
        const auto blend_start_frame = lead_frames;

        std::vector<std::vector<float>> blended;
        blended.reserve(new_anim_length);
        for (std::size_t frame_idx = 0; frame_idx < new_anim_length; ++frame_idx) {
            auto old_frame_idx = frame_idx + current_motion_frame;
            if (old_frame_idx >= existing_motion_qpos.size()) {
                old_frame_idx = existing_motion_qpos.size() - 1;
            }

            std::size_t new_frame_idx = 0;
            if (frame_idx + current_motion_frame >= gen_frame) {
                new_frame_idx = frame_idx + current_motion_frame - gen_frame;
            }
            if (new_frame_idx >= new_motion_qpos.size()) {
                new_frame_idx = new_motion_qpos.size() - 1;
            }

            const auto weight_new = std::clamp(
                (static_cast<float>(frame_idx) - static_cast<float>(blend_start_frame)) /
                    static_cast<float>(kPlannerBlendFrames),
                0.0f,
                1.0f
            );
            if (weight_new <= std::numeric_limits<float>::epsilon()) {
                blended.push_back(existing_motion_qpos[old_frame_idx]);
            } else if (std::abs(1.0f - weight_new) <= std::numeric_limits<float>::epsilon()) {
                blended.push_back(new_motion_qpos[new_frame_idx]);
            } else {
                blended.push_back(interpolate_planner_qpos(
                    existing_motion_qpos[old_frame_idx],
                    new_motion_qpos[new_frame_idx],
                    weight_new
                ));
            }
        }

        return blended;
    }

    static std::vector<float> build_velocity_encoder_obs_dict(
        const std::vector<std::vector<float>>& motion_qpos_50hz,
        const std::vector<std::vector<float>>& motion_joint_velocities_isaaclab,
        std::size_t current_motion_frame,
        const std::array<double, 4>& base_quat_wxyz,
        const std::array<double, 4>& init_base_quat_wxyz,
        const std::array<double, 4>& init_ref_root_quat_wxyz
    ) {
        if (motion_qpos_50hz.empty()) {
            throw std::runtime_error("planner motion buffer is empty");
        }

        const auto apply_delta_heading = quat_mul_d(
            calc_heading_quat_d(init_base_quat_wxyz),
            calc_heading_quat_inv_d(init_ref_root_quat_wxyz)
        );
        const auto base_quat = quat_unit_d(base_quat_wxyz);

        std::vector<float> buf(kEncoderObsDictDim, 0.0f);
        buf[kEncoderModeOffset] = 0.0f;

        for (std::size_t frame_idx = 0; frame_idx < kReferenceFutureFrames; ++frame_idx) {
            const auto target_frame = std::min(
                current_motion_frame + frame_idx * kReferenceFrameStep,
                motion_qpos_50hz.size() - 1
            );
            const auto& motion_frame = motion_qpos_50hz[target_frame];
            const auto isaaclab_positions = planner_joint_positions_isaaclab(motion_frame);

            const auto pos_offset =
                kEncoderMotionJointPositionsOffset + frame_idx * isaaclab_positions.size();
            std::copy(
                isaaclab_positions.begin(),
                isaaclab_positions.end(),
                buf.begin() + static_cast<std::ptrdiff_t>(pos_offset)
            );

            if (target_frame < motion_joint_velocities_isaaclab.size()) {
                const auto& joint_velocities = motion_joint_velocities_isaaclab[target_frame];
                const auto vel_offset =
                    kEncoderMotionJointVelocitiesOffset + frame_idx * joint_velocities.size();
                std::copy(
                    joint_velocities.begin(),
                    joint_velocities.end(),
                    buf.begin() + static_cast<std::ptrdiff_t>(vel_offset)
                );
            }

            const auto ref_root_quat =
                quat_mul_d(apply_delta_heading, planner_frame_root_quaternion(motion_frame));
            const auto base_to_ref = quat_mul_d(quat_conjugate_d(base_quat), ref_root_quat);
            const auto rotation_matrix = quat_to_rotation_matrix_d(base_to_ref);
            const auto orn_offset = kEncoderMotionAnchorOrientationOffset + frame_idx * 6;
            buf[orn_offset] = static_cast<float>(rotation_matrix[0][0]);
            buf[orn_offset + 1] = static_cast<float>(rotation_matrix[0][1]);
            buf[orn_offset + 2] = static_cast<float>(rotation_matrix[1][0]);
            buf[orn_offset + 3] = static_cast<float>(rotation_matrix[1][1]);
            buf[orn_offset + 4] = static_cast<float>(rotation_matrix[2][0]);
            buf[orn_offset + 5] = static_cast<float>(rotation_matrix[2][1]);
        }

        return buf;
    }

    static Observation motion_observation_from_planner_frame(
        const std::vector<std::vector<float>>& motion_qpos_50hz,
        const std::vector<std::vector<float>>& motion_joint_velocities_isaaclab,
        std::size_t frame_idx
    ) {
        if (motion_qpos_50hz.empty()) {
            throw std::runtime_error("planner motion buffer is empty");
        }

        const auto clamped_index = std::min(frame_idx, motion_qpos_50hz.size() - 1);
        const auto& motion_frame = motion_qpos_50hz[clamped_index];
        Observation obs;
        obs.joint_positions.assign(
            motion_frame.begin() + static_cast<std::ptrdiff_t>(kPlannerJointOffset),
            motion_frame.end()
        );
        obs.joint_velocities =
            clamped_index < motion_joint_velocities_isaaclab.size()
                ? isaaclab_to_mujoco_values(motion_joint_velocities_isaaclab[clamped_index])
                : std::vector<float>(G1_NUM_MOTOR, 0.0f);
        obs.base_quat_wxyz = planner_frame_root_quaternion(motion_frame);
        obs.gravity_vector = double_to_float(GetGravityOrientation_d(obs.base_quat_wxyz));
        obs.angular_velocity =
            angular_velocity_from_motion_quaternions(motion_qpos_50hz, clamped_index);
        return obs;
    }

    static std::array<float, 3> angular_velocity_from_motion_quaternions(
        const std::vector<std::vector<float>>& motion_qpos_50hz,
        std::size_t frame_idx
    ) {
        if (frame_idx == 0 || motion_qpos_50hz.empty()) {
            return {0.0f, 0.0f, 0.0f};
        }

        const auto prev_quat = planner_frame_root_quaternion(motion_qpos_50hz[frame_idx - 1]);
        const auto curr_quat = planner_frame_root_quaternion(motion_qpos_50hz[frame_idx]);
        auto delta = quat_mul_d(quat_conjugate_d(prev_quat), curr_quat);
        if (delta[0] < 0.0) {
            delta = {-delta[0], -delta[1], -delta[2], -delta[3]};
        }
        delta = quat_unit_d(delta);

        const auto sin_half = std::sqrt(
            delta[1] * delta[1] + delta[2] * delta[2] + delta[3] * delta[3]
        );
        if (sin_half <= 1e-6) {
            return {0.0f, 0.0f, 0.0f};
        }

        const auto [angle, axis] = quat_to_angle_axis(delta);
        if (!std::isfinite(angle)) {
            return {0.0f, 0.0f, 0.0f};
        }

        return {
            static_cast<float>(axis[0] * angle / kControlDtSeconds),
            static_cast<float>(axis[1] * angle / kControlDtSeconds),
            static_cast<float>(axis[2] * angle / kControlDtSeconds),
        };
    }

    std::vector<float> build_encoder_obs_dict() const {
        std::vector<float> buf(kEncoderObsDictDim, 0.0f);

        const auto pose = mujoco_to_isaaclab_positions(default_pose());
        const std::size_t pos_offset = 4;
        const std::size_t vel_offset = 294;
        const std::size_t orn_offset = 601;

        for (std::size_t frame = 0; frame < kDecoderHistoryLen; ++frame) {
            const std::size_t pose_index = pos_offset + frame * G1_NUM_MOTOR;
            const std::size_t velocity_index = vel_offset + frame * G1_NUM_MOTOR;
            std::copy(pose.begin(), pose.end(), buf.begin() + static_cast<std::ptrdiff_t>(pose_index));
            std::fill_n(buf.begin() + static_cast<std::ptrdiff_t>(velocity_index), G1_NUM_MOTOR, 0.0f);
        }

        for (std::size_t frame = 0; frame < kDecoderHistoryLen; ++frame) {
            const std::size_t orientation_index = orn_offset + frame * 6;
            buf[orientation_index] = 1.0f;
            buf[orientation_index + 4] = 1.0f;
        }

        return buf;
    }

    std::vector<float> build_decoder_obs_dict(const std::vector<float>& tokens) const {
        std::vector<float> buf;
        buf.reserve(kDecoderObsDictDim);
        buf.insert(buf.end(), tokens.begin(), tokens.end());

        const std::size_t history_skip = tracking_state_.gravity.size() > kDecoderHistoryLen
            ? tracking_state_.gravity.size() - kDecoderHistoryLen
            : 0;

        append_history_vectors(buf, tracking_state_.angular_velocity, history_skip);
        append_history_vectors(buf, tracking_state_.joint_positions, history_skip);
        append_history_vectors(buf, tracking_state_.joint_velocities, history_skip);
        append_history_vectors(buf, tracking_state_.last_actions, history_skip);
        append_history_vectors(buf, tracking_state_.gravity, history_skip);

        return buf;
    }

    template <typename Container>
    static void append_history_vectors(
        std::vector<float>& destination,
        const std::deque<Container>& history,
        std::size_t skip
    ) {
        for (std::size_t index = skip; index < history.size(); ++index) {
            destination.insert(destination.end(), history[index].begin(), history[index].end());
        }
    }

    std::vector<float> run_single_f32(
        Ort::Session& session,
        const char* input_name,
        const char* output_name,
        const std::vector<float>& input,
        std::size_t expected_input_dim
    ) {
        if (input.size() != expected_input_dim) {
            throw std::runtime_error(
                "input dimension mismatch for " + std::string(output_name) +
                ": expected " + std::to_string(expected_input_dim) +
                ", got " + std::to_string(input.size())
            );
        }

        const std::array<std::int64_t, 2> shape = {1, static_cast<std::int64_t>(expected_input_dim)};
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(input.data()),
            input.size(),
            shape.data(),
            shape.size()
        );

        const char* input_names[] = {input_name};
        const char* output_names[] = {output_name};
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        const Ort::Value& output = outputs.front();
        const auto info = output.GetTensorTypeAndShapeInfo();
        const std::size_t count = info.GetElementCount();
        const float* data = output.GetTensorData<float>();
        return std::vector<float>(data, data + static_cast<std::ptrdiff_t>(count));
    }

    static void write_vector_json(const fs::path& path, const std::vector<float>& values) {
        fs::create_directories(path.parent_path());
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to open tensor dump file: " + path.string());
        }

        out << "[\n";
        for (std::size_t index = 0; index < values.size(); ++index) {
            out << "  " << values[index];
            if (index + 1 < values.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "]\n";
    }

    static void write_matrix_json(
        const fs::path& path,
        const std::vector<std::vector<float>>& rows
    ) {
        fs::create_directories(path.parent_path());
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to open tensor dump file: " + path.string());
        }

        out << "[\n";
        for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
            out << "  [\n";
            for (std::size_t value_index = 0; value_index < rows[row_index].size(); ++value_index) {
                out << "    " << rows[row_index][value_index];
                if (value_index + 1 < rows[row_index].size()) {
                    out << ",";
                }
                out << "\n";
            }
            out << "  ]";
            if (row_index + 1 < rows.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "]\n";
    }

    static std::vector<std::vector<float>> deque_to_rows(
        const std::deque<std::vector<float>>& rows
    ) {
        return {rows.begin(), rows.end()};
    }

    template <std::size_t N>
    static std::vector<std::vector<float>> deque_to_rows(
        const std::deque<std::array<float, N>>& rows
    ) {
        std::vector<std::vector<float>> out;
        out.reserve(rows.size());
        for (const auto& row : rows) {
            out.emplace_back(row.begin(), row.end());
        }
        return out;
    }

    void maybe_dump_tracking_tensors(
        const std::vector<float>& encoder_obs,
        const std::vector<float>& tokens,
        const std::vector<float>& decoder_obs,
        const std::vector<float>& raw_actions
    ) {
        if (!dump_dir_.has_value() || dumped_tracking_tensors_) {
            return;
        }

        write_vector_json(*dump_dir_ / "tracking_encoder_obs.json", encoder_obs);
        write_vector_json(*dump_dir_ / "tracking_tokens.json", tokens);
        write_vector_json(*dump_dir_ / "tracking_decoder_obs.json", decoder_obs);
        write_vector_json(*dump_dir_ / "tracking_raw_actions.json", raw_actions);
        dumped_tracking_tensors_ = true;
    }
};

enum class CaseKind {
    ColdStartTick,
    WarmSteadyStateTick,
    ReplanTick,
    FirstLiveReplanDump,
    LaterMotionDump,
    StandingPlaceholderTick,
    EndToEndLoop,
};

CaseKind parse_case_kind(const std::string& case_id) {
    if (case_id == "gear_sonic_velocity/cold_start_tick") {
        return CaseKind::ColdStartTick;
    }
    if (case_id == "gear_sonic_velocity/warm_steady_state_tick") {
        return CaseKind::WarmSteadyStateTick;
    }
    if (case_id == "gear_sonic_velocity/replan_tick") {
        return CaseKind::ReplanTick;
    }
    if (case_id == "gear_sonic_velocity/first_live_replan_dump") {
        return CaseKind::FirstLiveReplanDump;
    }
    if (case_id == "gear_sonic_velocity/later_motion_dump") {
        return CaseKind::LaterMotionDump;
    }
    if (case_id == "gear_sonic_tracking/standing_placeholder_tick") {
        return CaseKind::StandingPlaceholderTick;
    }
    if (case_id == "gear_sonic/end_to_end_cli_loop") {
        return CaseKind::EndToEndLoop;
    }
    throw std::runtime_error("unsupported GEAR-Sonic case_id: " + case_id);
}

template <typename SetupFn, typename MeasureFn>
std::vector<std::uint64_t> run_microbench(
    GearSonicOfficialHarness& harness,
    int samples,
    SetupFn&& setup,
    MeasureFn&& measure
) {
    std::vector<std::uint64_t> timings;
    timings.reserve(static_cast<std::size_t>(samples));
    for (int sample = 0; sample < samples; ++sample) {
        harness.reset();
        setup();
        const auto start = std::chrono::steady_clock::now();
        const auto result = measure();
        const auto end = std::chrono::steady_clock::now();
        g_sink = g_sink + (result.empty() ? 0.0f : result.front());
        timings.push_back(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
            )
        );
    }
    return timings;
}

template <typename Fn>
std::pair<std::vector<std::uint64_t>, double> run_end_to_end_loop(
    GearSonicOfficialHarness& harness,
    int ticks,
    int control_frequency_hz,
    Fn&& fn
) {
    const auto period_ns = static_cast<std::int64_t>(std::llround(1'000'000'000.0 / static_cast<double>(control_frequency_hz)));
    std::vector<std::uint64_t> timings;
    timings.reserve(static_cast<std::size_t>(ticks));

    harness.reset();
    const auto wall_start = std::chrono::steady_clock::now();
    for (int tick = 0; tick < ticks; ++tick) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = fn();
        const auto end = std::chrono::steady_clock::now();
        g_sink = g_sink + (result.empty() ? 0.0f : result.front());

        const auto elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        timings.push_back(static_cast<std::uint64_t>(elapsed_ns));

        const auto remaining_ns = period_ns - elapsed_ns;
        if (remaining_ns > 0) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(remaining_ns));
        }
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const auto elapsed_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    const double achieved_hz = elapsed_seconds > 0.0 ? static_cast<double>(ticks) / elapsed_seconds : 0.0;
    return {timings, achieved_hz};
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto require_value = [&](const std::string& flag) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error(flag + " requires a value");
            }
            return argv[++index];
        };

        if (arg == "--case-id") {
            options.case_id = require_value(arg);
        } else if (arg == "--model-dir") {
            options.model_dir = require_value(arg);
        } else if (arg == "--output") {
            options.output = require_value(arg);
        } else if (arg == "--dump-dir") {
            options.dump_dir = fs::path(require_value(arg));
        } else if (arg == "--samples") {
            options.samples = std::stoi(require_value(arg));
        } else if (arg == "--ticks") {
            options.ticks = std::stoi(require_value(arg));
        } else if (arg == "--control-frequency-hz") {
            options.control_frequency_hz = std::stoi(require_value(arg));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.case_id.empty()) {
        throw std::runtime_error("--case-id is required");
    }
    if (options.model_dir.empty()) {
        throw std::runtime_error("--model-dir is required");
    }
    if (options.output.empty()) {
        throw std::runtime_error("--output is required");
    }
    if (options.samples <= 0) {
        throw std::runtime_error("--samples must be positive");
    }
    if (options.ticks <= 0) {
        throw std::runtime_error("--ticks must be positive");
    }
    if (options.control_frequency_hz <= 0) {
        throw std::runtime_error("--control-frequency-hz must be positive");
    }
    return options;
}

void write_json(
    const Options& options,
    const std::vector<std::uint64_t>& samples_ns,
    const std::optional<double>& hz
) {
    fs::create_directories(options.output.parent_path());
    std::ofstream out(options.output);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + options.output.string());
    }

    out << "{\n";
    out << "  \"case_id\": \"" << json_escape(options.case_id) << "\",\n";
    out << "  \"samples_ns\": [";
    for (std::size_t index = 0; index < samples_ns.size(); ++index) {
        if (index == 0) {
            out << "\n    ";
        } else {
            out << ",\n    ";
        }
        out << samples_ns[index];
    }
    if (!samples_ns.empty()) {
        out << '\n';
    }
    out << "  ],\n";
    out << "  \"hz\": ";
    if (hz.has_value()) {
        out << *hz;
    } else {
        out << "null";
    }
    out << ",\n";
    out << "  \"notes\": \"Measured via official GEAR-Sonic C++ ONNX Runtime harness on the published planner and encoder/decoder contracts.\"\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const CaseKind case_kind = parse_case_kind(options.case_id);
        GearSonicOfficialHarness harness(options.model_dir, options.dump_dir);

        std::vector<std::uint64_t> samples_ns;
        std::optional<double> hz;

        switch (case_kind) {
            case CaseKind::ColdStartTick:
                samples_ns = run_microbench(
                    harness,
                    options.samples,
                    []() {},
                    [&]() { return harness.velocity_tick(); }
                );
                break;
            case CaseKind::WarmSteadyStateTick:
                samples_ns = run_microbench(
                    harness,
                    options.samples,
                    [&]() { harness.velocity_tick(); },
                    [&]() { return harness.velocity_tick(); }
                );
                break;
            case CaseKind::ReplanTick:
                samples_ns = run_microbench(
                    harness,
                    options.samples,
                    [&]() {
                        for (std::size_t tick = 0; tick < kReplanIntervalTicks; ++tick) {
                            harness.velocity_tick();
                        }
                    },
                    [&]() { return harness.velocity_tick(); }
                );
                break;
            case CaseKind::FirstLiveReplanDump: {
                const auto start = std::chrono::steady_clock::now();
                const auto result = harness.velocity_first_live_replan_dump();
                const auto end = std::chrono::steady_clock::now();
                g_sink = g_sink + (result.empty() ? 0.0f : result.front());
                samples_ns.push_back(
                    static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                    )
                );
                break;
            }
            case CaseKind::LaterMotionDump: {
                const auto start = std::chrono::steady_clock::now();
                const auto result = harness.velocity_later_motion_dump();
                const auto end = std::chrono::steady_clock::now();
                g_sink = g_sink + (result.empty() ? 0.0f : result.front());
                samples_ns.push_back(
                    static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                    )
                );
                break;
            }
            case CaseKind::StandingPlaceholderTick:
                samples_ns = run_microbench(
                    harness,
                    options.samples,
                    []() {},
                    [&]() { return harness.tracking_tick(); }
                );
                break;
            case CaseKind::EndToEndLoop: {
                auto [timings, achieved_hz] = run_end_to_end_loop(
                    harness,
                    options.ticks,
                    options.control_frequency_hz,
                    [&]() { return harness.velocity_tick(); }
                );
                samples_ns = std::move(timings);
                hz = achieved_hz;
                break;
            }
        }

        write_json(options, samples_ns, hz);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
