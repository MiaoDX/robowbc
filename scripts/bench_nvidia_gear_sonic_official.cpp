#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "policy_parameters.hpp"
#include "robot_parameters.hpp"

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kPlannerQposDim = 36;
constexpr std::size_t kPlannerJointOffset = 7;
constexpr std::size_t kPlannerContextLen = 4;
constexpr std::size_t kReplanIntervalTicks = 5;
constexpr std::size_t kAllowedPredNumTokens = 11;
constexpr float kDefaultHeightMeters = 0.74f;
constexpr std::int64_t kDefaultModeWalk = 2;
constexpr float kPlannerInterpStep = 30.0f / 50.0f;
constexpr std::size_t kEncoderDim = 64;
constexpr std::size_t kEncoderObsDictDim = 1762;
constexpr std::size_t kDecoderObsDictDim = 994;
constexpr std::size_t kDecoderHistoryLen = 10;

volatile float g_sink = 0.0f;

struct Options {
    std::string case_id;
    fs::path model_dir;
    fs::path output;
    int samples = 100;
    int ticks = 200;
    int control_frequency_hz = 50;
};

struct Observation {
    std::vector<float> joint_positions;
    std::vector<float> joint_velocities;
    std::array<float, 3> gravity_vector{};
    std::array<float, 3> angular_velocity{};
};

struct Twist {
    std::array<float, 3> linear{};
    std::array<float, 3> angular{};
};

std::vector<float> default_pose() {
    std::vector<float> pose;
    pose.reserve(G1_NUM_MOTOR);
    for (double angle : default_angles) {
        pose.push_back(static_cast<float>(angle));
    }
    return pose;
}

Observation zero_observation() {
    Observation obs;
    obs.joint_positions.assign(G1_NUM_MOTOR, 0.0f);
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

        const auto pose = default_pose();
        for (std::size_t index = 0; index < kDecoderHistoryLen; ++index) {
            gravity.push_back({0.0f, 0.0f, -1.0f});
            angular_velocity.push_back({0.0f, 0.0f, 0.0f});
            joint_positions.push_back(pose);
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
        joint_positions.push_back(obs.joint_positions);
        joint_velocities.push_back(obs.joint_velocities);
        last_actions.push_back(actions);
    }
};

class GearSonicOfficialHarness {
public:
    explicit GearSonicOfficialHarness(const fs::path& model_dir)
        : env_(ORT_LOGGING_LEVEL_WARNING, "gear_sonic_official"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          encoder_(make_session(env_, model_dir / "model_encoder.onnx")),
          decoder_(make_session(env_, model_dir / "model_decoder.onnx")),
          planner_(make_session(env_, model_dir / "planner_sonic.onnx")),
          planner_state_(),
          tracking_state_(),
          velocity_obs_(zero_observation()),
          tracking_obs_(zero_observation())
    {
        validate_contracts();
    }

    void reset() {
        planner_state_.reset();
        tracking_state_.reset();
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

        const std::vector<float> current_actions(G1_NUM_MOTOR, 0.0f);
        const auto decoder_obs = build_decoder_obs_dict(tokens, current_actions);
        const auto raw_actions = run_single_f32(decoder_, "obs_dict", "action", decoder_obs, kDecoderObsDictDim);
        if (raw_actions.size() != G1_NUM_MOTOR) {
            throw std::runtime_error(
                "decoder output dimension mismatch: expected " + std::to_string(G1_NUM_MOTOR) +
                ", got " + std::to_string(raw_actions.size())
            );
        }

        std::vector<float> positions(G1_NUM_MOTOR, 0.0f);
        for (int mujoco_index = 0; mujoco_index < G1_NUM_MOTOR; ++mujoco_index) {
            const int isaaclab_index = isaaclab_to_mujoco[static_cast<std::size_t>(mujoco_index)];
            const float action = raw_actions[static_cast<std::size_t>(isaaclab_index)];
            const float scaled = action * static_cast<float>(g1_action_scale[static_cast<std::size_t>(mujoco_index)]);
            positions[static_cast<std::size_t>(mujoco_index)] =
                static_cast<float>(default_angles[static_cast<std::size_t>(mujoco_index)]) + scaled;
        }

        tracking_state_.push(tracking_obs_, raw_actions);
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
        const bool zero_quat = std::all_of(
            frame.begin() + 3,
            frame.begin() + 7,
            [](float value) { return std::abs(value) <= std::numeric_limits<float>::epsilon(); }
        );
        if (zero_quat) {
            frame[3] = 1.0f;
        }
        std::copy(obs.joint_positions.begin(), obs.joint_positions.end(), frame.begin() + static_cast<std::ptrdiff_t>(kPlannerJointOffset));
        return frame;
    }

    Twist velocity_command() const {
        Twist twist;
        twist.linear = {0.3f, 0.0f, 0.0f};
        twist.angular = {0.0f, 0.0f, 0.0f};
        return twist;
    }

    std::vector<std::vector<float>> run_planner(
        const std::deque<std::vector<float>>& context,
        const Twist& twist
    ) {
        std::vector<float> context_data;
        context_data.reserve(context.size() * kPlannerQposDim);
        for (const auto& frame : context) {
            context_data.insert(context_data.end(), frame.begin(), frame.end());
        }

        const float cmd_norm =
            std::sqrt(twist.linear[0] * twist.linear[0] + twist.linear[1] * twist.linear[1]);
        const std::array<float, 3> movement_direction =
            cmd_norm > 1e-6f ? std::array<float, 3>{twist.linear[0] / cmd_norm, twist.linear[1] / cmd_norm, 0.0f}
                             : std::array<float, 3>{1.0f, 0.0f, 0.0f};
        const float yaw = twist.angular[2];
        const std::array<float, 3> facing_direction = {std::cos(yaw), std::sin(yaw), 0.0f};
        const std::array<float, 1> target_vel = {cmd_norm};
        const std::array<std::int64_t, 1> mode = {kDefaultModeWalk};
        const std::array<float, 1> height = {kDefaultHeightMeters};
        const std::array<std::int64_t, 1> random_seed = {0};
        const std::array<std::int64_t, 1> has_specific_target = {0};
        const std::vector<float> specific_target_positions(12, 0.0f);
        const std::vector<float> specific_target_headings(4, 0.0f);
        const std::vector<std::int64_t> allowed_pred_num_tokens(kAllowedPredNumTokens, 1);

        const std::array<std::int64_t, 3> context_shape = {1, static_cast<std::int64_t>(kPlannerContextLen), static_cast<std::int64_t>(kPlannerQposDim)};
        const std::array<std::int64_t, 2> vec3_shape = {1, 3};
        const std::array<std::int64_t, 1> scalar_shape = {1};
        const std::array<std::int64_t, 2> has_target_shape = {1, 1};
        const std::array<std::int64_t, 3> specific_positions_shape = {1, 4, 3};
        const std::array<std::int64_t, 2> specific_headings_shape = {1, 4};
        const std::array<std::int64_t, 2> allowed_tokens_shape = {1, static_cast<std::int64_t>(kAllowedPredNumTokens)};

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
            const_cast<float*>(movement_direction.data()),
            movement_direction.size(),
            vec3_shape.data(),
            vec3_shape.size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(facing_direction.data()),
            facing_direction.size(),
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
            predicted_frames = static_cast<std::int64_t>(num_pred_frames.GetTensorData<std::int32_t>()[0]);
        } else if (frames_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            predicted_frames = num_pred_frames.GetTensorData<std::int64_t>()[0];
        } else {
            throw std::runtime_error("planner num_pred_frames output must be int32 or int64");
        }
        predicted_frames = std::max<std::int64_t>(predicted_frames, 1);
        const std::size_t frame_count = std::min<std::size_t>(static_cast<std::size_t>(predicted_frames), available_frames);

        if (!(qpos_shape.size() == 3 || qpos_shape.size() == 2)) {
            throw std::runtime_error("unexpected planner mujoco_qpos rank");
        }

        std::vector<std::vector<float>> trajectory;
        trajectory.reserve(frame_count);
        for (std::size_t frame = 0; frame < frame_count; ++frame) {
            const float* begin = qpos_data + frame * kPlannerQposDim;
            trajectory.emplace_back(begin, begin + static_cast<std::ptrdiff_t>(kPlannerQposDim));
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

    std::vector<float> build_encoder_obs_dict() const {
        std::vector<float> buf(kEncoderObsDictDim, 0.0f);

        const auto pose = default_pose();
        const std::size_t pos_offset = 4;
        const std::size_t vel_offset = 294;
        const std::size_t orn_offset = 584;

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

    std::vector<float> build_decoder_obs_dict(
        const std::vector<float>& tokens,
        const std::vector<float>& current_actions
    ) const {
        std::vector<float> buf;
        buf.reserve(kDecoderObsDictDim);
        buf.insert(buf.end(), tokens.begin(), tokens.end());

        const std::size_t history_skip = tracking_state_.gravity.size() > (kDecoderHistoryLen - 1)
            ? tracking_state_.gravity.size() - (kDecoderHistoryLen - 1)
            : 0;

        append_history_vectors(buf, tracking_state_.gravity, history_skip);
        buf.insert(buf.end(), tracking_obs_.gravity_vector.begin(), tracking_obs_.gravity_vector.end());

        append_history_vectors(buf, tracking_state_.angular_velocity, history_skip);
        buf.insert(buf.end(), tracking_obs_.angular_velocity.begin(), tracking_obs_.angular_velocity.end());

        append_history_vectors(buf, tracking_state_.joint_positions, history_skip);
        buf.insert(buf.end(), tracking_obs_.joint_positions.begin(), tracking_obs_.joint_positions.end());

        append_history_vectors(buf, tracking_state_.joint_velocities, history_skip);
        buf.insert(buf.end(), tracking_obs_.joint_velocities.begin(), tracking_obs_.joint_velocities.end());

        append_history_vectors(buf, tracking_state_.last_actions, history_skip);
        buf.insert(buf.end(), current_actions.begin(), current_actions.end());

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
};

enum class CaseKind {
    ColdStartTick,
    WarmSteadyStateTick,
    ReplanTick,
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
        GearSonicOfficialHarness harness(options.model_dir);

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
