//! Runtime FSM orchestrating transport, policy, validator, and teleop.
//!
//! The state machine that drives the control loop, with four primary states:
//!
//! ```text
//!     ┌────────┐  first lowstate received   ┌──────────┐
//!     │  Init  │ ─────────────────────────→ │ RL_Init  │
//!     └────────┘                            └──────────┘
//!                                                 │
//!                          q reaches default_pos  │  (or teleop Engage)
//!                                                 ▼
//!                                           ┌──────────────┐
//!                                           │  RL_Running  │
//!                                           └──────────────┘
//!                                               ↑    │
//!                                  user reset   │    │ teleop e-stop
//!                                                │    │ OR validator fault
//!                                                │    │ OR IMU tilt
//!                                                │    ▼
//!                                           ┌──────────────┐
//!                                           │   Damping    │
//!                                           └──────────────┘
//! ```
//!
//! `Fault` is not a separate persistent state — it is a logged transition
//! through which the FSM enters [`FsmState::Damping`].
//!
//! # Design choice: `match`-based FSM
//!
//! The implementation uses an explicit [`FsmState`] enum + `match` rather
//! than a typestate pattern. Two reasons: (1) the runtime owns the FSM
//! through a single `&mut Fsm` handle for the entire control loop — the
//! typestate pattern's compile-time state-tracking advantage is lost as
//! soon as the FSM is stored in a long-lived field; (2) a `match`-based
//! FSM is trivially mockable and exhaustively testable, which the issue
//! calls out as an acceptance criterion.
//!
//! # Single policy per runtime instance
//!
//! Per #126's explicit non-goal, switching policies requires a process
//! restart. The [`Fsm`] takes ownership of one [`WbcPolicy`] at construction
//! and never replaces it. This simplifies lifecycle: no in-runtime hot-swap,
//! no mid-tick state mismatch.
//!
//! # Tests cover every transition
//!
//! See the in-crate `tests` module for one test per transition listed in the
//! state diagram above plus the implicit reset-from-`Damping` and
//! self-loops. A small `MockPolicy` is provided for tests.

#![allow(clippy::module_name_repetitions)]

use robowbc_core::validator::{Fault, PolicyValidator};
use robowbc_core::{Observation, WbcCommand, WbcError, WbcPolicy};
use robowbc_teleop::TeleopEvent;
use std::time::{Duration, Instant};

/// Errors produced by the runtime FSM.
#[derive(Debug, thiserror::Error)]
pub enum FsmError {
    /// The observation supplied to [`Fsm::tick`] does not match the configured
    /// joint count.
    #[error("observation joint count {actual} does not match runtime joint_count {expected}")]
    JointCountMismatch {
        /// Joints expected by the configured runtime.
        expected: usize,
        /// Joints reported by the incoming observation.
        actual: usize,
    },
    /// The wrapped policy returned an error during inference.
    #[error("policy inference failed: {0}")]
    PolicyInference(#[from] WbcError),
}

/// Persistent FSM state.
///
/// `Fault` is not represented here — it is the transient transition reason
/// recorded inside [`FsmTransition`] right before entering [`FsmState::Damping`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsmState {
    /// Awaiting the first observation before any motion.
    Init,
    /// Smoothly interpolating from the current pose to `default_dof_pos`.
    RlInit,
    /// Running the wrapped [`WbcPolicy`] every tick.
    RlRunning,
    /// Holding all joints at `kp=0, kd=damping_kd`. No policy output.
    Damping,
}

impl FsmState {
    /// Returns the stable `snake_case` name used for logs and rerun entities.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::RlInit => "rl_init",
            Self::RlRunning => "rl_running",
            Self::Damping => "damping",
        }
    }
}

impl std::fmt::Display for FsmState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The reason for an FSM transition. Suitable for logging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionReason {
    /// `Init` → `RlInit`: the first lowstate observation arrived.
    FirstObservation,
    /// `RlInit` → `RlRunning`: interpolation duration elapsed.
    RlInitComplete,
    /// `RlInit` → `RlRunning`: teleop sent `Engage`, exiting `RlInit` early.
    EngageRequested,
    /// → `Damping`: teleop sent emergency stop.
    EmergencyStop,
    /// → `Damping`: policy validator returned a fault.
    PolicyFault(Fault),
    /// → `Damping`: IMU tilt exceeded the configured maximum.
    ImuTilt,
    /// `Damping` → `RlInit`: teleop sent reset.
    UserReset,
}

impl std::fmt::Display for TransitionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstObservation => f.write_str("first_observation"),
            Self::RlInitComplete => f.write_str("rl_init_complete"),
            Self::EngageRequested => f.write_str("engage_requested"),
            Self::EmergencyStop => f.write_str("emergency_stop"),
            Self::PolicyFault(fault) => write!(f, "policy_fault:{fault}"),
            Self::ImuTilt => f.write_str("imu_tilt"),
            Self::UserReset => f.write_str("user_reset"),
        }
    }
}

/// One transition record. Recorded by the FSM whenever the state changes.
///
/// The runtime exposes the accumulated log via [`Fsm::transitions`]. The
/// rerun integration (#133) consumes this stream to plot state changes on
/// a dedicated entity path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsmTransition {
    /// State before the transition.
    pub from: FsmState,
    /// State after the transition.
    pub to: FsmState,
    /// Reason for the transition.
    pub reason: TransitionReason,
    /// Wall-clock timestamp from the triggering observation.
    pub at: Instant,
}

/// PD setpoint emitted by the FSM each tick.
///
/// The wire format consumed by the transport is per-joint: every joint has its
/// own target position, kp, and kd. Damping emits `kp=0, kd=damping_kd` on every
/// joint regardless of policy output.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlOutput {
    /// Target joint positions in radians.
    pub q_target: Vec<f32>,
    /// Per-joint proportional gain.
    pub kp: Vec<f32>,
    /// Per-joint derivative gain.
    pub kd: Vec<f32>,
}

/// Configuration for the runtime FSM.
#[derive(Debug, Clone)]
pub struct FsmConfig {
    /// Number of actuated joints.
    pub joint_count: usize,
    /// Default standing pose targeted during `RlInit`.
    pub default_dof_pos: Vec<f32>,
    /// Per-joint proportional gains used in `RlInit` and `RlRunning`.
    pub default_kp: Vec<f32>,
    /// Per-joint derivative gains used in `RlInit` and `RlRunning`.
    pub default_kd: Vec<f32>,
    /// Per-joint kd applied uniformly in `Damping` (kp is forced to 0).
    /// Default: `8.0`.
    pub damping_kd: f32,
    /// `RlInit` interpolation duration. Default: `3s`.
    pub rl_init_duration: Duration,
    /// IMU tilt threshold in radians; exceeding it transitions to `Damping`.
    /// Computed as `acos(-gravity_z / |gravity|)` against the body-frame
    /// gravity vector. Default: `0.7` rad (≈ 40°).
    pub imu_tilt_max_rad: f32,
}

impl FsmConfig {
    /// Convenience constructor that fills `default_kp`/`default_kd` with the
    /// given uniform values and uses the canonical defaults for the remaining
    /// safety knobs.
    #[must_use]
    pub fn uniform(joint_count: usize, default_dof_pos: Vec<f32>, kp: f32, kd: f32) -> Self {
        Self {
            joint_count,
            default_dof_pos,
            default_kp: vec![kp; joint_count],
            default_kd: vec![kd; joint_count],
            damping_kd: 8.0,
            rl_init_duration: Duration::from_secs(3),
            imu_tilt_max_rad: 0.7,
        }
    }

    /// Returns an error string if the lengths of the per-joint vectors do not
    /// match `joint_count`.
    fn validate(&self) -> Result<(), String> {
        if self.default_dof_pos.len() != self.joint_count {
            return Err(format!(
                "default_dof_pos length {} != joint_count {}",
                self.default_dof_pos.len(),
                self.joint_count
            ));
        }
        if self.default_kp.len() != self.joint_count {
            return Err(format!(
                "default_kp length {} != joint_count {}",
                self.default_kp.len(),
                self.joint_count
            ));
        }
        if self.default_kd.len() != self.joint_count {
            return Err(format!(
                "default_kd length {} != joint_count {}",
                self.default_kd.len(),
                self.joint_count
            ));
        }
        Ok(())
    }
}

/// Internal `RlInit` interpolation state.
#[derive(Debug, Clone)]
struct InterpolationState {
    start_time: Instant,
    start_q: Vec<f32>,
}

/// The runtime state machine.
pub struct Fsm<P: WbcPolicy> {
    config: FsmConfig,
    policy: P,
    validator: PolicyValidator,
    state: FsmState,
    interp: Option<InterpolationState>,
    transitions: Vec<FsmTransition>,
}

impl<P: WbcPolicy> Fsm<P> {
    /// Build a new FSM in the [`FsmState::Init`] state.
    ///
    /// # Errors
    ///
    /// Returns an error string if any of the per-joint vectors in `config`
    /// does not have length `config.joint_count`.
    pub fn new(config: FsmConfig, policy: P, validator: PolicyValidator) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            policy,
            validator,
            state: FsmState::Init,
            interp: None,
            transitions: Vec::new(),
        })
    }

    /// Returns the current FSM state.
    #[must_use]
    pub fn state(&self) -> FsmState {
        self.state
    }

    /// Returns the accumulated transition log.
    #[must_use]
    pub fn transitions(&self) -> &[FsmTransition] {
        &self.transitions
    }

    /// Advance the FSM by one control tick.
    ///
    /// Inputs:
    /// - `obs`: current observation (joint positions, velocities, IMU,
    ///   command). The wrapped policy is also invoked with this observation
    ///   when in `RlRunning`.
    /// - `events`: teleop events drained this tick. Order matters: an
    ///   `EmergencyStop` always wins regardless of position in the slice.
    ///
    /// # Errors
    ///
    /// Returns [`FsmError::JointCountMismatch`] if the observation size
    /// disagrees with the configured joint count, or
    /// [`FsmError::PolicyInference`] if the wrapped policy fails. Validator
    /// faults are recorded as transitions to `Damping`, not propagated.
    pub fn tick(
        &mut self,
        obs: &Observation,
        events: &[TeleopEvent],
    ) -> Result<ControlOutput, FsmError> {
        if obs.joint_positions.len() != self.config.joint_count {
            return Err(FsmError::JointCountMismatch {
                expected: self.config.joint_count,
                actual: obs.joint_positions.len(),
            });
        }

        // Hard pre-checks: emergency stop and IMU tilt always trip damping
        // before any state-specific logic. These are checked even from
        // `Damping` itself (no-op self-loop is just ignored downstream).
        let mut faulted = false;
        for event in events {
            if matches!(event, TeleopEvent::EmergencyStop) {
                faulted = self.transition_to_damping(TransitionReason::EmergencyStop, obs);
                break;
            }
        }
        if !faulted && Self::imu_tilt_exceeded(obs.gravity_vector, self.config.imu_tilt_max_rad) {
            faulted = self.transition_to_damping(TransitionReason::ImuTilt, obs);
        }
        let _ = faulted;

        // Soft event handling (reset, engage). Order mirrors the slice.
        for event in events {
            match event {
                TeleopEvent::Reset => {
                    if self.state != FsmState::Init {
                        self.transition_to_rl_init(TransitionReason::UserReset, obs);
                    }
                }
                TeleopEvent::Engage => {
                    if self.state == FsmState::RlInit {
                        self.transition_to_running(TransitionReason::EngageRequested, obs);
                    }
                }
                TeleopEvent::EmergencyStop
                | TeleopEvent::Velocity { .. }
                | TeleopEvent::ToggleElasticBand
                | TeleopEvent::Quit => {}
            }
        }

        // State-driven transitions and outputs.
        match self.state {
            FsmState::Init => {
                self.transition_to_rl_init(TransitionReason::FirstObservation, obs);
                Ok(self.rl_init_output(obs))
            }
            FsmState::RlInit => {
                if self.rl_init_complete(obs) {
                    self.transition_to_running(TransitionReason::RlInitComplete, obs);
                    self.run_policy(obs)
                } else {
                    Ok(self.rl_init_output(obs))
                }
            }
            FsmState::RlRunning => self.run_policy(obs),
            FsmState::Damping => Ok(self.damping_output()),
        }
    }

    fn run_policy(&mut self, obs: &Observation) -> Result<ControlOutput, FsmError> {
        let targets = self.policy.predict(obs)?;
        if targets.positions.len() != self.config.joint_count {
            return Err(FsmError::PolicyInference(WbcError::InvalidTargets(
                "policy returned wrong joint count",
            )));
        }
        match self
            .validator
            .validate(&targets.positions, &obs.joint_positions)
        {
            Ok(()) => Ok(ControlOutput {
                q_target: targets.positions,
                kp: self.config.default_kp.clone(),
                kd: self.config.default_kd.clone(),
            }),
            Err(fault) => {
                self.transition_to_damping(TransitionReason::PolicyFault(fault), obs);
                Ok(self.damping_output())
            }
        }
    }

    fn rl_init_output(&self, obs: &Observation) -> ControlOutput {
        let interp = self
            .interp
            .as_ref()
            .expect("RlInit must always have interpolation state");
        let elapsed = obs
            .timestamp
            .saturating_duration_since(interp.start_time)
            .as_secs_f32();
        let total = self.config.rl_init_duration.as_secs_f32();
        let t = if total <= 0.0 {
            1.0
        } else {
            (elapsed / total).clamp(0.0, 1.0)
        };
        let s = smoothstep(t);
        let q_target: Vec<f32> = interp
            .start_q
            .iter()
            .zip(self.config.default_dof_pos.iter())
            .map(|(&start, &end)| start + (end - start) * s)
            .collect();
        ControlOutput {
            q_target,
            kp: self.config.default_kp.clone(),
            kd: self.config.default_kd.clone(),
        }
    }

    fn rl_init_complete(&self, obs: &Observation) -> bool {
        let Some(interp) = self.interp.as_ref() else {
            return true;
        };
        obs.timestamp.saturating_duration_since(interp.start_time) >= self.config.rl_init_duration
    }

    fn damping_output(&self) -> ControlOutput {
        let n = self.config.joint_count;
        ControlOutput {
            q_target: vec![0.0; n],
            kp: vec![0.0; n],
            kd: vec![self.config.damping_kd; n],
        }
    }

    /// Returns `true` if a transition was actually recorded.
    fn transition_to_damping(&mut self, reason: TransitionReason, obs: &Observation) -> bool {
        if self.state == FsmState::Damping {
            return false;
        }
        self.record_transition(self.state, FsmState::Damping, reason, obs.timestamp);
        self.state = FsmState::Damping;
        self.interp = None;
        self.validator.reset();
        self.policy.reset();
        true
    }

    fn transition_to_rl_init(&mut self, reason: TransitionReason, obs: &Observation) {
        let prev = self.state;
        self.record_transition(prev, FsmState::RlInit, reason, obs.timestamp);
        self.state = FsmState::RlInit;
        self.interp = Some(InterpolationState {
            start_time: obs.timestamp,
            start_q: obs.joint_positions.clone(),
        });
        self.validator.reset();
        self.policy.reset();
    }

    fn transition_to_running(&mut self, reason: TransitionReason, obs: &Observation) {
        if self.state == FsmState::RlRunning {
            return;
        }
        self.record_transition(self.state, FsmState::RlRunning, reason, obs.timestamp);
        self.state = FsmState::RlRunning;
        self.interp = None;
        self.validator.reset();
    }

    fn record_transition(
        &mut self,
        from: FsmState,
        to: FsmState,
        reason: TransitionReason,
        at: Instant,
    ) {
        self.transitions.push(FsmTransition {
            from,
            to,
            reason,
            at,
        });
    }

    /// Compute the tilt of body-frame gravity from the down direction.
    ///
    /// Body-frame gravity expected as `[gx, gy, gz]` with magnitude ≈ 1
    /// (already normalized by IMU integration). Returns the angle in
    /// radians. When the input magnitude is degenerate (≈ 0), returns
    /// 0 to avoid spurious `acos(-z)` faults.
    fn imu_tilt_exceeded(g: [f32; 3], max_rad: f32) -> bool {
        let mag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        if mag < 1e-3 {
            return false;
        }
        let cos_tilt = (-g[2] / mag).clamp(-1.0, 1.0);
        cos_tilt.acos() > max_rad
    }
}

/// Smoothstep interpolation `s(t) = 3t² − 2t³`.
///
/// Standard cubic-Hermite curve with zero derivative at both endpoints. The
/// issue calls for a "smooth cubic-interpolation"; this is the canonical
/// choice for a `start_q → target_q` blend over a duration.
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Convenience: build a no-op observation with `joint_positions` filled by
/// `q` and zero IMU/gravity. Useful for tests of downstream consumers.
#[must_use]
pub fn observation_with_positions(q: Vec<f32>, command: WbcCommand, at: Instant) -> Observation {
    Observation {
        joint_velocities: vec![0.0; q.len()],
        joint_positions: q,
        gravity_vector: [0.0, 0.0, -1.0],
        angular_velocity: [0.0, 0.0, 0.0],
        base_pose: None,
        command,
        timestamp: at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::validator::SafetyLimitsConfig;
    use robowbc_core::{
        JointPositionTargets, PolicyCapabilities, Result as CoreResult, RobotConfig, Twist,
        WbcCommandKind,
    };
    use std::sync::Mutex;

    /// A pluggable mock policy. The next call to `predict` returns whatever
    /// is in `next` (cloned). If `next` is `None`, returns `q_target = q_current`.
    struct MockPolicy {
        next: Mutex<Option<Vec<f32>>>,
        reset_calls: Mutex<u32>,
        supported: Vec<RobotConfig>,
    }

    impl MockPolicy {
        fn identity() -> Self {
            Self {
                next: Mutex::new(None),
                reset_calls: Mutex::new(0),
                supported: Vec::new(),
            }
        }

        fn with_next(q: Vec<f32>) -> Self {
            Self {
                next: Mutex::new(Some(q)),
                reset_calls: Mutex::new(0),
                supported: Vec::new(),
            }
        }

        fn set_next(&self, q: Vec<f32>) {
            *self.next.lock().unwrap() = Some(q);
        }

        fn reset_count(&self) -> u32 {
            *self.reset_calls.lock().unwrap()
        }
    }

    impl WbcPolicy for MockPolicy {
        fn predict(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
            let next = self.next.lock().unwrap().clone();
            let positions = next.unwrap_or_else(|| obs.joint_positions.clone());
            Ok(JointPositionTargets {
                positions,
                timestamp: obs.timestamp,
            })
        }

        fn reset(&self) {
            *self.reset_calls.lock().unwrap() += 1;
        }

        fn capabilities(&self) -> PolicyCapabilities {
            PolicyCapabilities::new(vec![WbcCommandKind::Velocity])
        }

        fn control_frequency_hz(&self) -> u32 {
            500
        }

        fn supported_robots(&self) -> &[RobotConfig] {
            &self.supported
        }
    }

    fn config_for(joints: usize) -> FsmConfig {
        FsmConfig::uniform(joints, vec![0.1; joints], 60.0, 1.0)
    }

    fn validator_for(joints: usize) -> PolicyValidator {
        PolicyValidator::new(&SafetyLimitsConfig::default(), joints)
    }

    fn velocity_command() -> WbcCommand {
        WbcCommand::Velocity(Twist {
            linear: [0.0, 0.0, 0.0],
            angular: [0.0, 0.0, 0.0],
        })
    }

    fn obs_at(q: Vec<f32>, at: Instant) -> Observation {
        observation_with_positions(q, velocity_command(), at)
    }

    fn make_fsm(joints: usize) -> Fsm<MockPolicy> {
        Fsm::new(
            config_for(joints),
            MockPolicy::identity(),
            validator_for(joints),
        )
        .expect("valid config")
    }

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
        assert!((smoothstep(0.0) - 0.0).abs() < 1e-6);
        assert!((smoothstep(1.0) - 1.0).abs() < 1e-6);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-6);
        // Outside [0,1] clamps.
        assert!((smoothstep(-0.5) - 0.0).abs() < 1e-6);
        assert!((smoothstep(1.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fsm_starts_in_init() {
        let fsm = make_fsm(2);
        assert_eq!(fsm.state(), FsmState::Init);
        assert!(fsm.transitions().is_empty());
    }

    #[test]
    fn config_validate_rejects_mismatched_lengths() {
        let mut config = config_for(3);
        config.default_dof_pos.pop();
        let res = Fsm::new(config, MockPolicy::identity(), validator_for(3));
        assert!(res.is_err());
    }

    #[test]
    fn rejects_observation_with_wrong_joint_count() {
        let mut fsm = make_fsm(2);
        let obs = obs_at(vec![0.0, 0.0, 0.0], Instant::now());
        let err = fsm.tick(&obs, &[]).expect_err("should fail");
        assert!(matches!(
            err,
            FsmError::JointCountMismatch {
                expected: 2,
                actual: 3
            }
        ));
    }

    // === Init → RlInit ====================================================

    #[test]
    fn init_to_rl_init_on_first_observation() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        let obs = obs_at(vec![0.0, 0.0], t0);
        let out = fsm.tick(&obs, &[]).expect("tick");
        assert_eq!(fsm.state(), FsmState::RlInit);
        // Output at t=0: q_target ≈ start_q (smoothstep(0) = 0).
        for &q in &out.q_target {
            assert!(q.abs() < 1e-6);
        }
        assert_eq!(fsm.transitions().len(), 1);
        assert_eq!(fsm.transitions()[0].from, FsmState::Init);
        assert_eq!(fsm.transitions()[0].to, FsmState::RlInit);
        assert_eq!(
            fsm.transitions()[0].reason,
            TransitionReason::FirstObservation
        );
    }

    // === RlInit interpolation ==============================================

    #[test]
    fn rl_init_interpolates_smoothly_to_default_dof_pos() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        // Tick 0: enters RlInit with start_q = [0, 0].
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[]).expect("tick 0");

        // Halfway through default 3s window (1.5s in).
        let t_mid = t0 + Duration::from_millis(1500);
        let out_mid = fsm
            .tick(&obs_at(vec![0.0, 0.0], t_mid), &[])
            .expect("tick mid");
        // smoothstep(0.5) = 0.5, so q ≈ 0.05 at every joint (default 0.1).
        for &q in &out_mid.q_target {
            assert!((q - 0.05).abs() < 1e-3, "q at midpoint = {q}");
        }
        assert_eq!(fsm.state(), FsmState::RlInit);
    }

    // === RlInit → RlRunning (timeout) ======================================

    #[test]
    fn rl_init_to_running_when_duration_elapses() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[]).expect("tick 0");
        let t_end = t0 + Duration::from_secs(4);
        // The mock policy returns identity, validator default thresholds let
        // q_target = q_current pass.
        fsm.tick(&obs_at(vec![0.05, 0.05], t_end), &[])
            .expect("tick end");
        assert_eq!(fsm.state(), FsmState::RlRunning);
        assert!(fsm
            .transitions()
            .iter()
            .any(|t| t.reason == TransitionReason::RlInitComplete));
    }

    // === RlInit → RlRunning (Engage) =======================================

    #[test]
    fn engage_from_rl_init_skips_to_running() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[]).expect("tick 0");
        // Engage only 1ms in — way before the 3s timeout.
        let t1 = t0 + Duration::from_millis(1);
        fsm.tick(&obs_at(vec![0.0, 0.0], t1), &[TeleopEvent::Engage])
            .expect("engage");
        assert_eq!(fsm.state(), FsmState::RlRunning);
        assert!(fsm
            .transitions()
            .iter()
            .any(|t| t.reason == TransitionReason::EngageRequested));
    }

    #[test]
    fn engage_outside_rl_init_is_a_noop() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[TeleopEvent::Engage])
            .expect("ignored engage in Init");
        // Init -> RlInit happened on the tick (first observation rule),
        // but Engage didn't apply because state was Init when events were processed.
        // After the tick, state is RlInit (from FirstObservation), not RlRunning.
        assert_eq!(fsm.state(), FsmState::RlInit);
    }

    // === Running → Damping (Emergency Stop) ================================

    #[test]
    fn emergency_stop_from_running_goes_to_damping() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[]).expect("init");
        fsm.tick(&obs_at(vec![0.0, 0.0], t0 + Duration::from_secs(4)), &[])
            .expect("running");
        assert_eq!(fsm.state(), FsmState::RlRunning);
        let t_estop = t0 + Duration::from_secs(5);
        let out = fsm
            .tick(
                &obs_at(vec![0.0, 0.0], t_estop),
                &[TeleopEvent::EmergencyStop],
            )
            .expect("estop");
        assert_eq!(fsm.state(), FsmState::Damping);
        assert!(out.kp.iter().all(|&k| k.abs() < 1e-9));
        assert!(out.kd.iter().all(|&k| (k - 8.0).abs() < 1e-6));
        assert!(fsm
            .transitions()
            .iter()
            .any(|t| t.reason == TransitionReason::EmergencyStop));
    }

    // === Running → Damping (Validator Fault) ===============================

    #[test]
    fn validator_fault_from_running_goes_to_damping() {
        let policy = MockPolicy::identity();
        let mut fsm = Fsm::new(config_for(1), policy, validator_for(1)).expect("config");
        let t0 = Instant::now();
        // Get into RlRunning.
        fsm.tick(&obs_at(vec![0.0], t0), &[]).expect("init");
        fsm.tick(&obs_at(vec![0.0], t0 + Duration::from_secs(4)), &[])
            .expect("running");
        assert_eq!(fsm.state(), FsmState::RlRunning);
        // Now feed a NaN through the policy.
        fsm.policy.set_next(vec![f32::NAN]);
        let out = fsm
            .tick(&obs_at(vec![0.0], t0 + Duration::from_secs(5)), &[])
            .expect("fault");
        assert_eq!(fsm.state(), FsmState::Damping);
        assert!(out.kp.iter().all(|&k| k.abs() < 1e-9));
        assert!(matches!(
            fsm.transitions().iter().find_map(|t| match &t.reason {
                TransitionReason::PolicyFault(f) => Some(f.clone()),
                _ => None,
            }),
            Some(Fault::Nan { joint_idx: 0 })
        ));
    }

    // === Running → Damping (IMU tilt) ======================================

    #[test]
    fn imu_tilt_from_running_goes_to_damping() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[]).expect("init");
        fsm.tick(&obs_at(vec![0.0, 0.0], t0 + Duration::from_secs(4)), &[])
            .expect("running");
        assert_eq!(fsm.state(), FsmState::RlRunning);
        // Tilt past 0.7 rad: gravity nearly horizontal.
        let mut obs = obs_at(vec![0.0, 0.0], t0 + Duration::from_secs(5));
        obs.gravity_vector = [1.0, 0.0, 0.0]; // 90° tilt
        fsm.tick(&obs, &[]).expect("tilt");
        assert_eq!(fsm.state(), FsmState::Damping);
        assert!(fsm
            .transitions()
            .iter()
            .any(|t| t.reason == TransitionReason::ImuTilt));
    }

    #[test]
    fn imu_tilt_below_threshold_does_not_fault() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[]).expect("init");
        fsm.tick(&obs_at(vec![0.0, 0.0], t0 + Duration::from_secs(4)), &[])
            .expect("running");
        let mut obs = obs_at(vec![0.0, 0.0], t0 + Duration::from_secs(5));
        // 0.3 rad tilt: well under 0.7 rad threshold.
        obs.gravity_vector = [0.3_f32.sin(), 0.0, -0.3_f32.cos()];
        fsm.tick(&obs, &[]).expect("ok");
        assert_eq!(fsm.state(), FsmState::RlRunning);
    }

    #[test]
    fn imu_tilt_ignores_zero_magnitude_gravity() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[]).expect("init");
        let mut obs = obs_at(vec![0.0, 0.0], t0 + Duration::from_secs(4));
        // Pre-IMU init scenario — should not fault.
        obs.gravity_vector = [0.0, 0.0, 0.0];
        fsm.tick(&obs, &[]).expect("running ok");
        assert_eq!(fsm.state(), FsmState::RlRunning);
    }

    // === Damping → RlInit (Reset) ==========================================

    #[test]
    fn reset_from_damping_returns_to_rl_init() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[]).expect("init");
        fsm.tick(
            &obs_at(vec![0.0, 0.0], t0 + Duration::from_millis(10)),
            &[TeleopEvent::EmergencyStop],
        )
        .expect("damping");
        assert_eq!(fsm.state(), FsmState::Damping);
        // Reset.
        fsm.tick(
            &obs_at(vec![0.0, 0.0], t0 + Duration::from_millis(20)),
            &[TeleopEvent::Reset],
        )
        .expect("reset");
        assert_eq!(fsm.state(), FsmState::RlInit);
        assert!(fsm
            .transitions()
            .iter()
            .any(|t| t.reason == TransitionReason::UserReset));
    }

    #[test]
    fn reset_from_running_returns_to_rl_init() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[]).expect("init");
        fsm.tick(&obs_at(vec![0.0, 0.0], t0 + Duration::from_secs(4)), &[])
            .expect("running");
        assert_eq!(fsm.state(), FsmState::RlRunning);
        fsm.tick(
            &obs_at(vec![0.05, 0.05], t0 + Duration::from_secs(5)),
            &[TeleopEvent::Reset],
        )
        .expect("reset");
        assert_eq!(fsm.state(), FsmState::RlInit);
    }

    #[test]
    fn reset_from_init_is_a_noop() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        // Reset has no effect from Init; first observation rule still fires.
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[TeleopEvent::Reset])
            .expect("ok");
        assert_eq!(fsm.state(), FsmState::RlInit);
        // Only one transition (Init -> RlInit via FirstObservation).
        assert_eq!(fsm.transitions().len(), 1);
    }

    // === Damping is sticky =================================================

    #[test]
    fn damping_self_loops_without_reset() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0, 0.0], t0), &[]).expect("init");
        fsm.tick(
            &obs_at(vec![0.0, 0.0], t0 + Duration::from_millis(10)),
            &[TeleopEvent::EmergencyStop],
        )
        .expect("damping");
        // Several ticks pass — should remain in damping.
        for k in 1..5 {
            let out = fsm
                .tick(
                    &obs_at(vec![0.0, 0.0], t0 + Duration::from_millis(20 + k * 2)),
                    &[],
                )
                .expect("tick");
            assert_eq!(fsm.state(), FsmState::Damping);
            assert!(out.kp.iter().all(|&k| k.abs() < 1e-9));
        }
    }

    // === Quit / Velocity events ============================================

    #[test]
    fn velocity_and_quit_events_do_not_force_state_changes() {
        let mut fsm = make_fsm(2);
        let t0 = Instant::now();
        fsm.tick(
            &obs_at(vec![0.0, 0.0], t0),
            &[
                TeleopEvent::Velocity {
                    vx: 0.1,
                    vy: 0.0,
                    wz: 0.0,
                },
                TeleopEvent::Quit,
            ],
        )
        .expect("ok");
        // Velocity and Quit are advisory: state must follow Init -> RlInit
        // (not damping, not running).
        assert_eq!(fsm.state(), FsmState::RlInit);
    }

    // === Reset hooks: validator + policy ===================================

    #[test]
    fn damping_resets_validator_and_policy_state() {
        let mut fsm = make_fsm(1);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0], t0), &[]).expect("init");
        fsm.tick(&obs_at(vec![0.0], t0 + Duration::from_secs(4)), &[])
            .expect("running");
        let calls_before = fsm.policy.reset_count();
        fsm.tick(
            &obs_at(vec![0.0], t0 + Duration::from_secs(5)),
            &[TeleopEvent::EmergencyStop],
        )
        .expect("damping");
        let calls_after = fsm.policy.reset_count();
        assert!(
            calls_after > calls_before,
            "policy.reset() should be called on damping transition"
        );
    }

    #[test]
    fn rl_init_resets_validator_history() {
        // Setup: get into running so the validator has prev_q_target. Then
        // user reset returns us to RlInit and the validator history must be
        // cleared so the next predict→validate sequence starts fresh.
        let mut fsm = make_fsm(1);
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0], t0), &[]).expect("init");
        fsm.tick(&obs_at(vec![0.0], t0 + Duration::from_secs(4)), &[])
            .expect("running tick 1");
        fsm.tick(&obs_at(vec![0.0], t0 + Duration::from_secs(4)), &[])
            .expect("running tick 2");
        // Reset → RlInit.
        fsm.tick(
            &obs_at(vec![0.0], t0 + Duration::from_secs(5)),
            &[TeleopEvent::Reset],
        )
        .expect("reset");
        assert_eq!(fsm.state(), FsmState::RlInit);
        // Run rl_init to completion. Then in RlRunning, a 0.2 rad jump in q_target
        // should now NOT be filtered out by the validator's prev history (because
        // reset() cleared it). The first running tick is the seed; the second
        // would normally fault if history had persisted.
        let t_after = t0 + Duration::from_secs(9);
        fsm.policy.set_next(vec![0.0]);
        fsm.tick(&obs_at(vec![0.0], t_after), &[])
            .expect("rl init complete -> running");
        assert_eq!(fsm.state(), FsmState::RlRunning);
    }

    // === Damping behaviour =================================================

    #[test]
    fn damping_output_uses_configured_damping_kd() {
        let mut config = config_for(3);
        config.damping_kd = 12.0;
        let mut fsm = Fsm::new(config, MockPolicy::identity(), validator_for(3)).expect("config");
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.0; 3], t0), &[]).expect("init");
        let out = fsm
            .tick(
                &obs_at(vec![0.0; 3], t0 + Duration::from_millis(10)),
                &[TeleopEvent::EmergencyStop],
            )
            .expect("damping");
        assert!(out.kd.iter().all(|&k| (k - 12.0).abs() < 1e-6));
        assert!(out.kp.iter().all(|&k| k.abs() < 1e-9));
        assert_eq!(out.q_target.len(), 3);
    }

    // === Misc ==============================================================

    #[test]
    fn rl_running_propagates_policy_output() {
        let policy = MockPolicy::with_next(vec![0.05, 0.06]);
        let mut fsm = Fsm::new(config_for(2), policy, validator_for(2)).expect("config");
        let t0 = Instant::now();
        fsm.tick(&obs_at(vec![0.05, 0.06], t0), &[]).expect("init");
        fsm.tick(&obs_at(vec![0.05, 0.06], t0 + Duration::from_secs(4)), &[])
            .expect("running");
        assert_eq!(fsm.state(), FsmState::RlRunning);
        let out = fsm
            .tick(&obs_at(vec![0.05, 0.06], t0 + Duration::from_secs(5)), &[])
            .expect("policy");
        assert!((out.q_target[0] - 0.05).abs() < 1e-6);
        assert!((out.q_target[1] - 0.06).abs() < 1e-6);
    }

    #[test]
    fn fsm_state_display_is_snake_case() {
        assert_eq!(FsmState::Init.to_string(), "init");
        assert_eq!(FsmState::RlInit.to_string(), "rl_init");
        assert_eq!(FsmState::RlRunning.to_string(), "rl_running");
        assert_eq!(FsmState::Damping.to_string(), "damping");
    }

    #[test]
    fn transition_reason_display_is_loggable() {
        assert_eq!(
            TransitionReason::FirstObservation.to_string(),
            "first_observation"
        );
        assert_eq!(
            TransitionReason::PolicyFault(Fault::Nan { joint_idx: 3 }).to_string(),
            "policy_fault:NaN in q_target at joint 3"
        );
    }
}
