# G1 MJCF Review (2026-04-24)

This note records the Unitree G1 MJCF review used to decide which MuJoCo model
RoboWBC should use for each public policy integration. The upstream repos move
quickly, so this document pins the exact commits reviewed on April 24, 2026.

## Review Snapshot

| Repository | Commit | Role In This Review | Notes |
|-----------|--------|---------------------|-------|
| `robowbc` | `f1b940ac0f78c77252c18eb69b8d9f2f6effd3b5` | Local baseline | Current repo under review |
| `third_party/GR00T-WholeBodyControl` | `bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8` | Vendored GR00T / GEAR-Sonic / Decoupled WBC assets | Local third-party checkout already present in this repo |
| `nvidia-isaac/WBC-AGILE` | `de7f1adf6c1b70ca1e1451051bf0507b16bbbe5f` | Upstream WBC-AGILE reference | Reviewed from a temporary shallow clone |
| `LeCAR-Lab/BFM-Zero` | `b87916f52d3d9e6eeba484f5e80851a235191837` | Upstream BFM-Zero reference | Cloned with `GIT_LFS_SKIP_SMUDGE=1` because the upstream LFS budget was exhausted; XML and config files were still available |
| `unitreerobotics/unitree_mujoco` | `1a37b051a10be723405b7ed6dc839361af036d88` | Canonical Unitree G1 MuJoCo model | Reviewed from a temporary shallow clone |

## Post-Review Implementation Status

- After this review, RoboWBC vendored the official Unitree
  `unitree_robots/g1/meshes/` directory under
  `assets/robots/unitree_g1/meshes/` from the pinned commit above.
- RoboWBC also added local runnable mirrors for the reviewed GEAR-Sonic and
  Decoupled WBC **base robot XMLs** under:
  - `assets/robots/groot_g1_gear_sonic/g1_29dof_old.xml`
  - `assets/robots/groot_g1_decoupled_wbc/g1_29dof_old.xml`
- The current MuJoCo runtime path now prefers exact reviewed **include-free
  base robot XMLs** in `[sim].model_path` rather than scene wrappers. This
  keeps the runtime on the official robot embodiments while avoiding the
  current nested-`<include>` fallback gap in `robowbc-sim`.
- `robot.config_path` and `robot.model_path` remain on include-free robot files
  so `robowbc-vis` can keep parsing the Rerun skeleton without MJCF
  `<include>` support.

## Review Constraints

- `robowbc-vis` reads a single MJCF XML file and does **not** resolve
  `<include>` tags when building the Rerun skeleton.
- `robowbc-sim` and the proof-pack MuJoCo replay path **can** load include-based
  scene wrappers when the referenced assets are present on disk.
- Because of that split, the reviewed policy configs intentionally separate:
  - `robot.config_path` / `robot.model_path`: include-free MJCF for Rerun robot
    scene parsing
  - `[sim].model_path`: policy-specific official scene or model used by MuJoCo
    stepping and proof-pack screenshot capture

## Upstream File Inventory

| Policy / Source | Reviewed File(s) | Local Availability In This Repo | License / Distribution Note | Summary |
|-----------------|------------------|---------------------------------|-----------------------------|---------|
| Shared RoboWBC G1 | `assets/robots/unitree_g1/g1_29dof.xml`, `scene_29dof.xml`, `meshes/` | Yes | Project-local mirror of the reviewed Unitree XML and mesh bundle | Shared baseline used by current RoboWBC G1 robot configs and proof-pack capture |
| Unitree official | `unitree_robots/g1/g1_29dof.xml`, `unitree_robots/g1/scene_29dof.xml`, `unitree_robots/g1/meshes/` | Fully mirrored locally under `assets/robots/unitree_g1/` | BSD-3-Clause upstream | Canonical Unitree G1 MuJoCo model family |
| GEAR-Sonic deploy | `gear_sonic_deploy/g1/g1_29dof.xml`, `gear_sonic_deploy/g1/g1_29dof_old.xml`, `gear_sonic_deploy/g1/scene_29dof.xml` | Yes via vendored `third_party/GR00T-WholeBodyControl` | Already vendored in repo | Provides both a slim deploy XML and an older freebase scene model |
| Decoupled WBC | `decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml`, `scene_29dof.xml` | Yes via vendored `third_party/GR00T-WholeBodyControl` | Already vendored in repo | Richer old freebase model with contact helper sites |
| WBC-AGILE | No G1 MJCF vendored in upstream repo; docs/tests point to Unitree official `scene_29dof.xml` / `g1_29dof.xml` | No distinct WBC-AGILE XML vendored upstream | Upstream delegates G1 MuJoCo assets to `unitree_mujoco` | Use Unitree official G1 as the reviewed official model for WBC-AGILE |
| BFM-Zero | `humanoidverse/data/robots/g1/g1_29dof.xml`, `g1_29dof_mujoco.xml`, `g1_29dof_old_freebase_noadditional_actuators.xml`, matching scene wrappers | Not vendored in this repo | Upstream repo is CC BY-NC 4.0; do not silently copy its scene XMLs into RoboWBC | Upstream ships multiple G1 variants for different MuJoCo tasks |

## Core Base-Model Comparison

Counts below include the floating-base joint, so the "Joints" column reads as
`30` for every 29-DOF G1 embodiment.

| Candidate | Repository / Path | Joints | Bodies | Mesh Refs | Sites | Sensors | Motors | Default Classes | Notable Extras |
|-----------|-------------------|--------|--------|-----------|-------|---------|--------|-----------------|----------------|
| RoboWBC shared G1 | `assets/robots/unitree_g1/g1_29dof.xml` | 30 | 30 | 36 | `imu`, `secondary_imu` | 8 | 29 | Yes | Shared baseline; secondary IMU and fuller sensor block |
| Unitree official G1 | `unitree_robots/g1/g1_29dof.xml` | 30 | 30 | 36 | `imu`, `secondary_imu` | 8 | 29 | Yes | **Exact match** with RoboWBC shared G1 |
| GEAR / BFM slim deploy XML | `gear_sonic_deploy/g1/g1_29dof.xml` and `BFM-Zero/humanoidverse/.../g1_29dof.xml` | 30 | 30 | 36 | `imu` | 2 | 29 | No | Minimal deploy-only XML; no secondary IMU; no default classes |
| GEAR old freebase XML | `gear_sonic_deploy/g1/g1_29dof_old.xml` | 30 | 30 | 36 | `imu` | 5 | 29 | Yes | Older freebase model used by GEAR scene wrapper |
| Decoupled WBC old freebase XML | `decoupled_wbc/.../g1_29dof_old.xml` | 30 | 40 | 36 | `imu` plus 8 foot-contact sites | 5 | 29 | Yes | Dummy foot-contact bodies and helper sites |
| BFM MuJoCo XML | `BFM-Zero/humanoidverse/.../g1_29dof_mujoco.xml` | 30 | 32 | 36 | `imu` | 5 | 35 | Yes | Track camera, hand bodies, stand keyframe, 6 floating-base actuators |
| BFM old freebase no-additional-actuators XML | `BFM-Zero/humanoidverse/.../g1_29dof_old_freebase_noadditional_actuators.xml` | 30 | 40 | 36 | `imu` plus 8 foot-contact sites | 15 | 29 | Yes | Cameras, contact-force sensors, helper sites, stand keyframe |

## Scene-Wrapper Comparison

| Scene Wrapper | Include Target | Worldbody Extras | Local Availability | Notes |
|---------------|----------------|------------------|--------------------|-------|
| Unitree `scene_29dof.xml` | `g1_29dof.xml` | Floor only | Vendored under `assets/robots/unitree_g1/scene_29dof.xml` | Canonical official Unitree wrapper used by WBC-AGILE docs |
| GEAR `scene_29dof.xml` | `g1_29dof_old.xml` | Floor plus `com_marker` site | Vendored in `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/g1/` | Exact GEAR-Sonic deploy scene |
| Decoupled `scene_29dof.xml` | `g1_29dof_old.xml` | Floor, COM / hand / foot marker sites, global camera | Vendored in `third_party/GR00T-WholeBodyControl/decoupled_wbc/.../g1/` | Exact Decoupled WBC scene wrapper |
| BFM `scene_29dof_freebase_mujoco.xml` | `g1_29dof_mujoco.xml` | Floor plus marker sites | Not vendored | Upstream `G1Env` default scene |
| BFM `scene_29dof_freebase_noadditional_actuators.xml` | `g1_29dof_old_freebase_noadditional_actuators.xml` | Floor plus marker sites | Not vendored | Used by upstream reward inference path |

## Equality And Near-Equality Findings

| Finding | Result |
|---------|--------|
| RoboWBC shared G1 vs Unitree official `g1_29dof.xml` | Exact file match at the reviewed commits |
| GEAR slim deploy XML vs BFM slim deploy XML | Exact file match at the reviewed commits |
| GEAR old freebase XML vs Decoupled old freebase XML | Same joint model family; Decoupled adds helper bodies / sites |
| Decoupled old freebase XML vs BFM old freebase XML | Same core joint model family; BFM adds cameras, extra IMU aliases, `framezaxis`, force sensors, and a keyframe |

## Policy Decisions For RoboWBC

| Policy | Reviewed Official MuJoCo Model | RoboWBC `robot.config_path` | RoboWBC `sim.model_path` Decision | Exactness Status | Rationale |
|--------|-------------------------------|-----------------------------|-----------------------------------|------------------|-----------|
| `gear_sonic` | `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof_old.xml` | `configs/robots/unitree_g1_gear_sonic.toml` | Use a local runnable mirror of the official GEAR old freebase G1 XML in `[sim]` | Exact official **base** model | The GEAR mesh hashes match the Unitree official mesh bundle exactly, so RoboWBC can keep the official robot XML while reusing the shared vendored Unitree geometry locally |
| `decoupled_wbc` | `third_party/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml` | `configs/robots/unitree_g1_decoupled_wbc.toml` | Use a local runnable mirror of the official Decoupled old freebase G1 XML in `[sim]` | Exact official **base** model | RoboWBC vendors the exact upstream Decoupled mesh bundle beside the mirrored base XML so the runtime and proof-pack replay stay on the reviewed Decoupled robot embodiment |
| `wbc_agile` | Unitree official `unitree_robots/g1/g1_29dof.xml` and `scene_29dof.xml` | `configs/robots/unitree_g1_35dof_wbc_agile.toml` | Use the local vendored `assets/robots/unitree_g1/g1_29dof.xml` in `[sim]` | Exact official **base** model | WBC-AGILE upstream delegates the G1 MJCF to Unitree; RoboWBC keeps `[sim]` on the exact reviewed base XML because it is include-free and works cleanly with the current runtime fallback path |
| `bfm_zero` | Upstream BFM ships `scene_29dof_freebase_mujoco.xml` and `scene_29dof_freebase_noadditional_actuators.xml` | `configs/robots/unitree_g1_bfm_zero.toml` | Default to the Unitree base XML; allow user override to a local BFM checkout | Exact official **robot pose config**, user-selectable upstream scene | BFM upstream ships multiple NC-licensed scene variants, so RoboWBC keeps the BFM-specific joint pose / gains locally and leaves the exact scene XML as an explicit local override instead of vendoring it silently |

## Config Wiring Added In This Review

| RoboWBC Config | Change |
|----------------|--------|
| `configs/sonic_g1.toml` | Added explicit `[sim]` section pointing at a local runnable mirror of the reviewed official GEAR old freebase G1 XML |
| `configs/showcase/gear_sonic_real.toml` | Added explicit `[sim]` section pointing at the same local runnable GEAR old freebase G1 XML |
| `configs/showcase/gear_sonic_tracking_real.toml` | Added explicit `[sim]` section so the reference-motion proof pack uses the same reviewed official GEAR old freebase G1 XML |
| `configs/decoupled_g1.toml` | Added explicit `[sim]` section pointing at a local runnable mirror of the reviewed official Decoupled old freebase G1 XML |
| `configs/showcase/decoupled_wbc_real.toml` | Added explicit `[sim]` section pointing at the same local runnable Decoupled old freebase G1 XML |
| `configs/wbc_agile_g1.toml` | Added explicit `[sim]` section using the exact reviewed Unitree base XML and mesh bundle |
| `configs/showcase/wbc_agile_real.toml` | Added explicit `[sim]` section using the exact reviewed Unitree base XML and mesh bundle |
| `configs/bfm_zero_g1.toml` | Switched to `configs/robots/unitree_g1_bfm_zero.toml` and added an explicit `[sim]` section with documented override hooks for local upstream BFM scene XMLs |

## Remaining Follow-Up Work

1. If a non-commercial policy sandbox is acceptable later, add opt-in BFM
   upstream XML mirrors under a clearly isolated path instead of distributing
   them silently as the default shared model.
2. If `robowbc-vis` gains `<include>` resolution in the future, revisit whether
   `robot.model_path` can safely point at scene wrappers instead of include-free
   base XML files.
