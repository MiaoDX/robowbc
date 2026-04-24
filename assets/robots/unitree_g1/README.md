# Unitree G1 MuJoCo Model

The files in this directory are sourced from
[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco):

Reviewed upstream commit:
`1a37b051a10be723405b7ed6dc839361af036d88`

- `g1_29dof.xml` mirrors `unitree_robots/g1/g1_29dof.xml`
- `scene_29dof.xml` mirrors `unitree_robots/g1/scene_29dof.xml`
- `meshes/` mirrors `unitree_robots/g1/meshes/`

The full official mesh bundle is vendored here so the shared Unitree G1 model
can render the actual robot body in MuJoCo, roboharness proof packs, and the
published showcase site. `g1_29dof.xml` currently references 36 of the 60 STL
files, but keeping the full upstream directory preserves parity with the
official Unitree scene wrapper.

The XML files remain the source of truth for kinematic chain description, joint
definitions, limit verification, and policy-specific MuJoCo model selection.
The proof-pack renderer still has a meshless fallback for missing assets, but
the shared Unitree G1 path no longer depends on it.

## License

The original model is published by Unitree Robotics under the BSD-3-Clause
license. See the upstream repository for full license terms.
