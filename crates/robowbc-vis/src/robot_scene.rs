use robowbc_core::RobotConfig;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone)]
pub(crate) struct RobotScene {
    bodies: Vec<BodyNode>,
    roots: Vec<usize>,
    edges: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
struct BodyNode {
    translation: [f32; 3],
    rotation_xyzw: [f32; 4],
    joints: Vec<JointSpec>,
    children: Vec<usize>,
}

#[derive(Debug, Clone)]
struct JointSpec {
    robot_joint_index: Option<usize>,
    kind: JointKind,
    position: [f32; 3],
    axis: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JointKind {
    Hinge,
    Slide,
    Ball,
    Free,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SkeletonPose {
    pub(crate) body_positions: Vec<[f32; 3]>,
    pub(crate) segments: Vec<[[f32; 3]; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Transform {
    translation: [f32; 3],
    rotation_xyzw: [f32; 4],
}

impl Transform {
    const IDENTITY: Self = Self {
        translation: [0.0, 0.0, 0.0],
        rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
    };

    fn from_translation(translation: [f32; 3]) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    fn from_rotation(rotation_xyzw: [f32; 4]) -> Self {
        Self {
            rotation_xyzw: normalize_quaternion(rotation_xyzw),
            ..Self::IDENTITY
        }
    }

    fn from_translation_rotation(translation: [f32; 3], rotation_xyzw: [f32; 4]) -> Self {
        Self {
            translation,
            rotation_xyzw: normalize_quaternion(rotation_xyzw),
        }
    }

    fn compose(self, child: Self) -> Self {
        Self {
            translation: add3(
                self.translation,
                rotate_vector(self.rotation_xyzw, child.translation),
            ),
            rotation_xyzw: normalize_quaternion(quaternion_multiply(
                self.rotation_xyzw,
                child.rotation_xyzw,
            )),
        }
    }
}

impl JointSpec {
    fn local_transform(&self, joint_positions: &[f32]) -> Transform {
        let value = self
            .robot_joint_index
            .and_then(|index| joint_positions.get(index))
            .copied()
            .unwrap_or(0.0);

        match self.kind {
            JointKind::Hinge => Transform::from_translation(self.position)
                .compose(Transform::from_rotation(quaternion_from_axis_angle(
                    self.axis, value,
                )))
                .compose(Transform::from_translation(scale3(self.position, -1.0))),
            JointKind::Slide => Transform::from_translation(scale3(self.axis, value)),
            JointKind::Ball | JointKind::Free => Transform::IDENTITY,
        }
    }
}

impl RobotScene {
    pub(crate) fn from_robot(robot: &RobotConfig) -> Result<Option<Self>, String> {
        let Some(model_path) = &robot.model_path else {
            return Ok(None);
        };

        let extension = model_path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or_default();
        if !extension.eq_ignore_ascii_case("xml") {
            return Err(format!(
                "unsupported robot model format for Rerun scene logging: {} (expected MJCF .xml)",
                model_path.display()
            ));
        }

        let mjcf = fs::read_to_string(model_path)
            .map_err(|e| format!("failed to read MJCF model {}: {e}", model_path.display()))?;
        let joint_lookup: HashMap<&str, usize> = robot
            .joint_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();

        Self::from_mjcf_str(&mjcf, &joint_lookup).map(Some)
    }

    pub(crate) fn pose(&self, joint_positions: &[f32]) -> SkeletonPose {
        let mut body_positions = vec![[0.0, 0.0, 0.0]; self.bodies.len()];
        for &root in &self.roots {
            self.populate_body_positions(
                root,
                Transform::IDENTITY,
                joint_positions,
                &mut body_positions,
            );
        }

        let segments = self
            .edges
            .iter()
            .filter_map(|&(parent, child)| {
                let segment = [body_positions[parent], body_positions[child]];
                (distance_squared(segment[0], segment[1]) > 1e-8).then_some(segment)
            })
            .collect();

        SkeletonPose {
            body_positions,
            segments,
        }
    }

    fn from_mjcf_str(xml: &str, joint_lookup: &HashMap<&str, usize>) -> Result<Self, String> {
        let doc = roxmltree::Document::parse(xml)
            .map_err(|e| format!("failed to parse MJCF XML: {e}"))?;
        let worldbody = doc
            .descendants()
            .find(|node| node.has_tag_name("worldbody"))
            .ok_or("MJCF model is missing <worldbody>".to_owned())?;

        let mut scene = Self {
            bodies: Vec::new(),
            roots: Vec::new(),
            edges: Vec::new(),
        };

        for body in worldbody
            .children()
            .filter(|node| node.is_element() && node.has_tag_name("body"))
        {
            let root = scene.parse_body(body, None, joint_lookup)?;
            scene.roots.push(root);
        }

        if scene.bodies.is_empty() {
            return Err("MJCF model has no <body> nodes under <worldbody>".to_owned());
        }

        Ok(scene)
    }

    fn parse_body(
        &mut self,
        node: roxmltree::Node<'_, '_>,
        parent: Option<usize>,
        joint_lookup: &HashMap<&str, usize>,
    ) -> Result<usize, String> {
        let body_label = node.attribute("name").unwrap_or("<unnamed body>");
        let translation = parse_vec3(node.attribute("pos"))
            .map_err(|e| format!("body `{body_label}` has invalid pos: {e}"))?;
        let rotation_xyzw = parse_mjcf_quaternion(node.attribute("quat"))
            .map_err(|e| format!("body `{body_label}` has invalid quat: {e}"))?;

        let joints = node
            .children()
            .filter(|child| child.is_element() && child.has_tag_name("joint"))
            .map(|joint| parse_joint(joint, body_label, joint_lookup))
            .collect::<Result<Vec<_>, _>>()?;

        let body_index = self.bodies.len();
        self.bodies.push(BodyNode {
            translation,
            rotation_xyzw,
            joints,
            children: Vec::new(),
        });

        if let Some(parent_index) = parent {
            self.edges.push((parent_index, body_index));
            self.bodies[parent_index].children.push(body_index);
        }

        for child_body in node
            .children()
            .filter(|child| child.is_element() && child.has_tag_name("body"))
        {
            self.parse_body(child_body, Some(body_index), joint_lookup)?;
        }

        Ok(body_index)
    }

    fn populate_body_positions(
        &self,
        body_index: usize,
        parent_transform: Transform,
        joint_positions: &[f32],
        body_positions: &mut [[f32; 3]],
    ) {
        let body = &self.bodies[body_index];
        let mut body_transform = parent_transform.compose(Transform::from_translation_rotation(
            body.translation,
            body.rotation_xyzw,
        ));

        for joint in &body.joints {
            body_transform = body_transform.compose(joint.local_transform(joint_positions));
        }

        body_positions[body_index] = body_transform.translation;

        for &child in &body.children {
            self.populate_body_positions(child, body_transform, joint_positions, body_positions);
        }
    }
}

fn parse_joint(
    joint: roxmltree::Node<'_, '_>,
    body_label: &str,
    joint_lookup: &HashMap<&str, usize>,
) -> Result<JointSpec, String> {
    let kind = match joint.attribute("type").unwrap_or("hinge") {
        "hinge" => JointKind::Hinge,
        "slide" => JointKind::Slide,
        "ball" => JointKind::Ball,
        "free" => JointKind::Free,
        other => {
            return Err(format!(
                "body `{body_label}` uses unsupported MJCF joint type `{other}`"
            ));
        }
    };

    let position = parse_vec3(joint.attribute("pos"))
        .map_err(|e| format!("body `{body_label}` has invalid joint pos: {e}"))?;
    let axis = normalize_axis(
        parse_vec3_with_default(joint.attribute("axis"), [0.0, 0.0, 1.0])
            .map_err(|e| format!("body `{body_label}` has invalid joint axis: {e}"))?,
    );
    let robot_joint_index = joint
        .attribute("name")
        .and_then(|name| joint_lookup.get(name).copied());

    Ok(JointSpec {
        robot_joint_index,
        kind,
        position,
        axis,
    })
}

fn parse_vec3(raw: Option<&str>) -> Result<[f32; 3], String> {
    parse_vec3_with_default(raw, [0.0, 0.0, 0.0])
}

fn parse_vec3_with_default(raw: Option<&str>, default: [f32; 3]) -> Result<[f32; 3], String> {
    raw.map_or(Ok(default), parse_floats::<3>)
}

fn parse_mjcf_quaternion(raw: Option<&str>) -> Result<[f32; 4], String> {
    raw.map_or(Ok([0.0, 0.0, 0.0, 1.0]), |value| {
        let [w, x, y, z] = parse_floats::<4>(value)?;
        Ok(normalize_quaternion([x, y, z, w]))
    })
}

fn parse_floats<const N: usize>(raw: &str) -> Result<[f32; N], String> {
    let values = raw
        .split_whitespace()
        .map(|part| {
            part.parse::<f32>()
                .map_err(|e| format!("failed to parse float `{part}`: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let found = values.len();
    values
        .try_into()
        .map_err(|_| format!("expected {N} values, found {found}"))
}

fn normalize_axis(axis: [f32; 3]) -> [f32; 3] {
    let norm = length3(axis);
    if norm <= f32::EPSILON {
        [0.0, 0.0, 1.0]
    } else {
        scale3(axis, 1.0 / norm)
    }
}

fn quaternion_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let half_angle = angle * 0.5;
    let sin_half = half_angle.sin();
    let axis = normalize_axis(axis);
    normalize_quaternion([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        half_angle.cos(),
    ])
}

fn quaternion_multiply(lhs: [f32; 4], rhs: [f32; 4]) -> [f32; 4] {
    let [lx, ly, lz, lw] = lhs;
    let [rx, ry, rz, rw] = rhs;
    [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]
}

fn normalize_quaternion(quaternion: [f32; 4]) -> [f32; 4] {
    let norm = (quaternion[0] * quaternion[0]
        + quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2]
        + quaternion[3] * quaternion[3])
        .sqrt();
    if norm <= f32::EPSILON {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [
            quaternion[0] / norm,
            quaternion[1] / norm,
            quaternion[2] / norm,
            quaternion[3] / norm,
        ]
    }
}

fn rotate_vector(quaternion: [f32; 4], vector: [f32; 3]) -> [f32; 3] {
    let [qx, qy, qz, qw] = normalize_quaternion(quaternion);
    let qvec = [qx, qy, qz];
    let uv = cross(qvec, vector);
    let uuv = cross(qvec, uv);
    add3(vector, add3(scale3(uv, 2.0 * qw), scale3(uuv, 2.0)))
}

fn add3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn scale3(vector: [f32; 3], scale: f32) -> [f32; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn cross(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn length3(vector: [f32; 3]) -> f32 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn distance_squared(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    let dx = lhs[0] - rhs[0];
    let dy = lhs[1] - rhs[1];
    let dz = lhs[2] - rhs[2];
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::f32::consts::FRAC_PI_2;
    use std::path::PathBuf;

    fn assert_close3(actual: [f32; 3], expected: [f32; 3]) {
        for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-4,
                "component {index} mismatch: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn mjcf_quaternions_use_wxyz_order() {
        let xml = r#"
<mujoco>
  <worldbody>
    <body name="root">
      <body name="rotated" quat="0.70710677 0 0 0.70710677">
        <body name="tip" pos="1 0 0" />
      </body>
    </body>
  </worldbody>
</mujoco>
"#;

        let scene = RobotScene::from_mjcf_str(xml, &HashMap::<&str, usize>::new())
            .expect("scene should parse");
        let pose = scene.pose(&[]);
        assert_eq!(pose.segments.len(), 1);
        assert_close3(pose.segments[0][0], [0.0, 0.0, 0.0]);
        assert_close3(pose.segments[0][1], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn hinge_joints_rotate_descendants() {
        let xml = r#"
<mujoco>
  <worldbody>
    <body name="root">
      <body name="shoulder">
        <joint name="yaw_joint" axis="0 0 1" />
        <body name="tip" pos="1 0 0" />
      </body>
    </body>
  </worldbody>
</mujoco>
"#;

        let joint_lookup = HashMap::from([("yaw_joint", 0_usize)]);
        let scene =
            RobotScene::from_mjcf_str(xml, &joint_lookup).expect("scene should parse with joint");
        let pose = scene.pose(&[FRAC_PI_2]);
        assert_eq!(pose.segments.len(), 1);
        assert_close3(pose.segments[0][1], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn unitree_g1_model_builds_non_empty_skeleton() {
        let config_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/robots/unitree_g1.toml");
        let mut robot =
            RobotConfig::from_toml_file(&config_path).expect("unitree g1 config should load");
        robot.model_path = Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/robots/unitree_g1/g1_29dof.xml"),
        );

        let scene = RobotScene::from_robot(&robot)
            .expect("scene construction should succeed")
            .expect("unitree g1 should expose a scene");
        let pose = scene.pose(&robot.default_pose);

        assert!(pose.segments.len() > 20);
        let min_z = pose
            .body_positions
            .iter()
            .map(|point| point[2])
            .fold(f32::INFINITY, f32::min);
        let max_z = pose
            .body_positions
            .iter()
            .map(|point| point[2])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max_z > 0.7, "expected torso height > 0.7 m, got {max_z}");
        assert!(min_z < 0.1, "expected feet near the ground, got {min_z}");
    }
}
