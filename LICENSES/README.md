# Third-party licenses

robowbc itself is released under the **MIT License** — see the root
[`LICENSE`](../LICENSE) file. This directory tracks the licenses of
third-party components that robowbc depends on, vendors, or wraps at
runtime.

The layout follows SPDX / [REUSE](https://reuse.software/) convention:
one file per `<component>-<SPDX-license-id>.txt`, plus shared canonical
license texts (`Apache-2.0.txt`, `MIT.txt`, `BSD-3-Clause.txt`,
`EPL-2.0.txt`, `OFL-1.1.txt`, `Ubuntu-font-1.0.txt`) that the
per-component files reference. Per-component files document the upstream
URL, the consuming robowbc crate, and the SPDX identifier for that
component.

## Index

| Component                | License file                                                       | Upstream                                                       | Note                                                                                                |
|--------------------------|--------------------------------------------------------------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| robowbc (this repo)      | [`../LICENSE`](../LICENSE)                                         | https://github.com/MiaoDX/robowbc                              | MIT.                                                                                                |
| cyclors                  | [`cyclors-Apache-2.0.txt`](cyclors-Apache-2.0.txt)                 | https://github.com/ZettaScaleLabs/cyclors                      | Rust binding to CycloneDDS, Apache-2.0.                                                             |
| Eclipse CycloneDDS (EPL) | [`cyclonedds-EPL-2.0.txt`](cyclonedds-EPL-2.0.txt)                 | https://github.com/eclipse-cyclonedds/cyclonedds               | Vendored via cyclors; dual licensed `EPL-2.0 OR BSD-3-Clause`.                                      |
| Eclipse CycloneDDS (BSD) | [`cyclonedds-BSD-3-Clause.txt`](cyclonedds-BSD-3-Clause.txt)       | https://github.com/eclipse-cyclonedds/cyclonedds               | BSD-3-Clause branch of the dual license; either branch may be selected when redistributing.         |
| ort (Apache)             | [`ort-Apache-2.0.txt`](ort-Apache-2.0.txt)                         | https://github.com/pykeio/ort                                  | ONNX Runtime Rust binding, dual licensed `Apache-2.0 OR MIT`.                                       |
| ort (MIT)                | [`ort-MIT.txt`](ort-MIT.txt)                                       | https://github.com/pykeio/ort                                  | MIT branch of the dual license.                                                                     |
| crossterm                | [`crossterm-MIT.txt`](crossterm-MIT.txt)                           | https://github.com/crossterm-rs/crossterm                      | Cross-platform terminal input, MIT.                                                                 |
| rerun (Apache)           | [`rerun-Apache-2.0.txt`](rerun-Apache-2.0.txt)                     | https://github.com/rerun-io/rerun                              | Visualization SDK + viewer, dual licensed `Apache-2.0 OR MIT`.                                      |
| rerun (MIT)              | [`rerun-MIT.txt`](rerun-MIT.txt)                                   | https://github.com/rerun-io/rerun                              | MIT branch of the dual license.                                                                     |
| epaint default fonts (OFL) | [`epaint_default_fonts-OFL-1.1.txt`](epaint_default_fonts-OFL-1.1.txt) | https://github.com/emilk/egui/tree/main/crates/epaint_default_fonts | Default egui viewer font assets pulled transitively through mujoco-rs.                              |
| epaint default fonts (Ubuntu) | [`epaint_default_fonts-Ubuntu-font-1.0.txt`](epaint_default_fonts-Ubuntu-font-1.0.txt) | https://github.com/emilk/egui/tree/main/crates/epaint_default_fonts | Ubuntu Light font asset pulled transitively through mujoco-rs.                                      |
| unitree_sdk2 IDL types   | [`unitree_sdk2-BSD-3-Clause.txt`](unitree_sdk2-BSD-3-Clause.txt)   | https://github.com/unitreerobotics/unitree_sdk2                | Source of the IDL message shapes ported into `crates/unitree-hg-idl/`.                              |
| NVIDIA Open Model License| [`nvidia-open-model-license.txt`](nvidia-open-model-license.txt)   | https://developer.download.nvidia.com/licenses/                | Governs the GEAR-SONIC weights — fetched at runtime from HuggingFace, **never bundled in this repo**. |

## Adding a new dependency

When you add a third-party dependency in a PR:

1. Identify the SPDX expression from the upstream `Cargo.toml` /
   `pyproject.toml` / repo. If the expression is `A OR B`, add **one
   file per branch** (e.g. both `foo-Apache-2.0.txt` and `foo-MIT.txt`).
2. Add the per-component file under `LICENSES/<component>-<SPDX>.txt`
   following the existing template — header (component, upstream, SPDX,
   used-by) plus a pointer to the canonical license text in this
   directory. Include the upstream copyright notice when one is
   published.
3. If the SPDX expression is not already represented (no
   `LICENSES/<SPDX>.txt`), add the canonical text for it.
4. Add a row to the index table above.
5. Update the allowlist in [`../deny.toml`](../deny.toml) if the new
   SPDX identifier is not already accepted.
6. If the new dependency is **strong-copyleft** (GPL-3.0-only,
   AGPL-3.0-only, etc.), pause and consult a human reviewer — robowbc
   currently rejects MIT-incompatible licenses by policy.

## Licensing posture summary

* robowbc Rust code: MIT (permissive).
* Permissive transitive deps (Apache-2.0, MIT, BSD-3-Clause, MPL-2.0,
  ISC, Unicode-3.0, Zlib): accepted.
* Weak copyleft (EPL-2.0, MPL-2.0, LGPL-3.0): accepted with attribution.
  CycloneDDS is the most prominent example; dual-licensed under
  `EPL-2.0 OR BSD-3-Clause`, redistributors may pick the BSD branch.
* Font licenses (OFL-1.1, Ubuntu-font-1.0): accepted for bundled UI font
  assets with attribution and license text preservation.
* Strong copyleft (GPL, AGPL): rejected by CI.
* Model licenses (NVIDIA Open Model License, etc.): governed
  per-bundle; weights are fetched at runtime, not redistributed.

See [`../docs/third-party-notices.md`](../docs/third-party-notices.md)
for the user-facing summary.
