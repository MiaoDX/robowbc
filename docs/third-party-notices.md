# Third-party notices

robowbc itself is released under the **MIT License** (see the root
[`LICENSE`](../LICENSE)). It builds on a stack of third-party software
and, at runtime, may load model weights distributed under licenses
that are not the same as robowbc's.

This page summarizes those licenses for end users — what they require,
what changes (if anything) when you redistribute robowbc, and where to
find the verbatim license text. The authoritative per-component files
live in [`LICENSES/`](../LICENSES/); this page is the human-readable
overview, not a substitute.

## TL;DR

| If you are…                                              | What you need to do                                                                                       |
|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| using robowbc on your own machine                        | Nothing. The MIT license and the third-party licenses all permit private use without further obligations. |
| building a binary against robowbc and shipping it       | Include `LICENSES/` in your distribution. Keep upstream copyright notices intact.                          |
| baking GEAR-SONIC weights into a Docker image            | **Don't.** The robowbc Docker image deliberately fetches weights at runtime — see [issue #131](https://github.com/MiaoDX/robowbc/issues/131). If you do bundle them yourself, you must include the verbatim NVIDIA Open Model License alongside the weights and accept its restrictions. |
| adding a new third-party dependency in a PR              | Add the matching `LICENSES/<component>-<SPDX>.txt`, update [`LICENSES/README.md`](../LICENSES/README.md), and let the license CI gate validate the SPDX expression against the allowlist. |
| writing GPL/AGPL-licensed code that depends on robowbc | Allowed — MIT is one-way compatible into copyleft. The reverse (pulling GPL code into robowbc) is not. |

## License categories

robowbc's third-party stack splits into three categories:

### 1. Permissive software licenses

Apache-2.0, MIT, BSD-3-Clause. Free use, modification, and
redistribution as long as you keep the copyright notice and license
text. No source-disclosure obligation. This covers most of the Rust
ecosystem we depend on:

* **ort** (`Apache-2.0 OR MIT`) — ONNX Runtime Rust bindings; chosen
  for CUDA / TensorRT execution-provider support.
* **rerun** (`Apache-2.0 OR MIT`) — visualization SDK + viewer.
* **crossterm** (`MIT`) — terminal input for keyboard teleop.
* **unitree_sdk2 IDL types** (`BSD-3-Clause`) — robowbc's
  `crates/unitree-hg-idl/` is a Rust port of the IDL message shapes
  and CRC32 helper from the upstream Unitree SDK.

### 2. Weak copyleft / file-level copyleft

Eclipse Public License v2.0 and Mozilla Public License v2.0 require
that *modifications to the licensed files themselves* be made
available under the same license. Combining them with permissively
licensed code is fine — the obligation only attaches to changed files.

* **CycloneDDS** (`EPL-2.0 OR BSD-3-Clause`) — vendored via cyclors as
  the wire protocol used to talk to `unitree_mujoco` and the real G1.
  Redistributors may select either branch of the dual license.
  robowbc redistributes CycloneDDS unmodified, so the EPL-2.0
  source-disclosure obligation is satisfied by linking back to
  upstream.

### 3. Model licenses

These govern *trained model weights*, not source code. They are
distinct from any of robowbc's software licenses and have their own
restrictions.

* **NVIDIA Open Model License** — covers GEAR-SONIC weights
  (`model_encoder.onnx`, `model_decoder.onnx`, `planner_sonic.onnx`).
  robowbc **never bundles these weights**. Users fetch them on first
  run via `scripts/download_gear_sonic_models.sh` from HuggingFace and
  accept the license at fetch time. If you build a derived
  distribution that bundles the weights, you take on the redistribution
  obligations of the NVIDIA Open Model License — see
  [`LICENSES/nvidia-open-model-license.txt`](../LICENSES/nvidia-open-model-license.txt)
  for the pointer to the authoritative text.

  Other policies (`decoupled_wbc`, `wbc_agile`, `bfm_zero`) similarly
  ship weights under their own licenses; check each policy's HuggingFace
  card before redistribution.

## Strong copyleft (GPL / AGPL): not accepted

robowbc's CI rejects PRs that introduce GPL or AGPL transitive
dependencies. The MIT license is one-way compatible into GPL — you can
embed robowbc in a GPL project — but pulling GPL code into robowbc
itself would force a license change. We don't take that on without an
explicit decision.

The allowlist enforced by `.github/workflows/license.yml` is the
machine-checked record of which SPDX identifiers are accepted; the
human-readable list is in [`LICENSES/README.md`](../LICENSES/README.md).

## Redistribution checklist

When you redistribute robowbc (binary, container image, or static
bundle) you should include:

1. The full [`LICENSES/`](../LICENSES/) directory, unmodified.
2. The root [`LICENSE`](../LICENSE) file (robowbc's own MIT license).
3. A `NOTICE` or equivalent that points the recipient at the above and
   names the third-party components used.
4. If you bundle model weights: the verbatim license file for those
   weights, alongside the weights themselves.

If you generate the bundle automatically and want a tool to do this,
the `cargo about generate` and `cargo deny check licenses` workflows
referenced in `.github/workflows/license.yml` are good starting points
— the same machinery that gates incoming PRs can produce a
distribution-ready notice file.
