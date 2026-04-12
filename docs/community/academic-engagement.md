# RoboWBC — Academic Community Engagement

_Covers: ICRA 2026, WBC survey (arXiv:2506.20487), and citation strategy._
_Update this file and the status table below as items are completed._

---

## Status

| Item | Status | Link / Notes |
|------|--------|--------------|
| WBC survey read + robowbc placement identified | [ ] | arXiv:2506.20487 |
| Survey companion repo PR submitted | [ ] | — |
| Survey authors contacted | [ ] | — |
| ICRA 2026 Sim-to-Real workshop checked | [ ] | Jun 19-25, Vienna |
| ICRA 2026 AgiBot World Competition WBC Track checked | [ ] | Jun 19-25, Vienna |
| Workshop/demo paper submitted | [ ] | Requires GEAR-SONIC demo (#37) |
| README Related Work cites survey | [x] | Already present |

---

## WBC Survey (arXiv:2506.20487)

The survey "A Comprehensive Survey on Whole-Body Control for Humanoid Robots" catalogues the WBC landscape across training, control architectures, and deployment. It is the canonical reference list read by every researcher entering the WBC space.

### Where robowbc fits

The survey covers policy training and evaluation extensively but deployment tooling (the gap robowbc fills) is under-represented. Robowbc belongs in a "Deployment / Inference Runtime" section:

- **Category:** Deployment runtimes / inference frameworks for WBC policies
- **Distinction:** robowbc is not a policy; it is a unified runtime that runs any surveyed policy (GEAR-SONIC, HOVER, BFM-Zero, WBC-AGILE, WholeBodyVLA) through one trait interface, one config swap, and one binary
- **Complementary framing:** every policy in the survey needs a deployment runtime; robowbc provides a common one instead of each team writing their own C++ stack

### Companion repo PR template

Check whether the survey has a companion GitHub repo (search for "awesome-whole-body-control", "WBC-survey", or the lead author's GitHub). If found, open a PR:

**PR title:**
```
Add robowbc — unified WBC inference runtime
```

**Entry line (to paste into the relevant section):**
```markdown
- [robowbc](https://github.com/MiaoDX/robowbc) — Unified WBC inference runtime (Rust + ONNX Runtime). Runs GEAR-SONIC, BFM-Zero, HOVER, WholeBodyVLA, and WBC-AGILE through one `WbcPolicy` trait; config-driven model switching via TOML.
```

**PR body:**
```markdown
## Add robowbc — unified WBC inference runtime

robowbc is a Rust inference runtime that provides a common deployment interface for the WBC policies covered in arXiv:2506.20487 (GEAR-SONIC, BFM-Zero, HOVER, WholeBodyVLA, WBC-AGILE, and more).

The core abstraction: a single `WbcPolicy` trait with `predict(observation) → joint_targets`. Any policy implementing this trait can be loaded by config name alone — no code changes required to switch from GEAR-SONIC to BFM-Zero or from G1 to H1.

This closes the deployment gap your survey identifies: 30+ papers in 2025 each built their own C++ deployment stack. robowbc aims to be the common layer they don't have to reinvent.

Repository: https://github.com/MiaoDX/robowbc
```

### Contact template for survey authors

Use GitHub (open an issue or discussion in their repo) rather than email where possible. Be specific and technical; lead with the deployment gap angle.

**GitHub issue/discussion title:**
```
robowbc — unified deployment runtime for the WBC policies you survey
```

**Message body:**
```
Hi,

I've been following your WBC survey (arXiv:2506.20487) closely — it's become the canonical reference for the field.

One gap I noticed: the survey covers training and evaluation but the deployment layer is under-represented. We're building robowbc (https://github.com/MiaoDX/robowbc) specifically to address this: a Rust inference runtime with one `WbcPolicy` trait that runs GEAR-SONIC, BFM-Zero, HOVER, WholeBodyVLA, and WBC-AGILE through a single config-driven interface.

The core observation that motivated it: every team in your survey built their own C++ deployment stack. robowbc aims to be the standard layer they don't have to rebuild.

Would you consider adding a "Deployment Runtimes" section to the survey or companion repo? Happy to contribute a write-up. I'd also appreciate feedback on whether we're framing the contribution correctly relative to your taxonomy.

Best,
[Your name]
```

**Tone note:** Approach as a contributor, not a promoter. Lead with the gap identification. Mention the survey paper with the arXiv ID to show you've read it.

---

## ICRA 2026 (June 19-25, Vienna)

ICRA 2026 has two high-value touchpoints for robowbc: the "Sim-to-Real Transfer for Humanoid Robots" workshop and the "AgiBot World Competition WBC Track."

### Checklist before Vienna

- [ ] Verify workshop CFP deadlines (typically 4-6 weeks before the conference, i.e., early-mid May 2026)
- [ ] Check AgiBot World Competition registration/submission page
- [ ] Confirm robowbc status: is GEAR-SONIC real model inference working? (Required for a credible demo paper)
- [ ] Decide: demo paper vs. attend-and-network

### Workshop demo paper (if GEAR-SONIC is ready)

A workshop demo paper does not require novel research — only a working system. The pitch:

**Paper title:**
```
RoboWBC: A Unified Inference Runtime for Humanoid Whole-Body Control Policies
```

**Abstract sketch (250 words):**
```
Humanoid whole-body control (WBC) research has produced a rich set of policies
(GEAR-SONIC, BFM-Zero, HOVER, WholeBodyVLA, WBC-AGILE) but each comes with a
bespoke C++ or Python deployment stack. In 2025 alone, 30+ papers using Unitree
G1/H1 each reimplemented the same WBC-to-joint-target pipeline.

We present RoboWBC, an open-source inference runtime that unifies WBC deployment
under one Rust library. The core abstraction is a single `WbcPolicy` trait
(predict(observation) → joint_targets at 50 Hz). Any ONNX-exported WBC policy
can be loaded by config name — switching from GEAR-SONIC to BFM-Zero requires
changing one line in a TOML file, not modifying deployment code.

RoboWBC supports ONNX Runtime (CUDA + TensorRT) and PyTorch-via-PyO3 inference
backends, a zenoh communication layer for Unitree G1/H1 hardware, and an
inventory-based policy registry. We benchmark RoboWBC's GEAR-SONIC inference
against NVIDIA's reference C++ deployment and show [X]% lower latency with
comparable accuracy.

We believe a unified deployment layer is as important for the WBC community as
LeRobot has been for manipulation: standardized interfaces reduce duplicated
engineering effort and enable fair cross-policy benchmarking.

Code: https://github.com/MiaoDX/robowbc
```

**Venues to target:**
1. "Sim-to-Real Transfer for Humanoid Robots" workshop (most directly relevant)
2. "AgiBot World Competition WBC Track" (if competitive demo track exists)

### Networking targets at ICRA 2026

People to identify and approach:
- WBC survey authors (in-person follow-up to the contact above)
- NVIDIA WBC team (GEAR-SONIC, HOVER, WBC-AGILE authors)
- CMU LeCAR Lab (BFM-Zero, SoFTA authors)
- OpenDriveLab (WholeBodyVLA authors)
- Unitree research contacts

Conversation starter: "We built a unified inference runtime for WBC policies — essentially a LeRobot-equivalent for the WBC deployment layer. It runs your [paper]'s policy through the same interface as GEAR-SONIC. Can I show you a quick demo?"

---

## Citation strategy

### What's already done

- README Related Work section cites arXiv:2506.20487 and all major WBC papers robowbc wraps ✓
- README positions robowbc as complementary ("designed to provide a common deployment interface across these lines of work") ✓

### What remains

1. **When the roboharness category-defining article is drafted:** Include a paragraph describing the robowbc + roboharness stack (robowbc runs WBC inference → MuJoCo steps physics → roboharness captures visual regression). This cross-mentions both projects in a single narrative.

2. **After first community paper/demo is published:** Add a `CITATION.cff` file so researchers who use robowbc can cite it properly:

   ```yaml
   cff-version: 1.2.0
   message: "If you use RoboWBC in your research, please cite it as below."
   type: software
   title: "RoboWBC: Unified Inference Runtime for Humanoid Whole-Body Control"
   repository-code: "https://github.com/MiaoDX/robowbc"
   license: Apache-2.0
   ```

3. **Future positioning note:** When papers using robowbc for deployment are published, ask authors to cite robowbc in their "Implementation" or "Deployment" section. The citation chain is: survey → robowbc → using-paper, which raises robowbc's profile over time.
