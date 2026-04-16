---
license: other
license_name: nvidia-open-model-license
license_link: LICENSE
tags:
- robotics
- humanoid
- whole-body-control
- reinforcement-learning
- motion-tracking
- teleoperation
- pytorch
- isaac-lab
pipeline_tag: reinforcement-learning
---

# GEAR-SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control

<div align="center">
  <img src="sonic-preview-gif-480P.gif" width="800">
</div>

## Model Description

**SONIC** (Supersizing Motion Tracking) is a humanoid behavior foundation model developed by NVIDIA that gives robots a core set of motor skills learned from large-scale human motion data. Rather than building separate controllers for predefined motions, SONIC uses motion tracking as a scalable training task, enabling a single unified policy to produce natural, whole-body movement and support a wide range of behaviors.

### Key Features

- 🤖 **Unified Whole-Body Control**: Single policy handles walking, running, crawling, jumping, manipulation, and more
- 🎯 **Motion Tracking**: Trained on large-scale human motion data for natural movements
- 🎮 **Real-Time Teleoperation**: VR-based whole-body teleoperation via PICO headset
- 🚀 **Hardware Deployment**: C++ inference stack for real-time control on humanoid robots
- 🎨 **Kinematic Planner**: Real-time locomotion generation with multiple movement styles
- 🔄 **Multi-Modal Control**: Supports keyboard, gamepad, VR, and high-level planning

## VR Whole-Body Teleoperation

SONIC supports real-time whole-body teleoperation via PICO VR headset, enabling natural human-to-robot motion transfer for data collection and interactive control.

<div align="center">
<table>
<tr>
<td align="center"><b>Walking</b></td>
<td align="center"><b>Running</b></td>
</tr>
<tr>
<td align="center"><img src="media/teleop_walking.gif" width="400"></td>
<td align="center"><img src="media/teleop_running.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Sideways Movement</b></td>
<td align="center"><b>Kneeling</b></td>
</tr>
<tr>
<td align="center"><img src="media/teleop_sideways.gif" width="400"></td>
<td align="center"><img src="media/teleop_kneeling.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Getting Up</b></td>
<td align="center"><b>Jumping</b></td>
</tr>
<tr>
<td align="center"><img src="media/teleop_getup.gif" width="400"></td>
<td align="center"><img src="media/teleop_jumping.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Bimanual Manipulation</b></td>
<td align="center"><b>Object Hand-off</b></td>
</tr>
<tr>
<td align="center"><img src="media/teleop_bimanual.gif" width="400"></td>
<td align="center"><img src="media/teleop_switch_hands.gif" width="400"></td>
</tr>
</table>
</div>

## Kinematic Planner

SONIC includes a kinematic planner for real-time locomotion generation — choose a movement style, steer with keyboard/gamepad, and adjust speed and height on the fly.

<div align="center">
<table>
<tr>
<td align="center" colspan="2"><b>In-the-Wild Navigation</b></td>
</tr>
<tr>
<td align="center" colspan="2"><img src="media/planner/planner_in_the_wild_navigation.gif" width="800"></td>
</tr>
<tr>
<td align="center"><b>Run</b></td>
<td align="center"><b>Happy</b></td>
</tr>
<tr>
<td align="center"><img src="media/planner/planner_run.gif" width="400"></td>
<td align="center"><img src="media/planner/planner_happy.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Stealth</b></td>
<td align="center"><b>Injured</b></td>
</tr>
<tr>
<td align="center"><img src="media/planner/planner_stealth.gif" width="400"></td>
<td align="center"><img src="media/planner/planner_injured.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Kneeling</b></td>
<td align="center"><b>Hand Crawling</b></td>
</tr>
<tr>
<td align="center"><img src="media/planner/planner_kneeling.gif" width="400"></td>
<td align="center"><img src="media/planner/planner_hand_crawling.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Elbow Crawling</b></td>
<td align="center"><b>Boxing</b></td>
</tr>
<tr>
<td align="center"><img src="media/planner/planner_elbow_crawling.gif" width="400"></td>
<td align="center"><img src="media/planner/planner_boxing.gif" width="400"></td>
</tr>
</table>
</div>

## Quick Start

📚 **See the [Quick Start Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html)** for step-by-step instructions on:
- Installation and setup
- Running SONIC with different control modes (keyboard, gamepad, VR)
- Deploying on real hardware
- Using the kinematic planner

**Key Resources:**
- [Installation Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html) - Complete setup instructions
- [Keyboard Control Tutorial](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/keyboard.html) - Get started with keyboard control
- [Gamepad Control Tutorial](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/gamepad.html) - Set up gamepad control
- [VR Teleoperation Setup](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html) - Full-body VR control

## Model Checkpoints

All checkpoints (ONNX format) are available directly in this repository. Inference is powered by TensorRT and runs on both desktop and Jetson hardware.

| Checkpoint | File | Description |
|---|---|---|
| Policy encoder | `model_encoder.onnx` | Encodes motion reference into latent |
| Policy decoder | `model_decoder.onnx` | Decodes latent into joint actions |
| Kinematic planner | `planner_sonic.onnx` | Real-time locomotion style planner |

**Quick download** (requires `pip install huggingface_hub`):

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="nvidia/GEAR-SONIC", local_dir="gear_sonic_deploy")
```

Or use the download script from the GitHub repo:

```bash
python download_from_hf.py             # policy + planner (default)
python download_from_hf.py --no-planner # policy only
```

See the [Download Models guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html) for full instructions.

## Documentation

📚 **[Full Documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/)**

### Guides
- [Installation (Deployment)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)
- [Installation (Training)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html)
- [Quick Start](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html)
- [VR Teleoperation Setup](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)

### Tutorials
- [Keyboard Control](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/keyboard.html)
- [Gamepad Control](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/gamepad.html)
- [VR Whole-Body Teleoperation](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html)

## Repository Structure

```
GR00T-WholeBodyControl/
├── gear_sonic_deploy/     # C++ inference stack for deployment
├── gear_sonic/            # Teleoperation and data collection tools
├── decoupled_wbc/         # Decoupled WBC (GR00T N1.5/N1.6)
├── docs/                  # Documentation source
└── media/                 # Videos and images
```

## Related Projects

This repository is part of NVIDIA's GR00T (Generalist Robot 00 Technology) initiative:
- **[GR00T N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/)**: Previous generation decoupled controller
- **[GR00T N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/)**: Improved decoupled WBC approach
- **[GEAR-SONIC Website](https://nvlabs.github.io/GEAR-SONIC/)**: Project page with videos and details

## Citation

If you use GEAR-SONIC in your research, please cite:

```bibtex
@article{luo2025sonic,
    title={SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control},
    author={Luo, Zhengyi and Yuan, Ye and Wang, Tingwu and Li, Chenran and Chen, Sirui and Casta\~neda, Fernando and Cao, Zi-Ang and Li, Jiefeng and Minor, David and Ben, Qingwei and Da, Xingye and Ding, Runyu and Hogg, Cyrus and Song, Lina and Lim, Edy and Jeong, Eugene and He, Tairan and Xue, Haoru and Xiao, Wenli and Wang, Zi and Yuen, Simon and Kautz, Jan and Chang, Yan and Iqbal, Umar and Fan, Linxi and Zhu, Yuke},
    journal={arXiv preprint arXiv:2511.07820},
    year={2025}
}
```

## License

This project uses **dual licensing**:

- **Source Code**: Apache License 2.0 - applies to all code, scripts, and software components
- **Model Weights**: NVIDIA Open Model License - applies to all trained model checkpoints

**Key points of the NVIDIA Open Model License:**
- ✅ Commercial use permitted with attribution
- ✅ Modification and distribution allowed
- ⚠️ Must comply with NVIDIA's Trustworthy AI terms
- ⚠️ Model outputs subject to responsible use guidelines

See [LICENSE](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/LICENSE) for complete terms.

## Support & Contact

- 📧 **Email**: [gear-wbc@nvidia.com](mailto:gear-wbc@nvidia.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/NVlabs/GR00T-WholeBodyControl/issues)
- 📖 **Documentation**: [https://nvlabs.github.io/GR00T-WholeBodyControl/](https://nvlabs.github.io/GR00T-WholeBodyControl/)
- 🌐 **Website**: [https://nvlabs.github.io/GEAR-SONIC/](https://nvlabs.github.io/GEAR-SONIC/)

## Acknowledgments

This work builds upon and acknowledges:
- [Beyond Mimic](https://github.com/HybridRobotics/whole_body_tracking) - Whole-body tracking foundation
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - Robot learning framework
- NVIDIA Research GEAR Lab team
- All contributors and collaborators

## Model Card Contact

For questions about this model card or responsible AI considerations, contact: [gear-wbc@nvidia.com](mailto:gear-wbc@nvidia.com)