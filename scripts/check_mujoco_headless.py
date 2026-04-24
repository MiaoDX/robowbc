#!/usr/bin/env python3
"""Fail fast when MuJoCo headless rendering is not available.

This is the same offscreen rendering path used by proof-pack screenshot
capture. `make showcase-verify` runs this first so local developer checks and
the GitHub Actions showcase job fail on the same prerequisite instead of
shipping a site bundle with skipped screenshots.
"""

from __future__ import annotations

import os
import sys

from roboharness_report import ensure_headless_mujoco_env, is_headless_render_backend_error

PROBE_XML = """
<mujoco model="headless_probe">
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body name="probe_box" pos="0 0 0.1">
      <joint type="free"/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.85 0.45 0.25 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def render_probe() -> tuple[str, tuple[int, ...]]:
    ensure_headless_mujoco_env()

    import mujoco

    backend = os.environ.get("MUJOCO_GL", "auto")
    model = mujoco.MjModel.from_xml_string(PROBE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frame = renderer.render()
    finally:
        close = getattr(renderer, "close", None)
        if callable(close):
            close()
    return backend, tuple(int(dimension) for dimension in frame.shape)


def package_hint() -> str:
    if sys.platform != "linux":
        return ""
    return (
        "Install the EGL/Mesa runtime used by CI before retrying, for example:\n"
        "  sudo apt-get install -y libegl1 libegl-mesa0 libgles2 libgl1-mesa-dri libgbm1"
    )


def main() -> int:
    try:
        backend, frame_shape = render_probe()
    except Exception as exc:
        backend = os.environ.get("MUJOCO_GL", "auto")
        if is_headless_render_backend_error(exc):
            hint = package_hint()
            raise SystemExit(
                "MuJoCo headless render smoke check failed for the configured "
                f"backend ({backend}): {type(exc).__name__}: {exc}\n"
                "This blocks proof-pack screenshot capture, so "
                "`make showcase-verify` refuses to continue.\n"
                f"{hint}".rstrip()
            ) from exc
        raise

    print(
        "MuJoCo headless render smoke check passed: "
        f"backend={backend} frame_shape={frame_shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
