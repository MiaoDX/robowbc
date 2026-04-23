# Upstream Helper Patches

No helper patches are checked in yet.

The current official wrapper is intentionally conservative:

- it records the pinned upstream commit
- it emits normalized blocked artifacts when the upstream stack does not expose
  a fair, non-interactive benchmark seam in the current environment

If a future comparison needs a small upstream patch to expose a matched path,
drop it in this directory and reference the patch name from the wrapper notes.
