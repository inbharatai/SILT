#!/usr/bin/env bash
# E2 GPU training environment (isolated, honest deviations recorded).
#
# The RTX 5050 is sm_120 (Blackwell): only cu128 torch builds run on it, and
# the production localmodels pin (torch>=2.6,<2.7) has no sm_120 build. The
# system Windows Python already carries torch 2.11.0+cu128 with working CUDA
# (verified by a real matmul). This script therefore creates a Windows venv
# WITH --system-site-packages (inheriting that torch) and installs the
# production-pinned transformers/peft/accelerate INTO the venv so they shadow
# the system's 5.2.0/0.20.0/1.14.0. The torch deviation is isolated to this
# environment and recorded in the E2 training receipt; the repo pins are not
# touched (same pattern as the GLM worker's isolated transformers 5.16.1).
set -euo pipefail
VENV="$HOME/siltgpu"

python -m venv --system-site-packages "$VENV"
"$VENV/Scripts/python" -m pip install --quiet \
  "transformers==4.51.3" "peft==0.15.2" "accelerate==1.10.1"

"$VENV/Scripts/python" - <<'PY'
import json, sys
import torch
import transformers
import peft
import accelerate
import asea

out = {
    "python": sys.version.split()[0],
    "asea_from": asea.__file__,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
}
assert transformers.__version__ == "4.51.3", "venv did not shadow transformers"
assert peft.__version__ == "0.15.2", "venv did not shadow peft"
assert "cu128" in torch.__version__, "expected the inherited cu128 torch build"
assert torch.cuda.is_available(), "CUDA must be visible in the GPU venv"
assert "Desktop" in asea.__file__ and "SILT" in asea.__file__, \
    "the venv must resolve the repo's asea, not a foreign install"
if torch.cuda.is_available():
    # REAL CUDA work (not a version string): a matmul on the device.
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = (a @ b).sum().item()
    assert c == c  # finite
    out["gpu"] = torch.cuda.get_device_name(0)
    out["compute_capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
print(json.dumps(out, indent=2))
PY
echo "GPU VENV READY: $VENV"