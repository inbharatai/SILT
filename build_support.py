"""Small setuptools hook: snapshot exact source audit inputs into each wheel.

No generated copies are checked into src/. The sdist includes this hook and its
original inputs so rebuilding an extracted sdist is independent of the checkout.
"""
import hashlib
import json
from pathlib import Path
import runpy
import shutil

from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        origins = runpy.run_path(str(root / "src/asea/_package_resources.py"))["AUDIT_ORIGINS"]
        target = Path(self.build_lib) / "asea/_audit_support"
        # Incremental builds must not retain removed or stale support inputs.
        if target.exists():
            shutil.rmtree(target)
        files = {}
        for origin in origins:
            source = root / origin
            if source.is_symlink() or not source.is_file():
                raise ValueError("required source audit input missing: " + str(source))
            raw = source.read_bytes()
            output = target / origin
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(raw)
            output.chmod(0o644)
            files[origin] = {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
        manifest = target / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1,
            "provenance": "exact build-time repository-relative source bytes",
            "files": files}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest.chmod(0o644)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        target = Path(self.build_lib) / "asea/_audit_support"
        return outputs + [str(p) for p in sorted(target.rglob("*")) if p.is_file()]
