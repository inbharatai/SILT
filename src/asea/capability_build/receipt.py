"""Capability-build receipts: immutable, signed, honest run reports.

Pattern-shared with :mod:`asea.capability_diff` and :mod:`asea.unlearning`:

  * Signing uses :class:`asea._signing.LocalSigner` with this package's own
    key file (``capability_build.key``) so a leaked diff or unlearn key can
    never forge a capability-build receipt, and vice versa.
  * The honesty note is carried verbatim in every receipt: the signature is
    a LOCAL HMAC -- tamper-evidence to the holder of the same local key --
    NOT a portable third-party attestation.
  * Unmeasured metrics are materialised as the literal token
    ``NOT_MEASURED`` before signing; they are never estimated.
  * ``verify`` never mints a key; a receipt signed under a vanished key is
    unverifiable (typed error), never a silent pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from asea._signing import SIGNATURE_ALG, LocalSigner

from . import HONESTY_NOTE
from .errors import CapabilityBuildError
from .schema import CAPABILITY_RECEIPT_SCHEMA, CapabilityBuildReceipt

KEY_FILENAME = "capability_build.key"


def _signer(workspace: Path) -> LocalSigner:
    return LocalSigner(Path(workspace), KEY_FILENAME)


def sign_receipt(workspace: Path, receipt: CapabilityBuildReceipt) -> Dict[str, Any]:
    """Materialise, sign and return the receipt as a plain dict.

    The dict carries ``schema``, ``honesty_note``, ``signature_alg``,
    ``key_fingerprint`` and ``signature``; the signature covers everything
    except ``signature`` itself (LocalSigner canonical-bytes contract).
    """
    signer = _signer(workspace)
    payload = receipt.materialised()
    payload["schema"] = CAPABILITY_RECEIPT_SCHEMA
    payload["honesty_note"] = HONESTY_NOTE
    payload["signature_alg"] = SIGNATURE_ALG
    payload["key_fingerprint"] = signer.key_fingerprint()
    payload["signature"] = signer.sign(payload)
    return payload


def verify_receipt(workspace: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a signed receipt dict against the local key.

    Returns ``{"valid": True, ...}`` on match. Raises the typed signing
    errors from :mod:`asea.core.errors` on tampering or a missing key --
    never a silent pass. Also re-validates the payload against the strict
    receipt schema so a structurally-invalid report fails even with a
    valid signature over invalid bytes.
    """
    if not isinstance(report, dict):
        raise CapabilityBuildError("receipt must be a JSON object")
    if report.get("schema") != CAPABILITY_RECEIPT_SCHEMA:
        raise CapabilityBuildError(
            "receipt must carry schema '%s'" % CAPABILITY_RECEIPT_SCHEMA
        )
    signer = _signer(workspace)
    # The store adds ``artifact_sha256`` AFTER signing (content addressing);
    # it is not part of the signed bytes, so strip it before the signature
    # check and the structural re-validation.
    payload = {k: v for k, v in report.items() if k != "artifact_sha256"}
    verified = signer.verify(payload)
    # Structural re-validation of the signed payload (signature excluded).
    # The on-disk form carries literal NOT_MEASURED tokens; dematerialise
    # them back to None first -- a missing measurement is never accepted
    # as a number, and a number is never silently treated as missing.
    CapabilityBuildReceipt.model_validate(
        {
            k: v
            for k, v in CapabilityBuildReceipt.dematerialise(payload).items()
            if k != "signature"
        }
    )
    return verified


def write_receipt(workspace: Path, receipt: CapabilityBuildReceipt, path: Path) -> Path:
    """Materialise, sign, and atomically write a receipt JSON file."""
    import json

    payload = sign_receipt(workspace, receipt)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return target