"""Local HMAC-SHA256 signing for SILT reports (B1a diff, B3 erasure cert).

A *signed report* here is a JSON dict that carries a ``signature`` field: an
HMAC-SHA256 over a canonical serialisation of the report MINUS that field, keyed
by a local 32-byte key. This is the shared mechanism; each report carries its own
``honesty_note`` stating what the signature does and does NOT prove (see the
caller modules -- :mod:`asea.capability_diff`, :mod:`asea.unlearning`).

HONESTY CONTRACT (binding, shared):

  * The signature is LOCAL. The key lives at ``<workspace>/<key_filename>``, is
    generated on first use, and is NEVER uploaded (the signing key is
    host-local; patent pending India, filed 2026-08-21). The signature proves the report was not tampered with *after
    generation, to the holder of that same local key*. It is NOT a portable
    third-party attestation and NOT proof of authorship to anyone without the
    key. That stronger property (portable asymmetric attestation) is deliberately
    NOT built (out of scope of this release).
  * ``verify`` never mints a key. The key is loaded ONCE, strictly
    (``generate=False``), and reused for both the HMAC and the fingerprint -- a
    missing key raises :class:`~asea.core.errors.SigningKeyError` (NEVER a silent
    pass), and verify never re-loads with ``generate=True`` so it cannot mint a
    fresh key as a side effect, even under a mid-verify key deletion
    (adversarial audit 2026-08-17, D1).
  * A missing OR non-string ``signature`` field is a typed
    :class:`~asea.core.errors.SignatureMismatchError`, never a bare ``TypeError``
    from ``hmac.compare_digest`` (adversarial audit 2026-08-17, A1).
  * The canonical bytes are deterministic: sorted keys, compact separators,
    ``ensure_ascii=False``. Callers pre-round floats so a re-serialisation
    cannot drift a float's repr and silently break verification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any, Dict

from .core.errors import SignatureMismatchError, SigningKeyError

#: Signature algorithm tag carried in every report so a verifier knows what it
#: is checking (and so the tag itself is covered by the signature -- it cannot
#: be swapped after signing).
SIGNATURE_ALG = "hmac-sha256-local"

#: Minimum acceptable key length (bytes). 32 = SHA-256 block-sized HMAC key.
_MIN_KEY_BYTES = 32


class LocalSigner:
    """Sign and verify a JSON report dict with a workspace-local HMAC key.

    Each caller passes its own ``key_filename`` (e.g. ``diff.key``,
    ``unlearn.key``) so the two report types do not share a key -- a leaked
    diff key cannot forge an erasure certificate, and vice versa.
    """

    def __init__(self, workspace: Path, key_filename: str) -> None:
        self.workspace = Path(workspace)
        self.key_filename = key_filename

    # -- paths ------------------------------------------------------------

    def key_path(self) -> Path:
        return self.workspace / self.key_filename

    # -- public API -------------------------------------------------------

    def sign(self, report: Dict[str, Any]) -> str:
        """Sign ``report`` (a dict WITHOUT a ``signature`` field, or with it
        ignored) and return the hex HMAC. Mints the key on first use."""
        return self._hmac(report, self._load_key(generate=True))

    def key_fingerprint(self) -> str:
        """A non-reversible short tag (sha256(key)[:16]) so a reader can tell
        which key signed a report WITHOUT the key being in the report. Used on
        the sign path (which legitimately mints the key on first use)."""
        return self._fingerprint_of(self._load_key(generate=True))

    def verify(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Verify ``report``'s HMAC against the local key.

        Returns ``{"valid": True, "signature_alg", "key_fingerprint"}`` on
        match. Raises :class:`SignatureMismatchError` on a tampered report or a
        missing/non-string signature, and :class:`SigningKeyError` if the key is
        missing/unreadable. A missing key is NEVER a silent pass.

        The key is loaded ONCE, strictly (``generate=False``) and reused for the
        fingerprint -- verify() never re-loads with ``generate=True`` (audit
        2026-08-17, D1).
        """
        key = self._load_key_strict()  # raises SigningKeyError if missing -- NO mint
        expected = self._hmac(report, key)
        actual = report.get("signature")
        if not isinstance(actual, str):
            # A missing OR non-string signature (a malformed/tampered report) is
            # a typed SignatureMismatchError, never a bare TypeError from
            # compare_digest (adversarial audit 2026-08-17, A1).
            raise SignatureMismatchError(
                "report signature field is missing or not a string"
            )
        if not hmac.compare_digest(expected, actual):
            raise SignatureMismatchError(
                "HMAC does not match contents (report edited after signing, or "
                "signed under a different key)"
            )
        return {
            "valid": True,
            "signature_alg": report.get("signature_alg"),
            "key_fingerprint": self._fingerprint_of(key),
        }

    # -- internals --------------------------------------------------------

    def _load_key(self, generate: bool) -> bytes:
        """Load the local signing key, generating it on first use if asked.

        The key is 32 random bytes. On POSIX the file is chmod 0600; on Windows
        chmod is a no-op (NTFS ACLs are the real control there) -- best-effort,
        never fatal. The key is NEVER printed, NEVER placed in a report, and
        NEVER uploaded.
        """
        path = self.key_path()
        if path.exists():
            data = path.read_bytes()
            if len(data) < _MIN_KEY_BYTES:
                raise SigningKeyError(str(path), "key file is too short (<32 bytes)")
            return data
        if not generate:
            raise SigningKeyError(str(path), "key file not found and generate=False")
        key = secrets.token_bytes(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
        # Best-effort restrictive perms (POSIX only; Windows ignores chmod).
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return key

    def _load_key_strict(self) -> bytes:
        # verify path: never generate a key to verify a report -- a report
        # signed under a vanished key is unverifiable, and silently minting a
        # new key would make every tampered report "verify" against a fresh key.
        return self._load_key(generate=False)

    def _hmac(self, report: Dict[str, Any], key: bytes) -> str:
        return hmac.new(key, self.canonical_bytes(report), hashlib.sha256).hexdigest()

    @staticmethod
    def _fingerprint_of(key: bytes) -> str:
        return hashlib.sha256(key).hexdigest()[:16]

    @staticmethod
    def canonical_bytes(report: Dict[str, Any]) -> bytes:
        """Deterministic bytes over the report MINUS the ``signature`` field.

        Sorted keys + compact separators + ``ensure_ascii=False``. Floats are
        expected to be pre-rounded by the caller (in each ``to_dict``), so a
        re-serialisation cannot drift a float's repr and silently break
        verification.
        """
        payload = {k: v for k, v in report.items() if k != "signature"}
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            default=str,
        ).encode("utf-8")