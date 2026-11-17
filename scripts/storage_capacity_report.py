#!/usr/bin/env python3
"""Emit one deterministic, read-only APFS capacity report as JSON.

The reporter invokes only ``diskutil info -plist /`` and does not traverse,
read, or modify project files. It has no cleanup or scheduling capability.
"""

from __future__ import annotations

import json
import plistlib
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

DISKUTIL_COMMAND: Final = ("/usr/sbin/diskutil", "info", "-plist", "/")
COMMAND_TIMEOUT_SECONDS: Final = 5.0
GIB: Final = 1024**3


class TelemetryError(RuntimeError):
    """Raised when exact APFS container telemetry cannot be obtained."""


@dataclass(frozen=True)
class CapacitySample:
    """Validated integer APFS container measurements."""

    total_bytes: int
    available_bytes: int

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.available_bytes


def _required_integer(data: dict[object, object], key: str) -> int:
    if key not in data:
        raise TelemetryError(f"missing {key} measurement")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TelemetryError(f"{key} must be an integer byte measurement")
    return value


def parse_apfs_telemetry(payload: bytes) -> CapacitySample:
    """Parse and validate ``diskutil info -plist /`` output."""

    try:
        data = plistlib.loads(payload)
    except (plistlib.InvalidFileException, ValueError, TypeError) as exc:
        raise TelemetryError("invalid diskutil plist output") from exc

    if not isinstance(data, dict):
        raise TelemetryError("invalid diskutil plist output: root must be a dictionary")

    total_bytes = _required_integer(data, "APFSContainerSize")
    available_bytes = _required_integer(data, "APFSContainerFree")

    if total_bytes <= 0:
        raise TelemetryError("APFSContainerSize must be greater than zero")
    if available_bytes < 0:
        raise TelemetryError("APFSContainerFree must be non-negative")
    if available_bytes > total_bytes:
        raise TelemetryError("APFSContainerFree cannot exceed APFSContainerSize")

    return CapacitySample(total_bytes=total_bytes, available_bytes=available_bytes)


def collect_apfs_telemetry(
    *, timeout_seconds: float = COMMAND_TIMEOUT_SECONDS
) -> CapacitySample:
    """Run the single approved read-only command and return validated telemetry."""

    try:
        result = subprocess.run(
            DISKUTIL_COMMAND,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        formatted_timeout = f"{timeout_seconds:g}"
        raise TelemetryError(
            f"diskutil timed out after {formatted_timeout} seconds"
        ) from exc
    except OSError as exc:
        raise TelemetryError(f"diskutil could not be executed: {exc.strerror}") from exc

    if result.returncode != 0:
        raise TelemetryError(f"diskutil exited with status {result.returncode}")

    return parse_apfs_telemetry(result.stdout)


def classify_severity(available_bytes: int) -> str:
    """Classify exact available bytes at the approved binary-GiB boundaries."""

    if available_bytes < 2 * GIB:
        return "EMERGENCY"
    if available_bytes < 15 * GIB:
        return "CRITICAL"
    if available_bytes < 30 * GIB:
        return "WARNING"
    if available_bytes < 50 * GIB:
        return "ADVISORY"
    return "NORMAL"


def _format_utc(timestamp: datetime) -> str:
    if timestamp.tzinfo is None:
        raise ValueError("observed_utc must be timezone-aware")
    return timestamp.astimezone(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def build_report(
    sample: CapacitySample, *, observed_utc: datetime | None = None
) -> dict[str, object]:
    """Build the stable capacity-mode JSON document."""

    observed = observed_utc or datetime.now(UTC)
    available_percent = round(
        sample.available_bytes * 100 / sample.total_bytes,
        6,
    )
    return {
        "schema_version": "1.0",
        "mode": "capacity",
        "observed_utc": _format_utc(observed),
        "apfs_container_total_bytes": sample.total_bytes,
        "apfs_container_free_bytes": sample.available_bytes,
        "apfs_container_used_bytes": sample.used_bytes,
        "available_percent": available_percent,
        "severity": classify_severity(sample.available_bytes),
    }


def render_json(report: dict[str, object]) -> str:
    """Serialize with stable key ordering and compact separators."""

    return json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"


def _error_report(message: str, *, observed_utc: datetime) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "mode": "capacity",
        "observed_utc": _format_utc(observed_utc),
        "apfs_container_total_bytes": None,
        "apfs_container_free_bytes": None,
        "apfs_container_used_bytes": None,
        "available_percent": None,
        "severity": "UNKNOWN",
        "errors": [message],
    }


def main() -> int:
    observed = datetime.now(UTC)
    try:
        sample = collect_apfs_telemetry()
    except TelemetryError as exc:
        sys.stdout.write(render_json(_error_report(str(exc), observed_utc=observed)))
        return 1

    sys.stdout.write(render_json(build_report(sample, observed_utc=observed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
