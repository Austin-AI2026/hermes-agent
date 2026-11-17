"""Focused tests for the read-only macOS APFS capacity reporter."""

from __future__ import annotations

import json
import plistlib
import subprocess
from datetime import UTC, datetime

import pytest

from scripts import storage_capacity_report as reporter


GIB = 1024**3


def plist_bytes(*, total: object = 100 * GIB, free: object = 60 * GIB) -> bytes:
    values: dict[str, object] = {
        "APFSContainerSize": total,
        "APFSContainerFree": free,
    }
    return plistlib.dumps(values)


def test_parse_apfs_telemetry_uses_exact_integer_container_measurements():
    sample = reporter.parse_apfs_telemetry(
        plist_bytes(total=494_384_795_648, free=8_978_513_920)
    )

    assert sample.total_bytes == 494_384_795_648
    assert sample.available_bytes == 8_978_513_920
    assert sample.used_bytes == 485_406_281_728


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (plistlib.dumps({"APFSContainerSize": 1}), "APFSContainerFree"),
        (plistlib.dumps({"APFSContainerFree": 1}), "APFSContainerSize"),
        (plist_bytes(total="100", free=1), "APFSContainerSize"),
        (plist_bytes(total=100, free=True), "APFSContainerFree"),
        (plist_bytes(total=0, free=0), "greater than zero"),
        (plist_bytes(total=100, free=-1), "non-negative"),
        (plist_bytes(total=100, free=101), "cannot exceed"),
        (b"not a plist", "invalid diskutil plist"),
    ],
)
def test_parse_apfs_telemetry_rejects_missing_or_invalid_measurements(payload, message):
    with pytest.raises(reporter.TelemetryError, match=message):
        reporter.parse_apfs_telemetry(payload)


@pytest.mark.parametrize(
    ("available_bytes", "expected"),
    [
        (50 * GIB, "NORMAL"),
        (50 * GIB - 1, "ADVISORY"),
        (30 * GIB, "ADVISORY"),
        (30 * GIB - 1, "WARNING"),
        (15 * GIB, "WARNING"),
        (15 * GIB - 1, "CRITICAL"),
        (2 * GIB, "CRITICAL"),
        (2 * GIB - 1, "EMERGENCY"),
    ],
)
def test_classify_severity_at_exact_threshold_boundaries(available_bytes, expected):
    assert reporter.classify_severity(available_bytes) == expected


def test_build_report_and_json_output_are_deterministic():
    observed = datetime(2026, 11, 17, 6, 30, 45, tzinfo=UTC)
    sample = reporter.CapacitySample(total_bytes=1000, available_bytes=125)

    report = reporter.build_report(sample, observed_utc=observed)
    rendered = reporter.render_json(report)

    assert report == {
        "schema_version": "1.0",
        "mode": "capacity",
        "observed_utc": "2026-11-17T06:30:45Z",
        "apfs_container_total_bytes": 1000,
        "apfs_container_free_bytes": 125,
        "apfs_container_used_bytes": 875,
        "available_percent": 12.5,
        "severity": "EMERGENCY",
    }
    assert rendered == json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    assert json.loads(rendered) == report


def test_collect_telemetry_invokes_only_read_only_diskutil_info(monkeypatch):
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, stdout=plist_bytes(), stderr=b"")

    monkeypatch.setattr(reporter.subprocess, "run", fake_run)

    reporter.collect_apfs_telemetry(timeout_seconds=3.5)

    assert calls == [
        (
            (("/usr/sbin/diskutil", "info", "-plist", "/"),),
            {
                "check": False,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "timeout": 3.5,
            },
        )
    ]


def test_collect_telemetry_reports_command_timeout(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(reporter.subprocess, "run", fake_run)

    with pytest.raises(reporter.TelemetryError, match="timed out after 2 seconds"):
        reporter.collect_apfs_telemetry(timeout_seconds=2)


def test_error_report_renders_complete_deterministic_json():
    observed = datetime(2026, 11, 17, 6, 45, 30, tzinfo=UTC)

    rendered = reporter.render_json(
        reporter._error_report("diskutil failed", observed_utc=observed)
    )

    assert rendered == (
        '{"apfs_container_free_bytes":null,'
        '"apfs_container_total_bytes":null,'
        '"apfs_container_used_bytes":null,'
        '"available_percent":null,'
        '"errors":["diskutil failed"],'
        '"mode":"capacity",'
        '"observed_utc":"2026-11-17T06:45:30Z",'
        '"schema_version":"1.0",'
        '"severity":"UNKNOWN"}\n'
    )


def test_collect_telemetry_reports_diskutil_oserror(monkeypatch):
    def fake_run(*args, **kwargs):
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr(reporter.subprocess, "run", fake_run)

    with pytest.raises(
        reporter.TelemetryError,
        match="diskutil could not be executed: No such file or directory",
    ):
        reporter.collect_apfs_telemetry()


def test_collect_telemetry_reports_nonzero_diskutil_exit(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 7, stdout=b"", stderr=b"failed")

    monkeypatch.setattr(reporter.subprocess, "run", fake_run)

    with pytest.raises(reporter.TelemetryError, match="diskutil exited with status 7"):
        reporter.collect_apfs_telemetry()


def test_format_utc_rejects_naive_datetime():
    with pytest.raises(ValueError, match="observed_utc must be timezone-aware"):
        reporter._format_utc(datetime(2026, 11, 17, 6, 45, 30))


def test_format_utc_converts_timezone_aware_datetime_to_z_suffix():
    observed = datetime.fromisoformat("2026-11-17T01:45:30.123456-05:00")

    assert reporter._format_utc(observed) == "2026-11-17T06:45:30Z"


def test_main_success_writes_capacity_json(monkeypatch, capsys):
    observed = datetime(2026, 11, 17, 6, 45, 30, tzinfo=UTC)

    class Clock:
        @staticmethod
        def now(timezone):
            assert timezone is UTC
            return observed

    monkeypatch.setattr(reporter, "datetime", Clock)
    monkeypatch.setattr(
        reporter,
        "collect_apfs_telemetry",
        lambda: reporter.CapacitySample(total_bytes=1000, available_bytes=125),
    )

    assert reporter.main() == 0
    assert json.loads(capsys.readouterr().out) == reporter.build_report(
        reporter.CapacitySample(total_bytes=1000, available_bytes=125),
        observed_utc=observed,
    )


def test_main_failure_writes_error_json(monkeypatch, capsys):
    observed = datetime(2026, 11, 17, 6, 45, 30, tzinfo=UTC)

    class Clock:
        @staticmethod
        def now(timezone):
            assert timezone is UTC
            return observed

    def fail_collection():
        raise reporter.TelemetryError("diskutil unavailable")

    monkeypatch.setattr(reporter, "datetime", Clock)
    monkeypatch.setattr(reporter, "collect_apfs_telemetry", fail_collection)

    assert reporter.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": "1.0",
        "mode": "capacity",
        "observed_utc": "2026-11-17T06:45:30Z",
        "apfs_container_total_bytes": None,
        "apfs_container_free_bytes": None,
        "apfs_container_used_bytes": None,
        "available_percent": None,
        "severity": "UNKNOWN",
        "errors": ["diskutil unavailable"],
    }
