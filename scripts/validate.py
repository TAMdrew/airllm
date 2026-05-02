#!/usr/bin/env python3
"""AirLLM environment-aware validation script.

Detects system hardware, determines which test categories can run,
executes pytest, and generates a comprehensive validation report.

Usage:
    python scripts/validate.py [--json-only] [--skip-tests]

Flags:
    --json-only   : Print only JSON report (no terminal summary).
    --skip-tests  : Skip running pytest; report system profile only.

The script requires no dependencies beyond the project's core deps
(torch, transformers, safetensors, accelerate) and Python stdlib.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPORT_PATH = _PROJECT_ROOT / "validation_report.json"
_AIRLLM_VERSION = "unknown"

# Bytes-per-parameter estimates for model size calculations.
_BYTES_FP16 = 2
_BYTES_4BIT = 0.5

# Overhead factor — OS, framework, KV cache, activations, etc.
_MEMORY_OVERHEAD_FACTOR = 0.6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GpuInfo:
    """GPU hardware details."""

    backend: str  # "cuda", "mps", "rocm", "none"
    name: str
    vram_gb: float | None
    cuda_version: str | None = None
    driver_version: str | None = None


@dataclass
class SystemProfile:
    """Full system hardware and software profile."""

    os_name: str
    os_version: str
    architecture: str
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    gpu: GpuInfo
    python_version: str
    disk_free_gb: float
    dependency_versions: dict[str, str | None] = field(default_factory=dict)


@dataclass
class TestResult:
    """Aggregated pytest results."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    per_file: dict[str, dict[str, int]] = field(default_factory=dict)
    failure_details: list[str] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class HardwareCompatibility:
    """Assessment of what this machine can run."""

    max_model_fp16_params: str
    max_model_4bit_params: str
    recommended_backend: str
    layer_by_layer_viable: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Top-level validation report."""

    airllm_version: str
    timestamp: str
    system_profile: SystemProfile
    test_categories: dict[str, bool]
    test_results: TestResult | None
    hardware_compatibility: HardwareCompatibility
    status: str  # "PASS", "FAIL", "SKIP"


# ---------------------------------------------------------------------------
# System detection helpers
# ---------------------------------------------------------------------------


def _detect_os() -> tuple[str, str, str]:
    """Detect operating system name, version, and CPU architecture.

    Returns:
        Tuple of (os_name, os_version, architecture).
    """
    system = platform.system()
    arch = platform.machine()
    if system == "Darwin":
        os_name = "macOS"
        mac_ver = platform.mac_ver()[0]
        os_version = mac_ver if mac_ver else platform.release()
    elif system == "Linux":
        os_name = "Linux"
        try:
            # Try reading /etc/os-release for distro info.
            with open("/etc/os-release") as f:
                content = f.read()
            match = re.search(r'PRETTY_NAME="(.+?)"', content)
            os_version = match.group(1) if match else platform.release()
        except OSError:
            os_version = platform.release()
    elif system == "Windows":
        os_name = "Windows"
        os_version = platform.version()
    else:
        os_name = system
        os_version = platform.release()
    return os_name, os_version, arch


def _detect_cpu() -> tuple[str, int]:
    """Detect CPU model name and core count.

    Returns:
        Tuple of (cpu_model, cpu_cores).
    """
    cores = os.cpu_count() or 1
    cpu_model = "Unknown"

    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                cpu_model = result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":")[1].strip()
                        break
        except OSError:
            pass
    elif system == "Windows":
        cpu_model = platform.processor() or "Unknown"

    return cpu_model, cores


def _detect_ram() -> float:
    """Detect total system RAM in GB.

    Falls back to sysctl on macOS if psutil is unavailable.

    Returns:
        Total RAM in gigabytes, rounded to 1 decimal.
    """
    # Try psutil first (optional dependency).
    try:
        import psutil  # type: ignore[import-untyped]

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass

    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return round(int(result.stdout.strip()) / (1024**3), 1)
        except (subprocess.SubprocessError, ValueError):
            pass
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 1)
        except (OSError, ValueError):
            pass
    elif system == "Windows":
        try:
            # Use ctypes to query Windows memory.
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))  # type: ignore[attr-defined]
            return round(mem.ullTotalPhys / (1024**3), 1)
        except (AttributeError, OSError):
            pass

    return 0.0


def _detect_gpu() -> GpuInfo:
    """Detect GPU backend and capabilities.

    Checks (in order): NVIDIA CUDA, Apple MPS, AMD ROCm.

    Returns:
        GpuInfo with detected GPU details.
    """
    # --- NVIDIA CUDA ---
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            vram_gb = round(vram_bytes / (1024**3), 1)
            cuda_version = torch.version.cuda
            return GpuInfo(
                backend="cuda",
                name=name,
                vram_gb=vram_gb,
                cuda_version=cuda_version,
            )
    except Exception:
        pass

    # --- Apple MPS ---
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS uses unified memory — report total RAM as "shared VRAM".
            ram_gb = _detect_ram()
            return GpuInfo(
                backend="mps",
                name="Apple MPS (Metal Performance Shaders)",
                vram_gb=ram_gb,
            )
    except Exception:
        pass

    # --- AMD ROCm ---
    try:
        import torch

        if hasattr(torch, "hip") or (
            hasattr(torch.version, "hip") and torch.version.hip is not None
        ):
            name = "AMD ROCm GPU"
            try:
                name = torch.cuda.get_device_name(0)
                vram_bytes = torch.cuda.get_device_properties(0).total_mem
                vram_gb = round(vram_bytes / (1024**3), 1)
            except Exception:
                vram_gb = None
            return GpuInfo(backend="rocm", name=name, vram_gb=vram_gb)
    except Exception:
        pass

    return GpuInfo(backend="none", name="No GPU detected", vram_gb=None)


def _detect_disk_free() -> float:
    """Detect free disk space at the project root in GB.

    Returns:
        Free disk space in gigabytes, rounded to 1 decimal.
    """
    try:
        usage = shutil.disk_usage(_PROJECT_ROOT)
        return round(usage.free / (1024**3), 1)
    except OSError:
        return 0.0


def _get_package_version(package_name: str) -> str | None:
    """Attempt to import a package and return its version string.

    Args:
        package_name: The importable package name.

    Returns:
        Version string or None if the package cannot be imported.
    """
    try:
        mod = __import__(package_name)
        return getattr(mod, "__version__", getattr(mod, "version", "unknown"))
    except ImportError:
        return None


def _detect_dependencies() -> dict[str, str | None]:
    """Detect versions of key project dependencies.

    Returns:
        Dict mapping package name to version string (or None).
    """
    packages = [
        "torch",
        "transformers",
        "safetensors",
        "accelerate",
        "tqdm",
        "bitsandbytes",
    ]

    # Add mlx on macOS.
    if platform.system() == "Darwin":
        packages.extend(["mlx", "mlx_lm"])

    versions: dict[str, str | None] = {}
    for pkg in packages:
        versions[pkg] = _get_package_version(pkg)

    # PyTorch CUDA version as separate key.
    try:
        import torch

        versions["torch_cuda"] = torch.version.cuda
    except Exception:
        versions["torch_cuda"] = None

    return versions


def _get_airllm_version() -> str:
    """Read AirLLM version from the package __init__.

    Returns:
        Version string.
    """
    try:
        # Try direct import first.
        sys.path.insert(0, str(_PROJECT_ROOT / "air_llm"))
        from airllm import __version__

        return __version__
    except ImportError:
        pass

    # Fallback: parse pyproject.toml.
    pyproject = _PROJECT_ROOT / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        match = re.search(r'version\s*=\s*"(.+?)"', content)
        if match:
            return match.group(1)

    return "unknown"


# ---------------------------------------------------------------------------
# System profile assembly
# ---------------------------------------------------------------------------


def detect_system_profile() -> SystemProfile:
    """Build a complete system profile by running all detectors.

    Returns:
        Populated SystemProfile dataclass.
    """
    os_name, os_version, arch = _detect_os()
    cpu_model, cpu_cores = _detect_cpu()
    ram_gb = _detect_ram()
    gpu = _detect_gpu()
    disk_free = _detect_disk_free()
    deps = _detect_dependencies()
    python_ver = platform.python_version()

    return SystemProfile(
        os_name=os_name,
        os_version=os_version,
        architecture=arch,
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu=gpu,
        python_version=python_ver,
        disk_free_gb=disk_free,
        dependency_versions=deps,
    )


# ---------------------------------------------------------------------------
# Test category determination
# ---------------------------------------------------------------------------


def determine_test_categories(profile: SystemProfile) -> dict[str, bool]:
    """Decide which test categories can run on this system.

    Args:
        profile: Detected system profile.

    Returns:
        Dict mapping category name to runnable boolean.
    """
    has_cuda = profile.gpu.backend == "cuda"
    has_mps = profile.gpu.backend == "mps"
    is_apple_silicon = profile.os_name == "macOS" and profile.architecture in ("arm64", "aarch64")
    has_mlx = profile.dependency_versions.get("mlx") is not None

    return {
        "full_suite": True,
        "cuda_tests": has_cuda,
        "mlx_tests": is_apple_silicon and has_mlx,
        "mps_backend": has_mps,
        "compression_tests": has_cuda,
        "model_backend_tests": has_cuda or has_mps,
    }


# ---------------------------------------------------------------------------
# Pytest runner
# ---------------------------------------------------------------------------


def _parse_pytest_summary(output: str) -> dict[str, int]:
    """Parse the pytest summary line for pass/fail/skip counts.

    Handles formats like:
        "10 passed, 3 skipped, 1 failed in 2.35s"
        "254 passed in 0.53s"

    Args:
        output: Full pytest stdout.

    Returns:
        Dict with keys: passed, failed, skipped, errors.
    """
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

    # Find the summary line (last line matching the pattern).
    for line in reversed(output.splitlines()):
        if re.search(r"\d+ passed", line) or re.search(r"\d+ failed", line):
            for key in counts:
                match = re.search(rf"(\d+) {key}", line)
                if match:
                    counts[key] = int(match.group(1))
                # Also check singular "error" for "errors".
                if key == "errors":
                    match = re.search(r"(\d+) error", line)
                    if match:
                        counts["errors"] = int(match.group(1))
            break

    return counts


def _parse_per_file_results(output: str) -> dict[str, dict[str, int]]:
    """Parse per-file test results from verbose pytest output.

    Looks for lines like:
        air_llm/tests/test_utils.py::test_something PASSED

    Args:
        output: Full pytest stdout.

    Returns:
        Dict mapping filename to {passed, failed, skipped} counts.
    """
    per_file: dict[str, dict[str, int]] = {}
    pattern = re.compile(r"(air_llm/tests/\S+\.py)::\S+\s+(PASSED|FAILED|SKIPPED|ERROR)")

    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            filepath = match.group(1)
            status = match.group(2).lower()
            if filepath not in per_file:
                per_file[filepath] = {"passed": 0, "failed": 0, "skipped": 0}
            if status in per_file[filepath]:
                per_file[filepath][status] += 1

    return per_file


def _parse_duration(output: str) -> float:
    """Extract test duration from pytest output.

    Args:
        output: Full pytest stdout.

    Returns:
        Duration in seconds, or 0.0 if not found.
    """
    match = re.search(r"in (\d+\.\d+)s", output)
    return float(match.group(1)) if match else 0.0


def _extract_failure_details(output: str) -> list[str]:
    """Extract failure detail blocks from pytest output.

    Args:
        output: Full pytest stdout.

    Returns:
        List of failure description strings.
    """
    failures: list[str] = []
    in_failure = False
    current: list[str] = []

    for line in output.splitlines():
        if "FAILED" in line and "::" in line:
            failures.append(line.strip())
        elif line.startswith("FAILURES") or (line.startswith("=") and "FAILURES" in line):
            in_failure = True
        elif in_failure and (line.startswith("=") and "short test summary" in line.lower()):
            if current:
                failures.append("\n".join(current))
            in_failure = False
        elif in_failure:
            current.append(line)

    return failures


def run_tests() -> TestResult:
    """Execute the project's pytest suite and parse results.

    Returns:
        TestResult with aggregated pass/fail/skip data.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(_PROJECT_ROOT / "air_llm" / "tests"),
        "-v",
        "--tb=short",
        "--no-header",
        "-q",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=300,  # 5 minute timeout.
        )
        output = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        return TestResult(
            raw_output="ERROR: pytest timed out after 300 seconds.",
            failure_details=["Timeout: tests exceeded 5-minute limit"],
        )
    except FileNotFoundError:
        return TestResult(
            raw_output="ERROR: pytest not found. Install with: pip install pytest",
            failure_details=["pytest not installed"],
        )

    counts = _parse_pytest_summary(output)
    per_file = _parse_per_file_results(output)
    duration = _parse_duration(output)
    failures = _extract_failure_details(output)

    return TestResult(
        total=counts["passed"] + counts["failed"] + counts["skipped"] + counts["errors"],
        passed=counts["passed"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        errors=counts["errors"],
        duration_seconds=duration,
        per_file=per_file,
        failure_details=failures,
        raw_output=output,
    )


# ---------------------------------------------------------------------------
# Hardware compatibility assessment
# ---------------------------------------------------------------------------


def _format_param_count(params_billions: float) -> str:
    """Format a parameter count estimate as a human-readable string.

    Args:
        params_billions: Number of parameters in billions.

    Returns:
        Formatted string like "~70B parameters".
    """
    if params_billions >= 1.0:
        return f"~{int(params_billions)}B parameters"
    return f"~{params_billions:.1f}B parameters"


def assess_hardware(profile: SystemProfile) -> HardwareCompatibility:
    """Assess what model sizes this hardware can handle.

    Uses available memory (RAM for CPU/MPS, VRAM for CUDA) and
    bytes-per-parameter estimates to project max model sizes.

    Args:
        profile: System hardware profile.

    Returns:
        HardwareCompatibility assessment.
    """
    notes: list[str] = []

    # Determine effective memory for model loading.
    if profile.gpu.backend == "cuda" and profile.gpu.vram_gb:
        effective_mem_gb = profile.gpu.vram_gb
        notes.append(f"CUDA GPU with {effective_mem_gb}GB VRAM")
    elif profile.gpu.backend == "mps":
        # Unified memory — use total RAM but note shared nature.
        effective_mem_gb = profile.ram_gb
        notes.append(f"Apple unified memory ({effective_mem_gb}GB shared with system)")
    else:
        effective_mem_gb = profile.ram_gb
        notes.append("CPU-only: using system RAM for model loading")

    # Usable memory after overhead.
    usable_gb = effective_mem_gb * _MEMORY_OVERHEAD_FACTOR

    # Max params = usable_bytes / bytes_per_param / 1e9
    max_fp16 = usable_gb * (1024**3) / _BYTES_FP16 / 1e9
    max_4bit = usable_gb * (1024**3) / _BYTES_4BIT / 1e9

    # Determine recommended backend.
    if profile.gpu.backend == "mps":
        recommended = "MLX (Apple Silicon native)"
        if profile.dependency_versions.get("mlx") is None:
            recommended = "MPS (install mlx for best perf)"
            notes.append("Consider installing mlx: pip install 'airllm[mlx]'")
    elif profile.gpu.backend == "cuda":
        recommended = "CUDA (NVIDIA GPU)"
    elif profile.gpu.backend == "rocm":
        recommended = "ROCm (AMD GPU)"
    else:
        recommended = "CPU (no GPU acceleration available)"
        notes.append("Consider using a machine with GPU for better performance")

    # AirLLM's layer-by-layer approach makes large models viable
    # with even modest memory (4GB+).
    layer_viable = effective_mem_gb >= 4.0
    if layer_viable:
        notes.append(f"Layer-by-layer inference viable with {effective_mem_gb}GB")

    return HardwareCompatibility(
        max_model_fp16_params=_format_param_count(max_fp16),
        max_model_4bit_params=_format_param_count(max_4bit),
        recommended_backend=recommended,
        layer_by_layer_viable=layer_viable,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    profile: SystemProfile,
    categories: dict[str, bool],
    test_results: TestResult | None,
    compatibility: HardwareCompatibility,
    airllm_version: str,
) -> ValidationReport:
    """Assemble the final validation report.

    Args:
        profile: System profile data.
        categories: Test category availability map.
        test_results: Pytest results (None if tests were skipped).
        compatibility: Hardware compatibility assessment.
        airllm_version: Detected AirLLM version.

    Returns:
        Complete ValidationReport.
    """
    if test_results is None:
        status = "SKIP"
    elif test_results.failed == 0 and test_results.errors == 0:
        status = "PASS"
    else:
        status = "FAIL"

    return ValidationReport(
        airllm_version=airllm_version,
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        system_profile=profile,
        test_categories=categories,
        test_results=test_results,
        hardware_compatibility=compatibility,
        status=status,
    )


def save_report(report: ValidationReport) -> Path:
    """Serialize report to JSON and write to disk.

    Args:
        report: The validation report to save.

    Returns:
        Path to the written JSON file.
    """
    data = asdict(report)
    # Remove raw_output from JSON to keep file size reasonable.
    if data.get("test_results") and "raw_output" in data["test_results"]:
        del data["test_results"]["raw_output"]

    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return _REPORT_PATH


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

_BOX_WIDTH = 64


def _box_top() -> str:
    return "╔" + "═" * _BOX_WIDTH + "╗"


def _box_bottom() -> str:
    return "╚" + "═" * _BOX_WIDTH + "╝"


def _box_divider() -> str:
    return "╠" + "═" * _BOX_WIDTH + "╣"


def _box_line(text: str) -> str:
    """Format a single line inside the box, right-padded.

    Args:
        text: Content to display.

    Returns:
        Box-formatted line string.
    """
    # Account for emoji/unicode width — use raw padding.
    padding = _BOX_WIDTH - len(text)
    if padding < 0:
        text = text[:_BOX_WIDTH]
        padding = 0
    return "║ " + text + " " * (padding - 1) + "║"


def print_summary(report: ValidationReport) -> None:
    """Print a human-readable validation summary to the terminal.

    Args:
        report: The completed validation report.
    """
    p = report.system_profile

    print()
    print(_box_top())
    title = f"AirLLM v{report.airllm_version} Validation Report"
    title_pad = (_BOX_WIDTH - len(title)) // 2
    print("║" + " " * title_pad + title + " " * (_BOX_WIDTH - title_pad - len(title)) + "║")

    # --- System Profile ---
    print(_box_divider())
    print(_box_line("System Profile"))
    print(_box_line(f"  OS:           {p.os_name} {p.os_version} ({p.architecture})"))
    print(_box_line(f"  CPU:          {p.cpu_model} ({p.cpu_cores} cores)"))
    print(_box_line(f"  RAM:          {p.ram_gb} GB"))
    print(_box_line(f"  GPU:          {p.gpu.name}"))

    if p.gpu.backend == "mps":
        print(_box_line(f"  VRAM:         Shared ({p.gpu.vram_gb} GB unified)"))
    elif p.gpu.vram_gb:
        print(_box_line(f"  VRAM:         {p.gpu.vram_gb} GB"))
    else:
        print(_box_line("  VRAM:         N/A"))

    print(_box_line(f"  Disk Free:    {p.disk_free_gb} GB"))
    print(_box_line(f"  Python:       {p.python_version}"))

    # Key dependency versions.
    dep_keys = ["torch", "transformers", "safetensors", "accelerate"]
    if p.os_name == "macOS":
        dep_keys.append("mlx")

    for dep in dep_keys:
        ver = p.dependency_versions.get(dep, None)
        label = dep.capitalize() if dep != "mlx" else "MLX"
        if dep == "safetensors":
            label = "Safetensors"
        elif dep == "transformers":
            label = "Transformers"
        elif dep == "torch":
            label = "PyTorch"
        elif dep == "accelerate":
            label = "Accelerate"

        print(_box_line(f"  {label + ':':14s} {ver or 'not installed'}"))

    # --- Test Results ---
    print(_box_divider())
    print(_box_line("Test Results"))

    tr = report.test_results
    if tr is None:
        print(_box_line("  Tests skipped (--skip-tests flag)"))
    else:
        print(_box_line(f"  Total:     {tr.total}"))
        print(_box_line(f"  Passed:    {tr.passed}"))

        # Break down skips if possible.
        cuda_skips = 0
        if tr.per_file:
            for file_counts in tr.per_file.values():
                cuda_skips += file_counts.get("skipped", 0)
        other_skips = tr.skipped - cuda_skips if cuda_skips else tr.skipped
        if cuda_skips > 0 and other_skips >= 0:
            print(_box_line(f"  Skipped:   {tr.skipped}"))
        else:
            print(_box_line(f"  Skipped:   {tr.skipped}"))

        print(_box_line(f"  Failed:    {tr.failed}"))
        if tr.errors > 0:
            print(_box_line(f"  Errors:    {tr.errors}"))
        print(_box_line(f"  Duration:  {tr.duration_seconds:.2f}s"))

        # Show failure details (truncated).
        if tr.failure_details:
            print(_box_line(""))
            print(_box_line("  Failure Details:"))
            for detail in tr.failure_details[:5]:
                # Truncate long lines.
                short = detail[: (_BOX_WIDTH - 6)]
                print(_box_line(f"    {short}"))
            if len(tr.failure_details) > 5:
                print(_box_line(f"    ... and {len(tr.failure_details) - 5} more"))

    # --- Hardware Compatibility ---
    print(_box_divider())
    print(_box_line("Hardware Compatibility"))

    hc = report.hardware_compatibility
    print(_box_line(f"  Max model size (FP16):  {hc.max_model_fp16_params}"))
    print(_box_line(f"  Max model size (4-bit): {hc.max_model_4bit_params}"))
    print(_box_line(f"  Recommended backend:    {hc.recommended_backend}"))
    viable_str = "Yes" if hc.layer_by_layer_viable else "No"
    if report.system_profile.gpu.backend == "mps":
        viable_str += f" ({report.system_profile.ram_gb}GB unified memory)"
    elif report.system_profile.gpu.vram_gb:
        viable_str += f" ({report.system_profile.gpu.vram_gb}GB VRAM)"
    print(_box_line(f"  Layer-by-layer viable:  {viable_str}"))

    # --- Status ---
    print(_box_divider())
    if report.status == "PASS":
        print(_box_line("Status: ✅ ALL TESTS PASSING"))
    elif report.status == "FAIL":
        print(_box_line("Status: ❌ TESTS FAILING"))
    else:
        print(_box_line("Status: ⏭️  TESTS SKIPPED"))

    print(_box_bottom())
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full validation pipeline.

    Returns:
        Exit code: 0 for success/pass, 1 for failures.
    """
    skip_tests = "--skip-tests" in sys.argv
    json_only = "--json-only" in sys.argv

    if not json_only:
        print("🔍 Detecting system environment...")

    # 1. Detect system profile.
    profile = detect_system_profile()
    airllm_version = _get_airllm_version()

    if not json_only:
        print(f"   ✓ {profile.os_name} {profile.os_version} ({profile.architecture})")
        print(f"   ✓ {profile.cpu_model} ({profile.cpu_cores} cores)")
        print(f"   ✓ {profile.ram_gb} GB RAM")
        print(f"   ✓ GPU: {profile.gpu.name}")

    # 2. Determine test categories.
    categories = determine_test_categories(profile)

    if not json_only:
        enabled = [k for k, v in categories.items() if v]
        print(f"   ✓ Test categories enabled: {', '.join(enabled)}")

    # 3. Run tests (unless skipped).
    test_results: TestResult | None = None
    if not skip_tests:
        if not json_only:
            print("\n🧪 Running pytest suite...")

        test_results = run_tests()

        if not json_only:
            print(
                f"   ✓ {test_results.passed} passed, "
                f"{test_results.failed} failed, "
                f"{test_results.skipped} skipped "
                f"in {test_results.duration_seconds:.2f}s"
            )

    # 4. Assess hardware compatibility.
    compatibility = assess_hardware(profile)

    # 5. Generate report.
    report = generate_report(
        profile=profile,
        categories=categories,
        test_results=test_results,
        compatibility=compatibility,
        airllm_version=airllm_version,
    )

    # 6. Save JSON report.
    report_path = save_report(report)
    if not json_only:
        print(f"\n📄 Report saved to: {report_path}")

    # 7. Print summary or JSON.
    if json_only:
        data = asdict(report)
        if data.get("test_results") and "raw_output" in data["test_results"]:
            del data["test_results"]["raw_output"]
        print(json.dumps(data, indent=2, default=str))
    else:
        print_summary(report)

    # Return exit code based on test status.
    if report.status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
