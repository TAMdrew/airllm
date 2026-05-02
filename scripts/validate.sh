#!/usr/bin/env bash
# AirLLM Validation Script — Bash Wrapper
#
# Detects Python environment, activates venv if available,
# and runs the validation script.
#
# Usage:
#   ./scripts/validate.sh              # Full validation
#   ./scripts/validate.sh --skip-tests # System profile only
#   ./scripts/validate.sh --json-only  # JSON output only
#
# Requirements:
#   - Python 3.11+ (checked automatically)
#   - macOS or Linux (Windows users: run validate.py directly)

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VALIDATE_PY="${SCRIPT_DIR}/validate.py"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11

# ANSI colors (disabled if not a TTY).
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'  # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_ok() {
    echo -e "${GREEN}[OK]${NC}   $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------

check_python_version() {
    local python_cmd="$1"

    if ! command -v "${python_cmd}" &>/dev/null; then
        return 1
    fi

    local version
    version="$("${python_cmd}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    local major minor
    major="$(echo "${version}" | cut -d. -f1)"
    minor="$(echo "${version}" | cut -d. -f2)"

    if [ "${major}" -ge "${MIN_PYTHON_MAJOR}" ] && [ "${minor}" -ge "${MIN_PYTHON_MINOR}" ]; then
        return 0
    fi
    return 1
}

find_python() {
    # Check common Python executable names in priority order.
    for cmd in python3 python python3.14 python3.13 python3.12 python3.11; do
        if check_python_version "${cmd}"; then
            echo "${cmd}"
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Virtual environment detection and activation
# ---------------------------------------------------------------------------

activate_venv() {
    # Check common venv locations relative to project root.
    local venv_dirs=(".venv" "venv" "env" ".env")

    for vdir in "${venv_dirs[@]}"; do
        local activate_path="${PROJECT_ROOT}/${vdir}/bin/activate"
        if [ -f "${activate_path}" ]; then
            log_info "Activating virtual environment: ${vdir}"
            # shellcheck disable=SC1090
            source "${activate_path}"
            return 0
        fi
    done

    # Check if we're already in a virtualenv.
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        log_ok "Already in virtual environment: ${VIRTUAL_ENV}"
        return 0
    fi

    log_warn "No virtual environment found. Using system Python."
    return 1
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

preflight() {
    log_info "AirLLM Validation — Pre-flight Checks"
    echo ""

    # 1. Check OS (this script is macOS/Linux only).
    local os_name
    os_name="$(uname -s)"
    if [ "${os_name}" = "Darwin" ] || [ "${os_name}" = "Linux" ]; then
        log_ok "Operating system: ${os_name}"
    else
        log_error "Unsupported OS: ${os_name}. Run validate.py directly on Windows."
        exit 1
    fi

    # 2. Try activating a virtual environment.
    activate_venv || true

    # 3. Find a suitable Python.
    PYTHON_CMD="$(find_python || true)"
    if [ -z "${PYTHON_CMD}" ]; then
        log_error "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ not found."
        log_error "Install Python 3.11+ and try again."
        exit 1
    fi

    local python_version
    python_version="$("${PYTHON_CMD}" --version 2>&1)"
    log_ok "Python: ${python_version} (${PYTHON_CMD})"

    # 4. Check that the validation script exists.
    if [ ! -f "${VALIDATE_PY}" ]; then
        log_error "Validation script not found: ${VALIDATE_PY}"
        exit 1
    fi
    log_ok "Validation script found: ${VALIDATE_PY}"

    # 5. Quick check for core dependencies.
    if "${PYTHON_CMD}" -c "import torch" &>/dev/null; then
        log_ok "PyTorch: available"
    else
        log_warn "PyTorch not installed — some checks will be limited"
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    preflight

    log_info "Running validation..."
    echo ""

    # Pass through all CLI arguments to the Python script.
    exec "${PYTHON_CMD}" "${VALIDATE_PY}" "$@"
}

main "$@"
