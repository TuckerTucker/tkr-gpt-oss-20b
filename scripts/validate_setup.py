#!/usr/bin/env python3
"""
Validate GPT-OSS CLI Chat setup and environment.

Checks:
- Platform compatibility (Apple Silicon recommended)
- Python version
- Required dependencies (MLX, vLLM, etc.)
- Metal/CUDA backend availability
- Memory requirements
- Disk space

Usage:
    python scripts/validate_setup.py
    python scripts/validate_setup.py --verbose
"""

import argparse
import platform
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import shutil


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class ValidationCheck:
    """Represents a single validation check."""

    def __init__(self, name: str, status: str, message: str, required: bool = True):
        self.name = name
        self.status = status  # "pass", "warn", "fail"
        self.message = message
        self.required = required

    def print(self) -> None:
        """Print check result."""
        if self.status == "pass":
            icon = f"{Colors.GREEN}✓{Colors.RESET}"
        elif self.status == "warn":
            icon = f"{Colors.YELLOW}⚠{Colors.RESET}"
        else:
            icon = f"{Colors.RED}✗{Colors.RESET}"

        required_str = "" if self.required else " (optional)"
        print(f"{icon} {self.name}{required_str}: {self.message}")


class SetupValidator:
    """Validates GPT-OSS CLI Chat setup."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks: List[ValidationCheck] = []

    def run_all_checks(self) -> Tuple[int, int, int]:
        """
        Run all validation checks.

        Returns:
            Tuple of (passed, warnings, failed) counts
        """
        print(f"\n{Colors.BOLD}GPT-OSS CLI Chat - Setup Validation{Colors.RESET}")
        print("=" * 60)

        # Platform checks
        self.check_platform()
        self.check_python_version()

        # Architecture checks
        self.check_apple_silicon()
        self.check_metal_backend()

        # Dependency checks
        self.check_pytorch()
        self.check_mlx()
        self.check_vllm()
        self.check_transformers()

        # Resource checks
        self.check_memory()
        self.check_disk_space()

        # Optional tools
        self.check_git()
        self.check_docker()

        print("\n" + "=" * 60)

        # Print all results
        for check in self.checks:
            check.print()

        # Summary
        passed = sum(1 for c in self.checks if c.status == "pass")
        warnings = sum(1 for c in self.checks if c.status == "warn")
        failed = sum(1 for c in self.checks if c.status == "fail")

        print("\n" + "=" * 60)
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"  Passed: {Colors.GREEN}{passed}{Colors.RESET}")
        print(f"  Warnings: {Colors.YELLOW}{warnings}{Colors.RESET}")
        print(f"  Failed: {Colors.RED}{failed}{Colors.RESET}")

        # Overall status
        required_failed = sum(
            1 for c in self.checks if c.status == "fail" and c.required
        )

        print("\n" + "=" * 60)
        if required_failed == 0:
            print(f"{Colors.GREEN}✓ Setup validation passed!{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗ Setup validation failed ({required_failed} required checks){Colors.RESET}")

        return passed, warnings, failed

    def check_platform(self) -> None:
        """Check operating system platform."""
        system = platform.system()
        machine = platform.machine()

        if system == "Darwin" and machine == "arm64":
            self.checks.append(ValidationCheck(
                "Platform",
                "pass",
                f"{system} {machine} (Apple Silicon)"
            ))
        elif system == "Linux":
            self.checks.append(ValidationCheck(
                "Platform",
                "pass",
                f"{system} {machine}"
            ))
        else:
            self.checks.append(ValidationCheck(
                "Platform",
                "warn",
                f"{system} {machine} (Apple Silicon or Linux recommended)"
            ))

    def check_python_version(self) -> None:
        """Check Python version."""
        version = sys.version_info

        if version >= (3, 10):
            self.checks.append(ValidationCheck(
                "Python Version",
                "pass",
                f"{version.major}.{version.minor}.{version.micro}"
            ))
        elif version >= (3, 8):
            self.checks.append(ValidationCheck(
                "Python Version",
                "warn",
                f"{version.major}.{version.minor}.{version.micro} (3.10+ recommended)"
            ))
        else:
            self.checks.append(ValidationCheck(
                "Python Version",
                "fail",
                f"{version.major}.{version.minor}.{version.micro} (3.8+ required)"
            ))

    def check_apple_silicon(self) -> None:
        """Check if running on Apple Silicon."""
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            self.checks.append(ValidationCheck(
                "Apple Silicon",
                "pass",
                "M1/M2/M3 detected",
                required=False
            ))
        else:
            self.checks.append(ValidationCheck(
                "Apple Silicon",
                "warn",
                "Not detected (MLX requires Apple Silicon)",
                required=False
            ))

    def check_metal_backend(self) -> None:
        """Check Metal backend availability."""
        if platform.system() == "Darwin":
            try:
                # Try importing Metal via MLX
                import mlx.core as mx
                self.checks.append(ValidationCheck(
                    "Metal Backend",
                    "pass",
                    "Available via MLX"
                ))
            except ImportError:
                self.checks.append(ValidationCheck(
                    "Metal Backend",
                    "warn",
                    "MLX not installed (install for Apple Silicon acceleration)",
                    required=False
                ))
        else:
            self.checks.append(ValidationCheck(
                "Metal Backend",
                "warn",
                "N/A (not on macOS)",
                required=False
            ))

    def check_pytorch(self) -> None:
        """Check PyTorch installation."""
        try:
            import torch
            version = torch.__version__

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

            device_info = []
            if cuda_available:
                device_info.append(f"CUDA {torch.version.cuda}")
            if mps_available:
                device_info.append("MPS")

            device_str = f" ({', '.join(device_info)})" if device_info else " (CPU only)"

            self.checks.append(ValidationCheck(
                "PyTorch",
                "pass",
                f"{version}{device_str}"
            ))

        except ImportError:
            self.checks.append(ValidationCheck(
                "PyTorch",
                "fail",
                "Not installed (required)"
            ))

    def check_mlx(self) -> None:
        """Check MLX installation."""
        try:
            import mlx
            version = mlx.__version__ if hasattr(mlx, '__version__') else "unknown"

            self.checks.append(ValidationCheck(
                "MLX",
                "pass",
                f"{version} (Apple Silicon acceleration)",
                required=False
            ))

        except ImportError:
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                self.checks.append(ValidationCheck(
                    "MLX",
                    "warn",
                    "Not installed (recommended for Apple Silicon)",
                    required=False
                ))
            else:
                self.checks.append(ValidationCheck(
                    "MLX",
                    "warn",
                    "Not applicable (not on Apple Silicon)",
                    required=False
                ))

    def check_vllm(self) -> None:
        """Check vLLM installation."""
        try:
            import vllm
            version = vllm.__version__ if hasattr(vllm, '__version__') else "unknown"

            self.checks.append(ValidationCheck(
                "vLLM",
                "pass",
                f"{version}"
            ))

        except ImportError:
            self.checks.append(ValidationCheck(
                "vLLM",
                "fail",
                "Not installed (required for inference)"
            ))

    def check_transformers(self) -> None:
        """Check Transformers installation."""
        try:
            import transformers
            version = transformers.__version__

            self.checks.append(ValidationCheck(
                "Transformers",
                "pass",
                f"{version}"
            ))

        except ImportError:
            self.checks.append(ValidationCheck(
                "Transformers",
                "fail",
                "Not installed (required)"
            ))

    def check_memory(self) -> None:
        """Check available memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024 ** 3)
            available_gb = memory.available / (1024 ** 3)

            if total_gb >= 32:
                status = "pass"
                message = f"{total_gb:.1f}GB total, {available_gb:.1f}GB available"
            elif total_gb >= 16:
                status = "warn"
                message = f"{total_gb:.1f}GB total (32GB+ recommended for 20B models)"
            else:
                status = "fail"
                message = f"{total_gb:.1f}GB total (insufficient for 20B models)"

            self.checks.append(ValidationCheck(
                "System Memory",
                status,
                message
            ))

        except ImportError:
            self.checks.append(ValidationCheck(
                "System Memory",
                "warn",
                "Unable to check (psutil not installed)",
                required=False
            ))

    def check_disk_space(self) -> None:
        """Check available disk space."""
        try:
            project_root = Path(__file__).parent.parent
            stat = shutil.disk_usage(project_root)
            free_gb = stat.free / (1024 ** 3)

            if free_gb >= 50:
                status = "pass"
                message = f"{free_gb:.1f}GB free"
            elif free_gb >= 20:
                status = "warn"
                message = f"{free_gb:.1f}GB free (50GB+ recommended)"
            else:
                status = "fail"
                message = f"{free_gb:.1f}GB free (insufficient for model downloads)"

            self.checks.append(ValidationCheck(
                "Disk Space",
                status,
                message
            ))

        except Exception:
            self.checks.append(ValidationCheck(
                "Disk Space",
                "warn",
                "Unable to check",
                required=False
            ))

    def check_git(self) -> None:
        """Check Git installation."""
        if shutil.which("git"):
            try:
                result = subprocess.run(
                    ["git", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version = result.stdout.strip().split()[-1] if result.returncode == 0 else "unknown"
                self.checks.append(ValidationCheck(
                    "Git",
                    "pass",
                    f"{version}",
                    required=False
                ))
            except Exception:
                self.checks.append(ValidationCheck(
                    "Git",
                    "pass",
                    "Installed",
                    required=False
                ))
        else:
            self.checks.append(ValidationCheck(
                "Git",
                "warn",
                "Not found (optional)",
                required=False
            ))

    def check_docker(self) -> None:
        """Check Docker installation."""
        if shutil.which("docker"):
            try:
                result = subprocess.run(
                    ["docker", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version = result.stdout.strip().split()[-1] if result.returncode == 0 else "unknown"
                self.checks.append(ValidationCheck(
                    "Docker",
                    "pass",
                    f"{version}",
                    required=False
                ))
            except Exception:
                self.checks.append(ValidationCheck(
                    "Docker",
                    "warn",
                    "Installed but not responding",
                    required=False
                ))
        else:
            self.checks.append(ValidationCheck(
                "Docker",
                "warn",
                "Not found (optional, for containerized deployment)",
                required=False
            ))


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GPT-OSS CLI Chat setup"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    validator = SetupValidator(verbose=args.verbose)
    passed, warnings, failed = validator.run_all_checks()

    # Exit with appropriate code
    required_failed = sum(
        1 for c in validator.checks if c.status == "fail" and c.required
    )

    sys.exit(1 if required_failed > 0 else 0)


if __name__ == "__main__":
    main()
