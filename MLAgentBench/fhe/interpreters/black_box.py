"""
Black Box FHE Interpreter.

Handles challenges with pre-encrypted testcases:
- Uses challenge's own Dockerfile
- Mounts pre-encrypted testcase directory
- Uses challenge's verifier.cpp for validation

Adapted from AIDE-FHE for MLAgentBench.
"""

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Tuple

from .base import BaseInterpreter, ExecutionResult, ValidationResult


class BlackBoxInterpreter(BaseInterpreter):
    """
    Interpreter for black box FHE challenges.

    Black box challenges have:
    - Pre-encrypted testcases (tests/testcase1/, tests/testcase2/, etc.)
    - Their own Dockerfile that builds solution + verifier
    - verifier.cpp that decrypts and validates output

    Workflow:
    1. Copy template files + inject eval() code
    2. Build using challenge's Dockerfile
    3. Run with mounted testcase directory
    4. Parse verifier output for accuracy
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = f"fhe-black-box-{self.spec.task}"

    def _prepare_source(self, code: str) -> None:
        """Prepare source by copying challenge files + injecting code."""
        challenge_dir = self.spec.challenge_dir

        # Copy challenge-level files
        for filename in ["Dockerfile", "CMakeLists.txt", "verifier.cpp"]:
            src = challenge_dir / filename
            if src.exists():
                shutil.copy2(src, self.workspace_dir / filename)

        # Copy template files
        if self.spec.template_dir and self.spec.template_dir.exists():
            template_dest = self.workspace_dir / "templates" / "openfhe"
            template_dest.mkdir(parents=True, exist_ok=True)

            for f in self.spec.template_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, template_dest / f.name)

            # Inject code into yourSolution.cpp
            solution_file = template_dest / "yourSolution.cpp"
            if solution_file.exists():
                self._inject_code(solution_file, code)

    def build(self) -> Tuple[bool, List[str]]:
        """Build using challenge's Dockerfile."""
        dockerfile = self.workspace_dir / "Dockerfile"

        if not dockerfile.exists():
            return False, ["ERROR: Dockerfile not found in challenge directory"]

        return self.docker_build(
            dockerfile=dockerfile,
            context=self.workspace_dir,
            tag=self.image_name,
            timeout=self.build_timeout,
        )

    def run(self, testcase_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
        """Run solution with mounted testcase."""
        if testcase_path is None:
            if self.spec.testcase_dirs:
                testcase_path = self.spec.testcase_dirs[0]
            else:
                return False, ["ERROR: No testcase available"]

        # Clean up stale output.txt
        output_file = testcase_path / "output.txt"
        if output_file.exists():
            try:
                output_file.unlink()
            except PermissionError:
                try:
                    subprocess.run(
                        ["docker", "run", "--rm",
                         "-v", f"{testcase_path}:/cleanup",
                         "alpine", "rm", "-f", "/cleanup/output.txt"],
                        capture_output=True, timeout=30)
                except Exception:
                    pass

        # Run with testcase mounted at /data
        return self.run_docker(
            image=self.image_name,
            command=[],
            volumes={str(testcase_path): "/data"},
            timeout=self.run_timeout,
        )

    def execute(self, code: str, testcase_path: Optional[Path] = None) -> ExecutionResult:
        """Execute solution with runtime error detection."""
        result = ExecutionResult()
        start_time = time.time()

        try:
            self._prepare_source(code)

            # Build
            build_start = time.time()
            result.build_success, result.build_output = self.build()
            result.build_time = time.time() - build_start

            if not result.build_success:
                self._analyze_build_error(result)
                result.total_time = time.time() - start_time
                return result

            # Run
            run_start = time.time()
            run_ok, run_out = self.run(testcase_path)
            result.run_time = time.time() - run_start
            result.run_output = run_out

            # Check for runtime errors
            output_text = "\n".join(run_out) if isinstance(run_out, list) else str(run_out)
            has_runtime_error = self._has_runtime_error(output_text)

            if not run_ok or has_runtime_error:
                if has_runtime_error:
                    result.run_success = False
                    self._analyze_runtime_error(result)
                    result.total_time = time.time() - start_time
                    return result

                # Check if verifier produced output
                has_verifier_output = bool(
                    re.search(r'(?:accuracy|slots?\s+passed|total\s+score)', output_text, re.IGNORECASE)
                )
                if has_verifier_output:
                    result.run_success = True
                else:
                    result.run_success = False
                    self._analyze_runtime_error(result)
                    result.total_time = time.time() - start_time
                    return result
            else:
                result.run_success = True

            # Check for output and validate
            if testcase_path is None and self.spec.testcase_dirs:
                testcase_path = self.spec.testcase_dirs[0]

            output_file = testcase_path / "output.txt" if testcase_path else None
            result.output_generated = output_file and output_file.exists()

            if result.output_generated and testcase_path:
                result.output_path = output_file
                result.validation = self.validate(output_file, testcase_path, run_out)

        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)

        result.total_time = time.time() - start_time
        return result

    def validate(self, output_path: Path, testcase_path: Path, run_output: list = None) -> ValidationResult:
        """Parse validation from verifier output."""
        result = ValidationResult()

        if run_output:
            output_text = "\n".join(run_output) if isinstance(run_output, list) else str(run_output)
        else:
            success, output = self.run_docker(
                image=self.image_name,
                command=[],
                volumes={str(testcase_path): "/data"},
                timeout=self.run_timeout,
            )
            output_text = "\n".join(output)

        # Parse accuracy
        acc_match = re.search(r'[Aa]ccuracy[:\s]+(\d+\.?\d*)', output_text)
        if acc_match:
            result.accuracy = float(acc_match.group(1))
            if result.accuracy > 1:
                result.accuracy /= 100

        # Parse slots
        slots_match = re.search(r'[Ss]lots\s+passed[:\s]+(\d+)/(\d+)', output_text)
        if slots_match:
            passed = int(slots_match.group(1))
            total = int(slots_match.group(2))
            result.total_slots = total
            if result.accuracy is None:
                result.accuracy = passed / total
        else:
            total_match = re.search(r'[Tt]otal\s+slots[:\s]+(\d+)', output_text)
            correct_match = re.search(r'[Cc]orrect\s+slots[:\s]+(\d+)', output_text)
            if total_match:
                result.total_slots = int(total_match.group(1))
            if correct_match and total_match and result.accuracy is None:
                result.accuracy = int(correct_match.group(1)) / int(total_match.group(1))

        # Parse fatal errors
        fatal_match = re.search(r'[Ff]atal\s+errors?[:\s]+(\d+)', output_text)
        if fatal_match:
            result.fatal_error_count = int(fatal_match.group(1))

        # Parse mean/max error
        mean_match = re.search(r'(?:[Mm]ean|[Aa]verage)\s+error[:\s]+(\d+\.?\d*)', output_text)
        if mean_match:
            result.mean_error = float(mean_match.group(1))

        max_match = re.search(r'[Mm]ax\s+error[:\s]+(\d+\.?\d*)', output_text)
        if max_match:
            result.max_error = float(max_match.group(1))

        # Determine if passed
        threshold = self.spec.scoring.accuracy_threshold
        max_fatal = self.spec.scoring.max_fatal_errors

        if result.accuracy is not None:
            result.passed = (
                result.accuracy >= threshold and
                result.fatal_error_count <= max_fatal
            )

        return result

    def _has_runtime_error(self, output: str) -> bool:
        """Detect runtime errors in output."""
        output_lower = output.lower()

        runtime_patterns = [
            "openfheexception",
            "lbcrypto::",
            "config_error",
            "what():",
            "terminate called",
            "segmentation fault",
            "aborted",
            "core dumped",
            "ciphertext passed to decrypt is empty",
        ]
        for pattern in runtime_patterns:
            if pattern in output_lower:
                return True

        return False

    def cleanup(self) -> None:
        """Remove Docker image."""
        try:
            subprocess.run(
                ["docker", "rmi", "-f", self.image_name],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass
