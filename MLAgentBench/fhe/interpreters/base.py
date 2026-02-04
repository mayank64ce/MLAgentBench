"""
Base interpreter for FHE challenges.

Provides common functionality for all challenge types:
- Template loading and code injection
- Docker execution
- Result parsing

Adapted from AIDE-FHE for MLAgentBench.
"""

import os
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from ..challenge_parser import FHEChallengeSpec


@dataclass
class ValidationResult:
    """Result from validating a solution."""
    passed: bool = False
    accuracy: Optional[float] = None
    error_count: int = 0
    fatal_error_count: int = 0
    total_slots: int = 0
    mean_error: float = 0.0
    max_error: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "accuracy": self.accuracy,
            "error_count": self.error_count,
            "fatal_error_count": self.fatal_error_count,
            "total_slots": self.total_slots,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "details": self.details,
        }


@dataclass
class ExecutionResult:
    """Result of executing an FHE solution."""

    # Build phase
    build_success: bool = False
    build_output: List[str] = field(default_factory=list)
    build_time: float = 0.0

    # Run phase
    run_success: bool = False
    run_output: List[str] = field(default_factory=list)
    run_time: float = 0.0

    # Output
    output_generated: bool = False
    output_path: Optional[Path] = None

    # Validation
    validation: Optional[ValidationResult] = None

    # Error info
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Timing
    total_time: float = 0.0

    @property
    def accuracy(self) -> Optional[float]:
        """Accuracy from validation, if available."""
        if self.validation:
            return self.validation.accuracy
        return None

    def _filter_docker_noise(self, output: List[str]) -> List[str]:
        """Filter out Docker build step noise."""
        filtered = []
        skip_patterns = [
            "Step ",
            " ---> ",
            "Using cache",
            "Sending build context",
        ]

        for line in output:
            if any(p in line for p in skip_patterns):
                continue
            if line.strip():
                filtered.append(line)

        return filtered if filtered else output[-50:]

    def get_observation(self) -> str:
        """Generate observation string for MLAgentBench agent."""
        lines = []

        if not self.build_success:
            lines.append("BUILD FAILED")
            lines.append(f"Error Type: {self.error_type}")
            lines.append(f"Error Message: {self.error_message}")
            lines.append("")
            lines.append("Build output (last 50 lines):")
            filtered_output = self._filter_docker_noise(self.build_output)
            lines.extend(filtered_output[-50:])
            return "\n".join(lines)

        if not self.run_success:
            lines.append("RUNTIME FAILED")
            lines.append(f"Error Type: {self.error_type}")
            lines.append(f"Error Message: {self.error_message}")
            lines.append("")
            lines.append("Runtime output (last 50 lines):")
            lines.extend(self.run_output[-50:])
            return "\n".join(lines)

        if not self.output_generated:
            lines.append("NO OUTPUT GENERATED")
            lines.append("The solution ran but did not produce an output file.")
            lines.append("Check that output is written to the correct path.")
            return "\n".join(lines)

        if self.validation:
            status = "PASSED" if self.validation.passed else "FAILED"
            lines.append(f"VALIDATION: {status}")
            if self.validation.accuracy is not None:
                lines.append(f"Accuracy: {self.validation.accuracy:.4f}")
            lines.append(f"Mean error: {self.validation.mean_error:.6f}")
            lines.append(f"Max error: {self.validation.max_error:.6f}")
            lines.append(f"Total slots: {self.validation.total_slots}")
            lines.append(f"Fatal errors: {self.validation.fatal_error_count}")
            return "\n".join(lines)

        lines.append("EXECUTION COMPLETE")
        lines.append(f"Build time: {self.build_time:.2f}s")
        lines.append(f"Run time: {self.run_time:.2f}s")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build_success": self.build_success,
            "build_time": self.build_time,
            "run_success": self.run_success,
            "run_time": self.run_time,
            "output_generated": self.output_generated,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "total_time": self.total_time,
            "validation": self.validation.to_dict() if self.validation else None,
        }


class BaseInterpreter(ABC):
    """
    Base class for FHE interpreters.

    Handles:
    - Loading template files
    - Injecting generated code into templates
    - Docker execution
    - Result parsing

    Subclasses implement:
    - build(): Build the solution
    - run(): Run the solution
    - validate(): Validate the output
    """

    def __init__(
        self,
        spec: FHEChallengeSpec,
        workspace_dir: Path | str,
        build_timeout: int = 300,
        run_timeout: int = 600,
    ):
        self.spec = spec
        self.workspace_dir = Path(workspace_dir).resolve()
        self.build_timeout = build_timeout
        self.run_timeout = run_timeout

        # Create workspace directories
        self.src_dir = self.workspace_dir / "src"
        self.build_dir = self.workspace_dir / "build"
        self.output_dir = self.workspace_dir / "output"

        self.src_dir.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str, testcase_path: Optional[Path] = None) -> ExecutionResult:
        """
        Execute a solution.

        Args:
            code: Generated code (eval() function body or full file)
            testcase_path: Path to testcase directory

        Returns:
            ExecutionResult with build/run/validation status
        """
        result = ExecutionResult()
        start_time = time.time()

        try:
            # 1. Prepare source files
            self._prepare_source(code)

            # 2. Build
            build_start = time.time()
            result.build_success, result.build_output = self.build()
            result.build_time = time.time() - build_start

            if not result.build_success:
                self._analyze_build_error(result)
                result.total_time = time.time() - start_time
                return result

            # 3. Run
            run_start = time.time()
            result.run_success, result.run_output = self.run(testcase_path)
            result.run_time = time.time() - run_start

            if not result.run_success:
                self._analyze_runtime_error(result)
                result.total_time = time.time() - start_time
                return result

            # 4. Check output
            result.output_generated, result.output_path = self._check_output()

            # 5. Validate
            if result.output_generated and testcase_path:
                result.validation = self.validate(result.output_path, testcase_path)

        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)

        result.total_time = time.time() - start_time
        return result

    def _prepare_source(self, code: str) -> None:
        """Prepare source files from template + generated code."""
        # Copy template files
        if self.spec.template_dir and self.spec.template_dir.exists():
            for f in self.spec.template_dir.iterdir():
                if f.is_file():
                    dest = self.src_dir / f.name
                    shutil.copy2(f, dest)

        # Inject code into yourSolution.cpp
        solution_file = self.src_dir / "yourSolution.cpp"
        if solution_file.exists():
            self._inject_code(solution_file, code)
        else:
            (self.src_dir / "solution.cpp").write_text(code)

    def _inject_code(self, solution_file: Path, code: str) -> None:
        """Inject generated code into the eval() function."""
        template = solution_file.read_text()

        # Strip void eval() wrapper if included
        code = self._strip_eval_wrapper(code)

        # Pattern 1: Empty eval function body
        pattern1 = r'(void\s+\w+::eval\s*\(\s*\)\s*\{)\s*(\})'
        if re.search(pattern1, template):
            new_content = re.sub(pattern1, rf'\1\n{code}\n\2', template)
            solution_file.write_text(new_content)
            return

        # Pattern 2: eval() with only comments
        pattern2 = r'(void\s+\w+::eval\s*\(\s*\)\s*\{)\s*((?:\s*//[^\n]*\n)+)\s*(\})'
        if re.search(pattern2, template):
            new_content = re.sub(pattern2, rf'\1\n{code}\n\3', template)
            solution_file.write_text(new_content)
            return

        # Pattern 3: TODO comment
        pattern3 = r'(void\s+\w+::eval\s*\(\s*\)\s*\{[^}]*)(//\s*TODO[^\n]*\n)([^}]*\})'
        if re.search(pattern3, template):
            new_content = re.sub(pattern3, rf'\1{code}\n\3', template)
            solution_file.write_text(new_content)
            return

        # Pattern 4: Your implementation comment
        pattern4 = r'//\s*[Yy]our\s+implementation\s+here[^\n]*\n'
        if re.search(pattern4, template):
            new_content = re.sub(pattern4, f'{code}\n', template)
            solution_file.write_text(new_content)
            return

        # Pattern 5: Find eval() and replace body using brace matching
        match = re.search(r'void\s+\w+::eval\s*\(\s*\)\s*\{', template)
        if match:
            start = match.end()
            depth = 1
            pos = start
            while pos < len(template) and depth > 0:
                if template[pos] == '{':
                    depth += 1
                elif template[pos] == '}':
                    depth -= 1
                pos += 1
            if depth == 0:
                new_content = template[:start] + '\n' + code + '\n' + template[pos-1:]
                solution_file.write_text(new_content)
                return

        # Fallback: Replace entire file
        solution_file.write_text(code)

    def _strip_eval_wrapper(self, code: str) -> str:
        """Strip void eval() wrapper if LLM included it."""
        code = code.strip()

        match = re.search(r'void\s+(?:\w+::)?eval\s*\(\s*\)\s*\{', code)
        if match:
            after_open = code[match.end():]
            depth = 1
            for i, ch in enumerate(after_open):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return after_open[:i].strip()

        return code

    @abstractmethod
    def build(self) -> Tuple[bool, List[str]]:
        """Build the solution. Returns (success, output_lines)."""
        pass

    @abstractmethod
    def run(self, testcase_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
        """Run the solution. Returns (success, output_lines)."""
        pass

    @abstractmethod
    def validate(self, output_path: Path, testcase_path: Path) -> ValidationResult:
        """Validate the solution output."""
        pass

    def _check_output(self) -> Tuple[bool, Optional[Path]]:
        """Check if output was generated."""
        output_names = ["output.bin", "output.txt", "result.bin", "result.txt"]

        for name in output_names:
            path = self.output_dir / name
            if path.exists() and path.stat().st_size > 0:
                return True, path

        return False, None

    def _analyze_build_error(self, result: ExecutionResult) -> None:
        """Analyze build error."""
        output = "\n".join(result.build_output)
        result.error_type = "BUILD_ERROR"
        result.error_message = self._extract_error_message(output)

    def _analyze_runtime_error(self, result: ExecutionResult) -> None:
        """Analyze runtime error."""
        output = "\n".join(result.run_output)
        result.error_type = "RUNTIME_ERROR"
        result.error_message = self._extract_error_message(output)

    def _extract_error_message(self, output: str) -> str:
        """Extract clean error message from output."""
        # Try to find C++ exception message
        what_match = re.search(r"what\(\):\s*(.+?)(?:\n|$)", output)
        if what_match:
            return self._clean_fhe_error(what_match.group(1).strip())

        # Look for specific FHE error patterns
        fhe_patterns = [
            (r"EvalKey for index \[(\d+)\] is not found", lambda m: f"EvalKey for index [{m.group(1)}] not found"),
            (r"ciphertext passed to (\w+) is empty", lambda m: f"Ciphertext passed to {m.group(1)} is empty"),
            (r"Removing last element.*renders it invalid", lambda _: "Depth exhausted - too many operations"),
            (r"Enable\((\w+)\) must be called", lambda m: f"Enable({m.group(1)}) must be called first"),
        ]

        for pattern, formatter in fhe_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return formatter(match)

        # Look for compile errors
        compile_match = re.search(r"error:\s*(.+?)(?:\n|$)", output)
        if compile_match:
            return compile_match.group(1).strip()[:200]

        # Look for generic error lines
        for line in reversed(output.split('\n')):
            line_lower = line.lower()
            if any(skip in line_lower for skip in ['make[', 'cmake', '===', '---']):
                continue
            if any(kw in line_lower for kw in ['error', 'exception', 'failed', 'abort']):
                cleaned = line.strip()
                if cleaned and len(cleaned) > 5:
                    return cleaned[:200]

        return "Unknown error"

    def _clean_fhe_error(self, raw_msg: str) -> str:
        """Clean FHE error message."""
        cleaned = re.sub(r'^/[^:]+\.(cpp|h|hpp):\d+:?', '', raw_msg).strip()
        cleaned = re.sub(r'^\w+\(\):\s*', '', cleaned).strip()

        if "Removing last element" in cleaned or "DCRTPoly" in cleaned:
            return "Depth exhausted - too many multiplicative operations"

        cleaned = re.sub(r'\s*\[called from:.*\]$', '', cleaned)
        return cleaned[:200] if len(cleaned) > 200 else cleaned

    def run_docker(
        self,
        image: str,
        command: List[str],
        volumes: Dict[str, str] = None,
        workdir: str = None,
        timeout: int = None,
    ) -> Tuple[bool, List[str]]:
        """Run a command in Docker."""
        timeout = timeout or self.run_timeout

        docker_cmd = ["docker", "run", "--rm"]
        docker_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

        if volumes:
            for host, container in volumes.items():
                docker_cmd.extend(["-v", f"{host}:{container}"])

        if workdir:
            docker_cmd.extend(["-w", workdir])

        docker_cmd.append(image)
        docker_cmd.extend(command)

        try:
            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "0"

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            output = (result.stdout + "\n" + result.stderr).strip().split("\n")
            return result.returncode == 0, output

        except subprocess.TimeoutExpired:
            return False, [f"TIMEOUT: Execution exceeded {timeout}s"]
        except Exception as e:
            return False, [f"ERROR: {e}"]

    def docker_build(
        self,
        dockerfile: Path,
        context: Path,
        tag: str,
        timeout: int = None,
    ) -> Tuple[bool, List[str]]:
        """Build a Docker image."""
        timeout = timeout or self.build_timeout

        try:
            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "0"

            result = subprocess.run(
                [
                    "docker", "build",
                    "-f", str(dockerfile),
                    "-t", tag,
                    str(context),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            output = (result.stdout + "\n" + result.stderr).strip().split("\n")
            return result.returncode == 0, output

        except subprocess.TimeoutExpired:
            return False, [f"TIMEOUT: Build exceeded {timeout}s"]
        except Exception as e:
            return False, [f"ERROR: {e}"]

    def cleanup(self) -> None:
        """Cleanup workspace and Docker resources."""
        pass
