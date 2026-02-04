"""
White Box OpenFHE Interpreter.

Handles OpenFHE challenges validated with fherma-validator:
- Uses yashalabinc/fherma-validator Docker image
- Validator generates keys, encrypts inputs, runs solution, validates output
- Parses result.json for accuracy metrics

Adapted from AIDE-FHE for MLAgentBench.
"""

import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional, List, Tuple

from .base import BaseInterpreter, ExecutionResult, ValidationResult

FHERMA_VALIDATOR_IMAGE = "yashalabinc/fherma-validator"


class WhiteBoxInterpreter(BaseInterpreter):
    """
    Interpreter for white box OpenFHE challenges.

    Uses fherma-validator which:
    1. Reads config.json for crypto parameters
    2. Generates CryptoContext and keys
    3. Encrypts test inputs
    4. Builds and runs the solution
    5. Decrypts output and validates accuracy
    6. Writes result.json

    Workflow:
    1. Copy template files to app_build/
    2. Inject eval() code into yourSolution.cpp
    3. Run fherma-validator with mounted directories
    4. Parse result.json for metrics
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app_build_dir = self.workspace_dir / "app_build"
        self.app_build_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_source(self, code: str) -> None:
        """Prepare app_build/ directory with template + injected code."""
        config_json, actual_code = self._parse_code_with_config(code)

        # Clear previous build
        if self.app_build_dir.exists():
            shutil.rmtree(self.app_build_dir, ignore_errors=True)
        self.app_build_dir.mkdir(parents=True, exist_ok=True)

        # Copy template files
        if self.spec.template_dir and self.spec.template_dir.exists():
            for f in self.spec.template_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, self.app_build_dir / f.name)

        # Replace config.json if provided
        if config_json:
            config_file = self.app_build_dir / "config.json"
            config_file.write_text(config_json)

        # Find and inject code
        solution_file = self.app_build_dir / "yourSolution.cpp"
        if not solution_file.exists():
            for cpp_file in self.app_build_dir.glob("*.cpp"):
                if cpp_file.name == "main.cpp":
                    continue
                content = cpp_file.read_text()
                if "void" in content and "eval()" in content:
                    solution_file = cpp_file
                    break

        if solution_file.exists():
            self._inject_code(solution_file, actual_code)
        else:
            (self.app_build_dir / "solution.cpp").write_text(actual_code)

    def _parse_code_with_config(self, code: str) -> Tuple[Optional[str], str]:
        """Parse code that may contain CONFIG and CODE sections."""
        config_json = None
        actual_code = code

        if '### CONFIG ###' in code:
            parts = code.split('### CONFIG ###', 1)
            if len(parts) > 1:
                rest = parts[1]
                if '### CODE ###' in rest:
                    config_part, code_part = rest.split('### CODE ###', 1)
                    actual_code = code_part.strip()
                    config_json = config_part.strip()
                else:
                    lines = rest.strip().split('\n')
                    config_lines = []
                    code_start = 0
                    in_json = False
                    brace_count = 0

                    for i, line in enumerate(lines):
                        if '{' in line:
                            in_json = True
                        if in_json:
                            config_lines.append(line)
                            brace_count += line.count('{') - line.count('}')
                            if brace_count <= 0:
                                code_start = i + 1
                                break

                    config_json = '\n'.join(config_lines)
                    actual_code = '\n'.join(lines[code_start:]).strip()

                if config_json:
                    try:
                        json.loads(config_json)
                    except json.JSONDecodeError:
                        config_json = None

        elif '### CODE ###' in code:
            actual_code = code.split('### CODE ###', 1)[1].strip()

        return config_json, actual_code

    def build(self) -> Tuple[bool, List[str]]:
        """Validate that files are in place (actual build done by fherma-validator)."""
        required_files = ["yourSolution.cpp", "yourSolution.h", "main.cpp", "CMakeLists.txt"]
        missing = []

        for f in required_files:
            if not (self.app_build_dir / f).exists():
                missing.append(f)

        if missing:
            if "yourSolution.cpp" in missing and (self.app_build_dir / "solution.cpp").exists():
                missing.remove("yourSolution.cpp")

        if missing:
            return False, [f"Missing required files: {missing}"]

        return True, ["Build files prepared"]

    def run(self, testcase_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
        """Run solution using fherma-validator."""
        # Find test_case.json
        if testcase_path and testcase_path.is_file():
            test_case_json = testcase_path
        elif testcase_path and testcase_path.is_dir():
            test_case_json = testcase_path / "test_case.json"
        elif self.spec.challenge_dir:
            test_case_json = self.spec.challenge_dir / "tests" / "test_case.json"
        else:
            return False, ["ERROR: No test_case.json found"]

        if not test_case_json.exists():
            return False, [f"ERROR: test_case.json not found at {test_case_json}"]

        # Copy test_case.json to workspace
        tests_dir = self.workspace_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        shutil.copy2(test_case_json, tests_dir / "test_case.json")

        volumes = {str(self.workspace_dir): "/fherma"}

        # Mount data directory if exists
        if self.spec.challenge_dir:
            data_dir = self.spec.challenge_dir / "data"
            if data_dir.exists() and data_dir.is_dir():
                volumes[str(data_dir)] = "/fherma/data"

        return self.run_docker(
            image=FHERMA_VALIDATOR_IMAGE,
            command=[
                "--project-folder=/fherma/app_build",
                "--testcase=/fherma/tests/test_case.json",
            ],
            volumes=volumes,
            timeout=self.run_timeout,
        )

    def validate(self, output_path: Path, testcase_path: Path) -> ValidationResult:
        """Parse result.json written by fherma-validator."""
        result = ValidationResult()

        result_file = self.app_build_dir / "result.json"
        if not result_file.exists():
            result.details["error"] = "result.json not found"
            return result

        try:
            data = json.loads(result_file.read_text())
            result.details = data

            if data.get("compilation_error"):
                result.details["error"] = data["compilation_error"]
                return result

            testcases = data.get("testcases", [])
            if not testcases:
                result.details["error"] = "No testcases in result.json"
                return result

            all_errors = []
            total_slots = 0
            total_correct = 0
            threshold = self.spec.scoring.error_threshold if self.spec and self.spec.scoring else 0.001

            for tc in testcases:
                runs = tc.get("runs", [])
                for run in runs:
                    actual = run.get("result", [])
                    expected = run.get("expected_output", [])

                    if not actual or not expected:
                        continue

                    for a, e in zip(actual, expected):
                        try:
                            err = abs(float(a) - float(e))
                            all_errors.append(err)
                            total_slots += 1
                            if err < threshold:
                                total_correct += 1
                        except (ValueError, TypeError):
                            pass

            if total_slots > 0:
                result.accuracy = total_correct / total_slots
                result.total_slots = total_slots
                result.mean_error = sum(all_errors) / len(all_errors)
                result.max_error = max(all_errors)
                result.passed = result.accuracy >= (
                    self.spec.scoring.accuracy_threshold if self.spec and self.spec.scoring else 0.8
                )
            else:
                result.details["error"] = "No valid result/expected pairs found"

        except json.JSONDecodeError as e:
            result.details["error"] = f"Failed to parse result.json: {e}"
        except Exception as e:
            result.details["error"] = f"Validation error: {e}"

        return result

    def execute(self, code: str, testcase_path: Optional[Path] = None) -> ExecutionResult:
        """Execute solution with fherma-validator."""
        result = ExecutionResult()
        start_time = time.time()

        # Handle missing CONFIG section
        if "### CONFIG ###" not in code:
            default_config = None
            if self.spec.template_dir:
                config_file = self.spec.template_dir / "config.json"
                if config_file.exists():
                    default_config = config_file.read_text()
            if default_config:
                code = f"### CONFIG ###\n{default_config}\n\n### CODE ###\n{code}"
            else:
                result.build_success = False
                result.error_type = "FORMAT_ERROR"
                result.error_message = "Missing ### CONFIG ### section and no default config.json"
                result.total_time = time.time() - start_time
                return result

        try:
            self._prepare_source(code)

            build_ok, build_out = self.build()
            result.build_output = build_out

            if not build_ok:
                result.build_success = False
                self._analyze_build_error(result)
                result.total_time = time.time() - start_time
                return result

            result.build_success = True

            run_start = time.time()
            run_ok, run_out = self.run(testcase_path)
            result.run_time = time.time() - run_start
            result.run_output = run_out

            output_text = "\n".join(run_out)
            is_build_error = self._is_build_error(output_text)
            has_runtime_error = self._has_runtime_error(output_text)

            if not run_ok or is_build_error or has_runtime_error:
                if is_build_error:
                    result.build_success = False
                    result.run_success = False
                    result.build_output = run_out
                    self._analyze_build_error(result)
                else:
                    result.run_success = False
                    self._analyze_runtime_error(result)

                # Check for partial results
                result_file = self.app_build_dir / "result.json"
                if result_file.exists():
                    result.output_generated = True
                    result.output_path = result_file
                    result.validation = self.validate(result_file, testcase_path)
                    if result.validation and result.validation.accuracy is not None:
                        result.run_success = True

                result.total_time = time.time() - start_time
                return result

            result.run_success = True

            result_file = self.app_build_dir / "result.json"
            result.output_generated = result_file.exists()

            if result.output_generated:
                result.output_path = result_file
                result.validation = self.validate(result_file, testcase_path)

        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)

        result.total_time = time.time() - start_time
        return result

    def _has_runtime_error(self, output: str) -> bool:
        """Detect runtime errors in fherma-validator output."""
        output_lower = output.lower()

        if "run error:" in output_lower:
            return True
        if "return code: -" in output_lower:
            return True

        runtime_patterns = [
            "openfheexception",
            "lbcrypto::",
            "what():",
            "terminate called",
            "segmentation fault",
            "ciphertext passed to decrypt is empty",
        ]
        for pattern in runtime_patterns:
            if pattern in output_lower:
                return True

        return False

    def _is_build_error(self, output: str) -> bool:
        """Detect if error is from build/compilation phase."""
        output_lower = output.lower()

        # Runtime indicators (check first)
        runtime_indicators = [
            "lbcrypto::", "openfheexception", "what():", "segmentation fault",
            "removing last element", "multiplicative depth", "cannot decrypt",
            "ciphertext passed to decrypt is empty",
        ]
        for indicator in runtime_indicators:
            if indicator in output_lower:
                return False

        # Build indicators
        build_indicators = [
            "cmake error", "cmake warning", "make[", "error: ",
            "undefined reference", "fatal error:", "collect2: error",
            "cannot find -l", "no such file or directory",
        ]
        for indicator in build_indicators:
            if indicator in output_lower:
                return True

        if re.search(r'\w+\.cpp:\d+:\d+: error:', output):
            return True

        return False

    def cleanup(self) -> None:
        """Cleanup app_build directory."""
        if self.app_build_dir.exists():
            shutil.rmtree(self.app_build_dir, ignore_errors=True)
