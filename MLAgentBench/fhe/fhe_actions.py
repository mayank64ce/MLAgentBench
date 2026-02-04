"""
FHE-specific actions for MLAgentBench.

These actions are registered when running FHE challenges and provide
Docker-based execution and validation.
"""

import os
from pathlib import Path
from functools import wraps

from ..schema import ActionInfo, EnvException


def _get_fhe_context(kwargs):
    """Extract FHE-related context from kwargs."""
    return {
        "fhe_spec": kwargs.get("fhe_spec"),
        "fhe_interpreter": kwargs.get("fhe_interpreter"),
        "work_dir": kwargs.get("work_dir", "."),
        "trace": kwargs.get("trace"),
    }


def execute_fhe_solution(script_name, work_dir=".", **kwargs):
    """
    Execute an FHE solution via Docker container.

    This action:
    1. Reads the solution code from script_name
    2. Injects it into the challenge template
    3. Builds and runs via Docker
    4. Returns validation results as observation

    Args:
        script_name: Path to solution file (e.g., 'yourSolution.cpp')
        work_dir: Working directory
        **kwargs: Additional context including fhe_interpreter

    Returns:
        Observation string with validation results
    """
    interpreter = kwargs.get("fhe_interpreter")
    fhe_spec = kwargs.get("fhe_spec")

    if interpreter is None:
        raise EnvException("FHE interpreter not available. This action is only available for FHE challenges.")

    # Read solution file
    solution_path = Path(work_dir) / script_name
    if not solution_path.exists():
        raise EnvException(f"Solution file not found: {script_name}")

    code = solution_path.read_text()

    # Get testcase path
    testcase_path = None
    if fhe_spec and fhe_spec.testcase_dirs:
        testcase_path = fhe_spec.testcase_dirs[0]

    # Execute via interpreter
    try:
        result = interpreter.execute(code, testcase_path)
        return result.get_observation()
    except Exception as e:
        raise EnvException(f"FHE execution failed: {e}")


def validate_fhe_output(work_dir=".", **kwargs):
    """
    Validate the current FHE solution output.

    This action re-runs validation on the last executed solution
    to get detailed metrics.

    Returns:
        Observation string with accuracy metrics and error details
    """
    interpreter = kwargs.get("fhe_interpreter")
    fhe_spec = kwargs.get("fhe_spec")

    if interpreter is None:
        raise EnvException("FHE interpreter not available.")

    # Check for output files
    output_dir = interpreter.output_dir if hasattr(interpreter, 'output_dir') else Path(work_dir) / "output"

    output_files = ["output.bin", "output.txt", "result.bin", "result.txt"]
    output_path = None
    for name in output_files:
        path = output_dir / name
        if path.exists():
            output_path = path
            break

    if output_path is None:
        # Check app_build for white box challenges
        if hasattr(interpreter, 'app_build_dir'):
            result_json = interpreter.app_build_dir / "result.json"
            if result_json.exists():
                output_path = result_json

    if output_path is None:
        return "No output file found. Run 'Execute FHE Solution' first."

    # Get testcase path
    testcase_path = None
    if fhe_spec and fhe_spec.testcase_dirs:
        testcase_path = fhe_spec.testcase_dirs[0]

    if testcase_path is None:
        return "No testcase available for validation."

    try:
        result = interpreter.validate(output_path, testcase_path)
        lines = [
            f"Validation Result: {'PASSED' if result.passed else 'FAILED'}",
            f"Accuracy: {result.accuracy:.4f}" if result.accuracy is not None else "Accuracy: N/A",
            f"Total Slots: {result.total_slots}",
            f"Mean Error: {result.mean_error:.6f}",
            f"Max Error: {result.max_error:.6f}",
            f"Fatal Errors: {result.fatal_error_count}",
        ]
        return "\n".join(lines)
    except Exception as e:
        raise EnvException(f"Validation failed: {e}")


def understand_fhe_challenge(work_dir=".", **kwargs):
    """
    Get information about the current FHE challenge.

    Returns a summary of the challenge specification including:
    - Challenge type and encryption scheme
    - Constraints (depth, batch size, input range)
    - Available keys
    - Target accuracy

    Returns:
        Observation string with challenge details
    """
    fhe_spec = kwargs.get("fhe_spec")

    if fhe_spec is None:
        raise EnvException("FHE spec not available. This action is only available for FHE challenges.")

    lines = [
        f"Challenge: {fhe_spec.challenge_name or fhe_spec.task}",
        f"Type: {fhe_spec.challenge_type.value}",
        f"Scheme: {fhe_spec.scheme.value}",
        f"Library: {fhe_spec.library.value}",
        "",
        "Constraints:",
        f"  - Multiplicative Depth: {fhe_spec.constraints.depth}",
        f"  - Batch Size: {fhe_spec.constraints.batch_size}",
        f"  - Input Range: [{fhe_spec.constraints.input_range[0]}, {fhe_spec.constraints.input_range[1]}]",
        "",
        "Available Keys:",
        f"  - Public Key: {'Yes' if fhe_spec.keys.public else 'No'}",
        f"  - Multiplication Key: {'Yes' if fhe_spec.keys.multiplication else 'No'}",
    ]

    if fhe_spec.keys.rotation_indices:
        lines.append(f"  - Rotation Keys: {fhe_spec.keys.rotation_indices}")

    lines.extend([
        "",
        "Scoring:",
        f"  - Target Accuracy: {fhe_spec.scoring.accuracy_threshold}",
        f"  - Error Threshold: {fhe_spec.scoring.error_threshold}",
        f"  - Max Fatal Errors: {fhe_spec.scoring.max_fatal_errors}",
    ])

    if fhe_spec.template_files:
        lines.extend([
            "",
            "Template Files:",
        ])
        for f in fhe_spec.template_files:
            lines.append(f"  - {f}")

    if fhe_spec.useful_links:
        lines.extend([
            "",
            "Useful Resources:",
        ])
        for link in fhe_spec.useful_links:
            lines.append(f"  - {link['name']}: {link['url']}")

    return "\n".join(lines)


# Define FHE actions
FHE_ACTIONS = [
    ActionInfo(
        name="Execute FHE Solution",
        description="Execute an FHE solution file via Docker container. "
                    "The solution should implement the eval() function body for the challenge. "
                    "This action builds the solution with the challenge template and runs it "
                    "against the testcases, returning validation results including accuracy.",
        usage={
            "script_name": "Path to solution file (e.g., 'yourSolution.cpp' or 'solution.cpp')"
        },
        return_value="Validation results including accuracy, mean error, max error, and any build/runtime errors.",
        function=execute_fhe_solution,
        is_primitive=True
    ),
    ActionInfo(
        name="Validate FHE Output",
        description="Validate the output from the last FHE solution execution. "
                    "Use this to get detailed metrics about the solution's accuracy.",
        usage={},
        return_value="Detailed validation metrics including accuracy, error counts, and slot information.",
        function=validate_fhe_output,
        is_primitive=True
    ),
    ActionInfo(
        name="Understand FHE Challenge",
        description="Get detailed information about the current FHE challenge, including "
                    "the encryption scheme, constraints, available keys, and target accuracy.",
        usage={},
        return_value="Challenge specification including type, scheme, constraints, keys, and scoring parameters.",
        function=understand_fhe_challenge,
        is_primitive=True
    ),
]
