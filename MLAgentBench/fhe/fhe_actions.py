"""
FHE-specific actions for MLAgentBench.

These actions are registered when running FHE challenges and provide
Docker-based execution, validation, and focused eval() code generation.
"""

import os
import re
import difflib
import datetime
import shutil
from pathlib import Path
from functools import wraps

from ..schema import ActionInfo, EnvException
from ..LLM import complete_text
from .. import high_level_actions


def _get_fhe_context(kwargs):
    """Extract FHE-related context from kwargs."""
    return {
        "fhe_spec": kwargs.get("fhe_spec"),
        "fhe_interpreter": kwargs.get("fhe_interpreter"),
        "work_dir": kwargs.get("work_dir", "."),
        "trace": kwargs.get("trace"),
    }


def _extract_template_variables(header_content):
    """
    Parse a C++ header file to extract FHE variable names.

    Looks for member variable declarations like:
        CryptoContext<DCRTPoly> m_cc;
        Ciphertext<DCRTPoly> m_InputC;

    Returns:
        dict with keys: 'context', 'inputs', 'output', 'public_key'
    """
    result = {
        'context': 'm_cc',
        'inputs': ['m_InputC'],
        'output': 'm_OutputC',
        'public_key': 'm_PublicKey',
    }

    if not header_content:
        return result

    # Extract CryptoContext variable
    cc_match = re.search(r'CryptoContext<DCRTPoly>\s+(m_\w+)', header_content)
    if cc_match:
        result['context'] = cc_match.group(1)

    # Extract Ciphertext member variables
    ciphertext_vars = re.findall(r'Ciphertext<DCRTPoly>\s+(m_\w+)', header_content)
    if ciphertext_vars:
        inputs = []
        output = None
        for var in ciphertext_vars:
            if 'Output' in var or 'output' in var:
                output = var
            else:
                inputs.append(var)
        if inputs:
            result['inputs'] = inputs
        if output:
            result['output'] = output

    # Extract PublicKey variable
    pk_match = re.search(r'PublicKey<DCRTPoly>\s+(m_\w+)', header_content)
    if pk_match:
        result['public_key'] = pk_match.group(1)

    return result


def _extract_eval_body(cpp_content):
    """
    Extract the current eval() function body from a C++ source file.

    Uses brace-matching to find the body of void ClassName::eval() { ... }.

    Returns:
        The body text (stripped), or empty string if eval() is empty/placeholder.
    """
    if not cpp_content:
        return ""

    match = re.search(r'void\s+\w+::eval\s*\(\s*\)\s*\{', cpp_content)
    if not match:
        return ""

    start = match.end()
    depth = 1
    pos = start
    while pos < len(cpp_content) and depth > 0:
        if cpp_content[pos] == '{':
            depth += 1
        elif cpp_content[pos] == '}':
            depth -= 1
        pos += 1

    if depth != 0:
        return ""

    body = cpp_content[start:pos - 1].strip()

    # Check if body is just comments or placeholder
    stripped = re.sub(r'//[^\n]*', '', body).strip()
    stripped = re.sub(r'/\*.*?\*/', '', stripped, flags=re.DOTALL).strip()
    if not stripped or stripped.lower() in ('todo', ''):
        return ""

    return body


def _format_variable_docs(template_vars):
    """
    Format template variable documentation for the LLM prompt.

    Args:
        template_vars: dict from _extract_template_variables()

    Returns:
        Formatted string describing variable names and usage.
    """
    lines = [
        "IMPORTANT - Use these EXACT variable names (class members):",
        f"  {template_vars['context']}       - CryptoContext<DCRTPoly>",
    ]

    if len(template_vars['inputs']) == 1:
        lines.append(f"  {template_vars['inputs'][0]}   - Input Ciphertext<DCRTPoly>")
    else:
        for inp in template_vars['inputs']:
            lines.append(f"  {inp}  - Input Ciphertext<DCRTPoly>")

    lines.extend([
        f"  {template_vars['output']}  - Output Ciphertext (ASSIGN to this, don't return)",
        f"  {template_vars['public_key']} - PublicKey<DCRTPoly>",
        "",
        f"The eval() function is void - assign result to {template_vars['output']}:",
        f"  {template_vars['output']} = result;  // CORRECT",
        "  return result;       // WRONG - eval() is void!",
    ])

    return "\n".join(lines)


EDIT_SCRIPT_MAX_TOKENS = 4000


def implement_fhe_eval(edit_instruction, work_dir=".", **kwargs):
    """
    Generate or edit the eval() function body for an FHE challenge.

    This action sends only relevant context (challenge spec, variable docs,
    current eval body) to the LLM and receives only the eval() function body
    back, rather than sending/receiving the entire file.

    Args:
        edit_instruction: Description of what the eval() function should implement
        work_dir: Working directory containing yourSolution.cpp/h
        **kwargs: Additional context including fhe_spec, fhe_interpreter, log_file

    Returns:
        Observation string with the generated code diff
    """
    fhe_spec = kwargs.get("fhe_spec")
    interpreter = kwargs.get("fhe_interpreter")

    if fhe_spec is None:
        raise EnvException("FHE spec not available. This action is only available for FHE challenges.")

    # 1. Read template files from workspace
    solution_path = Path(work_dir) / "yourSolution.cpp"
    header_path = Path(work_dir) / "yourSolution.h"

    if not solution_path.exists():
        raise EnvException("yourSolution.cpp not found in workspace.")

    cpp_content = solution_path.read_text()
    header_content = header_path.read_text() if header_path.exists() else ""

    # 2. Extract template variables and current eval body
    template_vars = _extract_template_variables(header_content)
    current_eval_body = _extract_eval_body(cpp_content)
    variable_docs = _format_variable_docs(template_vars)

    # 3. Build focused prompt
    spec = fhe_spec
    prompt_parts = [
        "You are an expert in Fully Homomorphic Encryption (FHE).",
        "Implement the eval() function body for the following challenge.",
        "",
        "## Challenge Specification",
        f"- Task: {spec.task}",
    ]

    if spec.task_description:
        prompt_parts.append(f"- Description: {spec.task_description}")

    prompt_parts.extend([
        f"- Scheme: {spec.scheme.value}",
        f"- Library: {spec.library.value}",
        f"- Multiplicative Depth Budget: {spec.constraints.depth}",
        f"- Batch Size: {spec.constraints.batch_size}",
        f"- Input Range: [{spec.constraints.input_range[0]}, {spec.constraints.input_range[1]}]",
        "",
        "## Available Keys",
        f"- Public Key: {'Yes' if spec.keys.public else 'No'}",
        f"- Multiplication Key: {'Yes' if spec.keys.multiplication else 'No'}",
    ])

    if spec.keys.rotation_indices:
        prompt_parts.append(f"- Rotation Keys: {spec.keys.rotation_indices}")

    prompt_parts.extend([
        "",
        "## Template Variables",
        variable_docs,
    ])

    if header_content:
        prompt_parts.extend([
            "",
            "## Template File: yourSolution.h",
            "```cpp",
            header_content.strip(),
            "```",
        ])

    if current_eval_body:
        prompt_parts.extend([
            "",
            "## Current eval() body:",
            "```cpp",
            current_eval_body,
            "```",
        ])
    else:
        prompt_parts.extend([
            "",
            "## Current eval() body:",
            "```cpp",
            "// Empty - no implementation yet",
            "```",
        ])

    prompt_parts.extend([
        "",
        "## Instruction",
        edit_instruction,
        "",
        "## CRITICAL Rules",
        "- NEVER decrypt inputs or use secret keys",
        f"- NEVER return from eval() - assign result to {template_vars['output']}",
        f"- Stay within multiplicative depth budget of {spec.constraints.depth}",
    ])

    if spec.keys.rotation_indices:
        prompt_parts.append(f"- Only use rotation indices that are available: {spec.keys.rotation_indices}")

    prompt_parts.extend([
        "",
        "## Response Format",
        "Provide ONLY the eval() function body (no function signature, no braces).",
        "Start the C++ code with ```cpp and end with ```.",
    ])

    prompt = "\n".join(prompt_parts)

    # 4. Call LLM
    completion = complete_text(
        prompt,
        log_file=kwargs.get("log_file", os.path.join(work_dir, "implement_fhe_eval.log")),
        model=high_level_actions.EDIT_SCRIPT_MODEL,
        max_tokens_to_sample=EDIT_SCRIPT_MAX_TOKENS,
    )

    # 5. Extract code from response
    new_code = None
    if "```cpp" in completion:
        new_code = completion.split("```cpp")[1].split("```")[0].strip()
    elif "```c++" in completion:
        new_code = completion.split("```c++")[1].split("```")[0].strip()
    elif "```" in completion:
        new_code = completion.split("```")[1].split("```")[0].strip()

    if not new_code:
        raise EnvException("LLM did not return valid C++ code in the expected format.")

    # 6. Backup old file
    backup_name = os.path.join(
        work_dir, "backup",
        f"yourSolution.cpp_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    backup_dir = os.path.join(work_dir, "backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    shutil.copyfile(str(solution_path), backup_name)

    # 7. Inject code into yourSolution.cpp
    if interpreter is not None:
        interpreter._inject_code(solution_path, new_code)
    else:
        # Fallback: use the same injection logic directly
        from .interpreters.base import BaseInterpreter
        # Read template, do brace-matching injection
        template = solution_path.read_text()
        stripped_code = template  # Will be overwritten below

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
                stripped_code = template[:start] + '\n' + new_code + '\n' + template[pos - 1:]
                solution_path.write_text(stripped_code)
            else:
                solution_path.write_text(new_code)
        else:
            solution_path.write_text(new_code)

    # 8. Generate and return diff
    new_content = solution_path.read_text()
    diff = list(difflib.unified_diff(
        cpp_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile="yourSolution.cpp (before)",
        tofile="yourSolution.cpp (after)",
    ))
    diff_str = "".join(diff)

    return (
        f"The eval() function has been implemented in yourSolution.cpp. "
        f"Here is the diff:\n\n{diff_str}"
    )


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
        name="Implement FHE Eval",
        description="Generate or edit the eval() function body for the FHE challenge. "
                    "Provide a description of what the eval function should compute. "
                    "The LLM will receive challenge constraints, variable documentation, "
                    "and template context to generate only the eval() body. "
                    "Use this instead of 'Edit Script (AI)' for FHE C++ challenges.",
        usage={
            "edit_instruction": "Description of what the eval() function should implement "
                               "(e.g., 'implement ReLU using polynomial approximation with depth 10')"
        },
        return_value="The generated code and a diff showing changes to yourSolution.cpp.",
        function=implement_fhe_eval,
        is_primitive=False
    ),
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
