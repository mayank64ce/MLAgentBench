
"""
Challenge parser for FHE challenges.

Parses challenge.md to extract:
- Challenge type (black_box, white_box_openfhe, ml_inference, non_openfhe)
- Encryption scheme (CKKS, BFV, BGV)
- Constraints (depth, batch size, input range)
- Available keys
- Task specification
- Scoring parameters

Adapted from AIDE-FHE for MLAgentBench.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any


class ChallengeType(str, Enum):
    """Types of FHE challenges."""
    BLACK_BOX = "black_box"
    WHITE_BOX_OPENFHE = "white_box_openfhe"
    ML_INFERENCE = "ml_inference"
    NON_OPENFHE = "non_openfhe"


class Scheme(str, Enum):
    """FHE encryption schemes."""
    CKKS = "CKKS"
    BFV = "BFV"
    BGV = "BGV"
    TFHE = "TFHE"


class Library(str, Enum):
    """FHE libraries."""
    OPENFHE = "OpenFHE"
    HELAYERS = "HElayers"
    SWIFT_HE = "swift-homomorphic-encryption"
    SEAL = "SEAL"


@dataclass
class Constraints:
    """FHE constraints from challenge specification."""
    depth: int = 10
    batch_size: int = 4096
    scale_mod_size: int = 50
    first_mod_size: int = 60
    ring_dimension: Optional[int] = None
    input_range: Tuple[float, float] = (-1.0, 1.0)
    plaintext_modulus: Optional[int] = None
    security_level: str = "HEStd_128_classic"


@dataclass
class Keys:
    """Available keys from challenge specification."""
    public: bool = True
    secret: bool = False
    multiplication: bool = True
    rotation_indices: List[int] = field(default_factory=list)
    bootstrapping: bool = False


@dataclass
class Scoring:
    """Scoring specification."""
    metric_type: str = "accuracy"
    error_threshold: float = 0.001
    accuracy_threshold: float = 0.8
    max_fatal_errors: int = 40
    score_per_slot: float = 10.0


@dataclass
class FHEChallengeSpec:
    """Complete FHE challenge specification."""

    # Challenge identity
    challenge_id: Optional[str] = None
    challenge_name: Optional[str] = None
    challenge_dir: Optional[Path] = None

    # Challenge type
    challenge_type: ChallengeType = ChallengeType.WHITE_BOX_OPENFHE

    # Encryption
    scheme: Scheme = Scheme.CKKS
    library: Library = Library.OPENFHE
    library_version: Optional[str] = None

    # Constraints
    constraints: Constraints = field(default_factory=Constraints)

    # Keys
    keys: Keys = field(default_factory=Keys)

    # Task
    task: str = "unknown"
    task_description: str = ""
    function_signature: str = ""

    # Input/Output
    input_format: str = ""
    output_format: str = ""
    num_inputs: int = 1
    num_outputs: int = 1

    # Scoring
    scoring: Scoring = field(default_factory=Scoring)

    # Template info
    template_dir: Optional[Path] = None
    template_files: List[str] = field(default_factory=list)
    has_dockerfile: bool = False
    has_verifier: bool = False

    # Testcase info
    testcase_dirs: List[Path] = field(default_factory=list)
    has_test_case_json: bool = False

    # Raw text
    raw_text: str = ""

    # Useful links
    useful_links: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "challenge_id": self.challenge_id,
            "challenge_name": self.challenge_name,
            "challenge_dir": str(self.challenge_dir) if self.challenge_dir else None,
            "challenge_type": self.challenge_type.value,
            "scheme": self.scheme.value,
            "library": self.library.value,
            "library_version": self.library_version,
            "constraints": {
                "depth": self.constraints.depth,
                "batch_size": self.constraints.batch_size,
                "scale_mod_size": self.constraints.scale_mod_size,
                "first_mod_size": self.constraints.first_mod_size,
                "ring_dimension": self.constraints.ring_dimension,
                "input_range": list(self.constraints.input_range),
                "plaintext_modulus": self.constraints.plaintext_modulus,
                "security_level": self.constraints.security_level,
            },
            "keys": {
                "public": self.keys.public,
                "secret": self.keys.secret,
                "multiplication": self.keys.multiplication,
                "rotation_indices": self.keys.rotation_indices,
                "bootstrapping": self.keys.bootstrapping,
            },
            "task": self.task,
            "task_description": self.task_description,
            "function_signature": self.function_signature,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "scoring": {
                "metric_type": self.scoring.metric_type,
                "error_threshold": self.scoring.error_threshold,
                "accuracy_threshold": self.scoring.accuracy_threshold,
                "max_fatal_errors": self.scoring.max_fatal_errors,
            },
            "template_dir": str(self.template_dir) if self.template_dir else None,
            "template_files": self.template_files,
            "has_dockerfile": self.has_dockerfile,
            "has_verifier": self.has_verifier,
            "testcase_dirs": [str(d) for d in self.testcase_dirs],
            "useful_links": self.useful_links,
        }

    def save(self, path: Path) -> None:
        """Save spec to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def generate_research_problem(self) -> str:
        """Generate research problem text from spec for MLAgentBench."""
        lines = [
            f"# FHE Challenge: {self.challenge_name or self.task}",
            "",
            self.task_description or f"Implement the {self.task} function using Fully Homomorphic Encryption.",
            "",
            "## Technical Details",
            f"- Encryption Scheme: {self.scheme.value}",
            f"- Library: {self.library.value}",
            f"- Multiplicative Depth Budget: {self.constraints.depth}",
            f"- Batch Size: {self.constraints.batch_size}",
            f"- Input Range: [{self.constraints.input_range[0]}, {self.constraints.input_range[1]}]",
            "",
            "## Available Keys",
            f"- Public Key: {'Yes' if self.keys.public else 'No'}",
            f"- Multiplication Key: {'Yes' if self.keys.multiplication else 'No'}",
        ]

        if self.keys.rotation_indices:
            lines.append(f"- Rotation Keys: {self.keys.rotation_indices}")

        lines.extend([
            "",
            "## Your Task",
            "1. Read the template files in the workspace",
            "2. Implement the eval() function body in yourSolution.cpp",
            "3. Use the 'Execute FHE Solution' action to test your implementation",
            "4. Iterate until you achieve the target accuracy",
            "",
            f"## Target Accuracy: {self.scoring.accuracy_threshold}",
            f"## Error Threshold: {self.scoring.error_threshold}",
        ])

        if self.useful_links:
            lines.extend(["", "## Useful Resources"])
            for link in self.useful_links:
                lines.append(f"- [{link['name']}]({link['url']})")

        return "\n".join(lines)


def parse_challenge(challenge_path: Path | str) -> FHEChallengeSpec:
    """
    Parse challenge.md and directory structure to create FHEChallengeSpec.

    Args:
        challenge_path: Path to challenge directory or challenge.md file

    Returns:
        FHEChallengeSpec with parsed challenge information
    """
    challenge_path = Path(challenge_path)

    if challenge_path.is_file():
        challenge_dir = challenge_path.parent
        challenge_file = challenge_path
    else:
        challenge_dir = challenge_path
        challenge_file = challenge_path / "challenge.md"

    if not challenge_file.exists():
        raise FileNotFoundError(f"challenge.md not found: {challenge_file}")

    text = challenge_file.read_text()
    spec = FHEChallengeSpec(raw_text=text, challenge_dir=challenge_dir)

    # Parse all components
    _parse_metadata(text, spec)
    _parse_challenge_type(text, spec)
    _parse_scheme(text, spec)
    _parse_library(text, spec)
    _parse_constraints(text, spec)
    _parse_keys(text, spec)
    _parse_task(text, spec)
    _parse_scoring(text, spec)
    _parse_useful_links(text, spec)
    _parse_directory_structure(challenge_dir, spec)

    return spec


def _parse_metadata(text: str, spec: FHEChallengeSpec) -> None:
    """Extract challenge name and ID."""
    title_match = re.search(r'^#\s+(.+?)(?:\n|$)', text, re.MULTILINE)
    if title_match:
        spec.challenge_name = title_match.group(1).strip()

    id_match = re.search(r'challenge_id[:\s]+([a-f0-9]+)', text, re.IGNORECASE)
    if id_match:
        spec.challenge_id = id_match.group(1)


def _parse_challenge_type(text: str, spec: FHEChallengeSpec) -> None:
    """Detect challenge type from challenge.md content."""
    text_lower = text.lower()

    # Check for non-OpenFHE libraries first
    if re.search(r'helayers|ibm\s+fhe|pyhelayers', text_lower):
        spec.challenge_type = ChallengeType.NON_OPENFHE
        return

    if re.search(r'swift-homomorphic-encryption|apple.*swift.*homomorphic|swift\s+he', text_lower):
        spec.challenge_type = ChallengeType.NON_OPENFHE
        return

    # Check for explicit type declaration
    explicit_type_match = re.search(r'challenge\s+type[:\s]+([^\n]+)', text_lower)
    if explicit_type_match:
        type_text = explicit_type_match.group(1).strip()
        if 'black' in type_text and 'box' in type_text:
            spec.challenge_type = ChallengeType.BLACK_BOX
            return

    # Check for ML-specific patterns
    ml_patterns = [
        r'cifar[-\s]?10', r'mnist', r'sentiment\s+(analysis|classification)',
        r'fraud\s+detection', r'house\s+(price\s+)?prediction',
        r'svm\s+(model|classification|fraud)', r'training\s+data\s+(is\s+)?provided',
    ]
    for pattern in ml_patterns:
        if re.search(pattern, text_lower):
            spec.challenge_type = ChallengeType.ML_INFERENCE
            return

    # Check for white box declaration
    if explicit_type_match:
        type_text = explicit_type_match.group(1).strip()
        if 'white' in type_text and 'box' in type_text:
            spec.challenge_type = ChallengeType.WHITE_BOX_OPENFHE
            return

    # Check for black box indicators
    black_box_patterns = [
        r'black\s*box', r'pre-encrypted\s+test', r'testcase.*pre-encrypted',
        r'encrypted\s+input\s+is\s+provided', r'ciphertext\s+is\s+already\s+provided',
    ]
    for pattern in black_box_patterns:
        if re.search(pattern, text_lower):
            spec.challenge_type = ChallengeType.BLACK_BOX
            return

    # Default to white box
    spec.challenge_type = ChallengeType.WHITE_BOX_OPENFHE


def _parse_scheme(text: str, spec: FHEChallengeSpec) -> None:
    """Extract encryption scheme."""
    text_upper = text.upper()

    if "CKKS" in text_upper:
        spec.scheme = Scheme.CKKS
    elif "BFV" in text_upper:
        spec.scheme = Scheme.BFV
    elif "BGV" in text_upper:
        spec.scheme = Scheme.BGV
    elif "TFHE" in text_upper or "BOOLEAN" in text_upper:
        spec.scheme = Scheme.TFHE


def _parse_library(text: str, spec: FHEChallengeSpec) -> None:
    """Extract FHE library and version."""
    text_lower = text.lower()

    if re.search(r'helayers|pyhelayers', text_lower):
        spec.library = Library.HELAYERS
    elif re.search(r'swift-homomorphic-encryption|apple.*swift', text_lower):
        spec.library = Library.SWIFT_HE
    elif re.search(r'\bseal\b', text_lower):
        spec.library = Library.SEAL
    else:
        spec.library = Library.OPENFHE

    if spec.library == Library.OPENFHE:
        version_match = re.search(r'openfhe[:\s]*v?(\d+\.\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if version_match:
            spec.library_version = version_match.group(1)


def _parse_constraints(text: str, spec: FHEChallengeSpec) -> None:
    """Extract crypto constraints."""
    constraints = spec.constraints

    # Multiplicative depth
    for pattern in [
        r'[Mm]ultiplicative\s+depth[:\s]+(\d+)',
        r'depth[:\s]+(\d+)',
        r'depth\s*(?:budget|limit)?[:\s]*(\d+)',
    ]:
        match = re.search(pattern, text)
        if match:
            constraints.depth = int(match.group(1))
            break

    # Batch size
    for pattern in [
        r'"batch_size":\s*(\d+)',
        r'batch_size\s*[=:]\s*(\d+)',
        r'[Bb]atch\s+[Ss]ize:\s*(\d+)',
        r'[Vv]ector\s+length[:\s]+(\d+)',
    ]:
        match = re.search(pattern, text)
        if match:
            val = int(match.group(1))
            if val >= 1024:
                constraints.batch_size = val
                break

    # Scale mod size
    match = re.search(r'[Ss]cale[Mm]od[Ss]ize[:\s]+(\d+)', text)
    if match:
        constraints.scale_mod_size = int(match.group(1))

    # Ring dimension
    match = re.search(r'[Rr]ing\s+dimension[:\s]+(\d+)', text)
    if match:
        constraints.ring_dimension = int(match.group(1))

    # Input range
    match = re.search(r'[Rr]ange[:\s]*\[([^\]]+)\]', text)
    if match:
        try:
            parts = [float(x.strip()) for x in match.group(1).split(',')]
            if len(parts) == 2:
                constraints.input_range = tuple(parts)
        except ValueError:
            pass


def _parse_keys(text: str, spec: FHEChallengeSpec) -> None:
    """Extract available keys."""
    keys = spec.keys

    keys.public = bool(re.search(r'public\s+key', text, re.IGNORECASE))
    keys.multiplication = bool(re.search(r'multiplication|relinearization|key_mult', text, re.IGNORECASE))
    keys.bootstrapping = bool(re.search(r'bootstrap', text, re.IGNORECASE))

    rot_match = re.search(r'rotation\s+key[^[]*\[([^\]]+)\]', text, re.IGNORECASE)
    if rot_match:
        try:
            indices = [int(x.strip()) for x in rot_match.group(1).split(',')]
            keys.rotation_indices = indices
        except ValueError:
            pass


def _parse_task(text: str, spec: FHEChallengeSpec) -> None:
    """Identify the computational task."""
    text_lower = text.lower()

    # For ML inference, use folder name
    if spec.challenge_type == ChallengeType.ML_INFERENCE:
        if spec.challenge_dir:
            folder_name = spec.challenge_dir.name
            spec.task = folder_name.replace('challenge_', '')
        elif spec.challenge_name:
            spec.task = re.sub(r'[^a-z0-9]+', '_', spec.challenge_name.lower()).strip('_')[:50]
        else:
            spec.task = "ml_inference"
        spec.function_signature = "model(encrypted_input)"
        return

    # Try matching title first
    title = (spec.challenge_name or "").lower()

    task_patterns = [
        (r'\bgelu\b', "gelu", "gelu(x)"),
        (r'\brelu\b', "relu", "max(0, x)"),
        (r'\bsoftmax\b', "softmax", "softmax(x)"),
        (r'\bsigmoid\b|logistic', "sigmoid", "1/(1+exp(-x))"),
        (r'\bsign\b', "sign", "sign(x)"),
        (r'\btanh\b', "tanh", "tanh(x)"),
        (r'singular\s+value|svd', "svd", "svd(A)"),
        (r'invertible\s+matrix', "invertible_matrix", "det(A) != 0"),
        (r'matrix\s+mult', "matrix_multiplication", "A @ B"),
        (r'array\s+sort', "array_sorting", "sort(x)"),
        (r'max\s+element', "max", "max(x)"),
        (r'k-?nearest|knn', "knn", "knn(x, k)"),
        (r'lookup\s+table', "lookup_table", "table[idx]"),
        (r'\bshl\b|shift\s+left', "shl", "x << n"),
        (r'\bparity\b', "parity", "parity(x)"),
        (r'string\s+search', "string_search", "find(str, text)"),
    ]

    # Check title
    for pattern, task, signature in task_patterns:
        if re.search(pattern, title):
            spec.task = task
            spec.function_signature = signature
            break
    else:
        # Check full text
        for pattern, task, signature in task_patterns:
            if re.search(pattern, text_lower):
                spec.task = task
                spec.function_signature = signature
                break
        else:
            spec.task = "custom"
            spec.function_signature = "f(x)"

    # Extract task description
    intro_match = re.search(
        r'##\s*Introduction\s*\n(.*?)(?=\n##|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if intro_match:
        spec.task_description = intro_match.group(1).strip()[:1000]


def _parse_scoring(text: str, spec: FHEChallengeSpec) -> None:
    """Extract scoring parameters."""
    scoring = spec.scoring

    match = re.search(r'(?:error|threshold)[:\s<]+(\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        scoring.error_threshold = float(match.group(1))

    match = re.search(r'(?:min|minimum)\s*(?:slot\s*)?accuracy[:\s]+(\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        scoring.accuracy_threshold = val / 100 if val > 1 else val

    match = re.search(r'(?:fatal|allowed)\s*(?:_)?error[s]?[:\s]+(\d+)', text, re.IGNORECASE)
    if match:
        scoring.max_fatal_errors = int(match.group(1))


def _parse_useful_links(text: str, spec: FHEChallengeSpec) -> None:
    """Extract useful links section."""
    links_match = re.search(
        r'##\s*[Uu]seful\s+[Ll]inks?\s*\n(.*?)(?=\n##|\Z)',
        text, re.DOTALL
    )
    if not links_match:
        return

    links_section = links_match.group(1)
    link_pattern = r'-\s*\[([^\]]+)\]\(([^)]+)\)([^\n]*)'

    for match in re.finditer(link_pattern, links_section):
        link_name = match.group(1).strip().rstrip(':')
        link_url = match.group(2).strip()
        desc_raw = match.group(3).strip()

        link_desc = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', desc_raw)
        link_desc = re.sub(r'^[\s-]+', '', link_desc).strip()

        if any(skip in link_url.lower() for skip in ['fherma.io/how_it_works', 'fhe.org/resources']):
            continue

        spec.useful_links.append({
            "name": link_name,
            "url": link_url,
            "description": link_desc
        })


def _parse_directory_structure(challenge_dir: Path, spec: FHEChallengeSpec) -> None:
    """Analyze challenge directory structure."""
    # Find template directory
    template_candidates = [
        challenge_dir / "templates" / "openfhe",
        challenge_dir / "templates" / "helayers",
        challenge_dir / "templates" / "swift",
        challenge_dir / "templates" / "openfhe-python",
    ]

    for template_dir in template_candidates:
        if template_dir.exists():
            spec.template_dir = template_dir
            spec.template_files = [f.name for f in template_dir.iterdir() if f.is_file()]
            break

    # Check for Dockerfile
    dockerfile = challenge_dir / "Dockerfile"
    spec.has_dockerfile = dockerfile.exists()

    # Check for verifier
    verifier = challenge_dir / "verifier.cpp"
    spec.has_verifier = verifier.exists()

    # Find testcases
    tests_dir = challenge_dir / "tests"
    if tests_dir.exists():
        for d in tests_dir.iterdir():
            if d.is_dir() and d.name.startswith("testcase"):
                spec.testcase_dirs.append(d)

        test_case_json = tests_dir / "test_case.json"
        spec.has_test_case_json = test_case_json.exists()
