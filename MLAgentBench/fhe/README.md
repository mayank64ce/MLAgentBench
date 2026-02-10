# MLAgentBench FHE: Fully Homomorphic Encryption Challenge Support

This module extends MLAgentBench to support FHE (Fully Homomorphic Encryption) challenges, enabling AI agents to autonomously solve cryptographic challenges using Docker-based execution.

## Quick Start

```bash
# Basic usage
python -m MLAgentBench.runner \
    --challenge-dir /path/to/challenge \
    --log-dir logs/fhe \
    --work-dir workspace

# With custom model and timeouts
python -m MLAgentBench.runner \
    --challenge-dir /path/to/challenge \
    --llm-name gpt-4o-mini \
    --docker-timeout 600 \
    --docker-build-timeout 300 \
    --max-steps 20

# With specific agent type
python -m MLAgentBench.runner \
    --challenge-dir /path/to/challenge \
    --agent-type ResearchAgent \
    --retrieval
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MLAgentBench Runner                             │
│                  --challenge-dir=/path/to/challenge                  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        1. Challenge Parser                           │
│  challenge.md → FHEChallengeSpec (type, scheme, constraints, keys)  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     2. Environment Setup                             │
│  - Copy template files to workspace                                  │
│  - Generate research_problem from spec                               │
│  - Register FHE actions                                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     3. Interpreter Factory                           │
│  ChallengeType → BlackBoxInterpreter | WhiteBoxInterpreter          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      4. ResearchAgent Loop                           │
│  - Understand challenge (read files, understand constraints)         │
│  - Write solution code (yourSolution.cpp)                           │
│  - Execute FHE Solution action → Docker build/run/validate          │
│  - Analyze results, iterate                                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      5. Docker Execution                             │
│  Build → Run → Validate → Accuracy metrics                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      6. Feedback to Agent                            │
│  Observation: accuracy, errors, build output → next iteration       │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Challenge Parser (`challenge_parser.py`)

Parses `challenge.md` to extract:
- **Challenge Type**: `black_box`, `white_box_openfhe`, `ml_inference`, `non_openfhe`
- **Scheme**: CKKS, BFV, BGV, TFHE
- **Constraints**: Multiplicative depth, batch size, input range
- **Keys**: Available rotation indices, bootstrapping support
- **Scoring**: Accuracy threshold, error tolerance

```python
from MLAgentBench.fhe import parse_challenge

spec = parse_challenge("/path/to/challenge")
print(spec.challenge_type)  # ChallengeType.BLACK_BOX
print(spec.scheme)          # Scheme.CKKS
print(spec.constraints.depth)  # 12
```

### 2. Interpreters (`interpreters/`)

Type-specific Docker execution environments:

| Type | Interpreter | Description |
|------|-------------|-------------|
| `black_box` | `BlackBoxInterpreter` | Pre-encrypted testcases, uses challenge's Dockerfile |
| `white_box_openfhe` | `WhiteBoxInterpreter` | OpenFHE with fherma-validator image |
| `ml_inference` | `WhiteBoxInterpreter` | ML model + encrypted inference |

### 3. FHE Actions (`fhe_actions.py`)

Three actions available to agents when running FHE challenges:

| Action | Description |
|--------|-------------|
| `Execute FHE Solution` | Build and run solution via Docker, return validation results |
| `Validate FHE Output` | Re-validate last execution with detailed metrics |
| `Understand FHE Challenge` | Get challenge specification (scheme, constraints, keys) |

### 4. Code Injection

The agent writes solution code, which is injected into the template's `eval()` function:

```cpp
// Template (yourSolution.cpp)
void CKKSTaskSolver::eval() {
    // Agent's code is injected here
}

// Agent writes:
auto y = m_cc->EvalMult(m_InputC, 2.0);
m_OutputC = m_cc->EvalAdd(y, 1.0);
```

## Command Line Options

```bash
python -m MLAgentBench.runner \
    --challenge-dir /path/to/challenge \  # Path to FHE challenge directory
    --log-dir logs/fhe \                  # Log directory
    --work-dir workspace \                # Workspace directory
    --llm-name gpt-4o-mini \              # LLM model for agent
    --fast-llm-name gpt-4o-mini \         # Fast LLM for file reading
    --edit-script-llm-name gpt-4o-mini \  # LLM for code editing
    --docker-timeout 600 \                # Docker run timeout (seconds)
    --docker-build-timeout 300 \          # Docker build timeout (seconds)
    --max-steps 50 \                      # Maximum agent steps
    --agent-type ResearchAgent            # Agent class to use
```

## Output Files

After running, find results in `<log-dir>/`:

```
logs/fhe/
├── agent_log/
│   ├── main_log              # Agent's reasoning process
│   └── agent_*.json          # Saved agent states
└── env_log/
    ├── challenge_spec.json   # Parsed challenge specification
    ├── tool_logs/            # Per-step execution logs
    ├── traces/               # Workspace snapshots per step
    └── trace.json            # Full interaction trace
```

## Error Detection

The interpreter detects and categorizes errors:

| Error Type | Description |
|------------|-------------|
| `BUILD_ERROR` | CMake/make compilation failure |
| `RUNTIME_ERROR` | Exception during execution |
| `DEPTH_EXCEEDED` | Multiplicative depth budget exceeded |
| `TIMEOUT` | Execution exceeded time limit |

Error messages are extracted and returned to the agent as observations:
- Missing rotation keys: `"EvalKey for index [X] not found"`
- Depth exhausted: `"Depth exhausted - too many multiplicative operations"`
- Empty ciphertext: `"Ciphertext passed to Decrypt is empty"`

## Supported Challenge Types

### Black Box Challenges
- Pre-encrypted testcases provided in `tests/testcase1/`, `tests/testcase2/`
- Uses challenge's own Dockerfile
- Solution only accesses ciphertexts and crypto operations
- Verifier decrypts and validates output

### White Box OpenFHE Challenges
- Uses `yashalabinc/fherma-validator` Docker image
- Validator generates keys, encrypts inputs, runs solution, validates
- Parses `result.json` for accuracy metrics

## File Structure

```
MLAgentBench/fhe/
├── __init__.py              # Module exports
├── README.md                # This file
├── challenge_parser.py      # Parse challenge.md → FHEChallengeSpec
├── fhe_actions.py           # FHE-specific actions for agents
└── interpreters/
    ├── __init__.py          # Interpreter factory (create_interpreter)
    ├── base.py              # BaseInterpreter with Docker utilities
    ├── black_box.py         # Pre-encrypted testcase handling
    └── white_box.py         # OpenFHE with fherma-validator
```

## Example Workflow

1. **Run Command**
   ```bash
   python -m MLAgentBench.runner \
       --challenge-dir ../fhe_challenge/black_box/challenge_relu \
       --llm-name gpt-4o-mini \
       --max-steps 10
   ```

2. **Environment Setup**
   ```
   FHE Challenge: RELU Function
   Type: black_box
   Scheme: CKKS

   Workspace contains:
   - yourSolution.cpp (template)
   - yourSolution.h
   - challenge.md
   ```

3. **Agent Reads Challenge**
   ```
   Action: Understand FHE Challenge
   Observation:
     Challenge: RELU Function
     Type: black_box
     Scheme: CKKS
     Multiplicative Depth: 12
     Batch Size: 4096
     Input Range: [-1.0, 1.0]
     Target Accuracy: 0.8
   ```

4. **Agent Writes Solution**
   ```
   Action: Write File
   File: yourSolution.cpp
   Content: [eval() function body with polynomial approximation]
   ```

5. **Agent Executes Solution**
   ```
   Action: Execute FHE Solution
   Args: {"script_name": "yourSolution.cpp"}

   Observation:
     VALIDATION: PASSED
     Accuracy: 0.9234
     Mean error: 0.000823
     Max error: 0.012451
     Total slots: 4096
   ```

6. **Agent Iterates or Submits**
   ```
   Action: Final Answer
   Args: {"final_answer": "Achieved 92.34% accuracy using..."}
   ```

## Comparison with AIDE-FHE

| Feature | MLAgentBench FHE | AIDE-FHE |
|---------|------------------|----------|
| Agent Type | ResearchAgent (action-based) | Tree-search (draft/debug/improve) |
| Code Generation | Full file or eval() body | Only eval() body |
| Exploration | Sequential with research log | Parallel tree exploration |
| Actions | Standard + FHE actions | Code generation only |
| Use Case | General ML + FHE benchmarks | FHE-specialized |

## Integration with MLAgentBench

FHE support is fully integrated with MLAgentBench:

- **Backwards Compatible**: Existing `--task` parameter works unchanged
- **Same Agent Interface**: ResearchAgent works with FHE challenges
- **Standard Logging**: Uses existing trace/snapshot system
- **Action System**: FHE actions registered alongside standard actions

The agent can use all standard actions (Read File, Write File, Execute Script, etc.) plus the FHE-specific actions when working on FHE challenges.
