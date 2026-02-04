# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MLAgentBench is a benchmark suite for evaluating AI agents on machine learning experimentation tasks. Agents receive a dataset and task description, then autonomously develop or improve ML models.

## Common Commands

### Installation
```bash
pip install -e .
bash install.sh
```

### Run an Experiment
```bash
python -u -m MLAgentBench.runner --python $(which python) --task <task_name> --device 0 --log-dir <log_dir> --work-dir workspace --llm-name <model> --edit-script-llm-name <model> --fast-llm-name <model> > <log_dir>/log 2>&1
```

Key runner arguments:
- `--task`: Task name (matches benchmark folder or tasks.json entry)
- `--agent-type`: Agent class (`ResearchAgent`, `Agent`, `AutoGPTAgent`, `ReasoningActionAgent`, `LangChainAgent`)
- `--llm-name`, `--fast-llm-name`, `--edit-script-llm-name`: LLM models to use
- `--retrieval`: Enable retrieval-augmented mode for ResearchAgent
- `--max-steps`: Maximum agent steps (default 50)
- `--max-time`: Maximum time in seconds (default 5 hours)

### Prepare a Task Dataset
```bash
python -u -m MLAgentBench.prepare_task <task_name> $(which python)
```

### Run Baseline
```bash
python -u -m MLAgentBench.runner --python $(which python) --task <task_name> --device 0 --log-dir <log_dir> --work-dir workspace --agent_type Agent
```

### Evaluate Results
```bash
python -m MLAgentBench.eval --log-folder <log_folder> --task <task_name> --output-file <output_name>.json
```

### Parallel Experiments
Use `multi_run_experiment.sh` for running experiments across multiple devices:
```bash
bash multi_run_experiment.sh <exp_path> <task> <n_devices> <device_ids...> [extra_args]
```

## Architecture

### Core Components (`MLAgentBench/`)

- **runner.py**: Entry point that initializes environment and runs selected agent
- **environment.py**: `Environment` class manages task workspace, action execution, and trace logging
- **LLM.py**: Unified interface for LLM APIs (Claude, OpenAI, Gemini, HuggingFace, CRFM)
- **low_level_actions.py**: File operations and script execution primitives
- **high_level_actions.py**: LLM-powered actions (understand_file, edit_script, reflection)
- **eval.py**: Evaluation logic for scoring agent runs
- **plot.py**: Result visualization and analysis

### Agents (`MLAgentBench/agents/`)

- **agent.py**: Base `Agent` class (runs train.py + submits); also defines `SimpleActionAgent`, `ReasoningActionAgent`
- **agent_research.py**: `ResearchAgent` - main agent with research plan tracking, fact checking, and optional retrieval
- **agent_langchain.py**: LangChain-based agent wrapper
- **agent_autogpt.py**: AutoGPT integration

### Benchmarks (`MLAgentBench/benchmarks/`)

Each task folder contains:
- `env/`: Files visible to the agent at start (e.g., train.py, data)
- `scripts/`: Hidden files - `prepare.py` (data download), `eval.py` (evaluation), `research_problem.txt`, `read_only_files.txt`

Available tasks: cifar10, imdb, house-price, spaceship-titanic, feedback, identify-contrails, fathomnet, amp-parkinsons-disease-progression-prediction, CLRS, ogbn-arxiv, vectorization, llama-inference, babylm, bibtex-generation, literature-review-tool

### API Keys

Place API key files in the repository root:
- `openai_api_key.txt`: Format `organization:APIkey`
- `claude_api_key.txt`: Anthropic API key
- `crfm_api_key.txt`: CRFM API key
- Kaggle: `~/.kaggle/kaggle.json`

### Log Structure

Experiments produce logs in this structure:
```
<log_dir>/
  agent_log/
    main_log          # Agent's research process
    agent_*.json      # Saved agent states
  env_log/
    tool_logs/        # Per-step tool execution logs
    traces/           # Workspace snapshots per step
    trace.json        # Full interaction trace
```
