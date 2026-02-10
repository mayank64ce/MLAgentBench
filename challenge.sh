# # For sign challenge with gpt-4o-mini

# python -u -m MLAgentBench.runner \
#     --python $(which python) \
#     --challenge-dir ../fhe_challenge/black_box/challenge_sign \
#     --device 0 \
#     --log-dir logs/fhe_sign/gpt_4o_mini \
#     --work-dir workspace \
#     --llm-name gpt-4o-mini \
#     --fast-llm-name gpt-4o-mini \
#     --edit-script-llm-name gpt-4o-mini \
#     --docker-timeout 600 \
#     --docker-build-timeout 300 \
#     --agent-max-steps 10 \
#     --actions-remove-from-prompt "Copy File" "Execute Script" "Edit Script (AI)"

# # For relu challenge with gpt-4o-mini

# python -u -m MLAgentBench.runner \
#     --python $(which python) \
#     --challenge-dir ../fhe_challenge/black_box/challenge_relu \
#     --device 0 \
#     --log-dir logs/fhe_relu/gpt_4o_mini \
#     --work-dir workspace \
#     --llm-name gpt-4o-mini \
#     --fast-llm-name gpt-4o-mini \
#     --edit-script-llm-name gpt-4o-mini \
#     --docker-timeout 600 \
#     --docker-build-timeout 300 \
#     --agent-max-steps 10 \
#     --actions-remove-from-prompt "Copy File" "Execute Script" "Edit Script (AI)"

# # For sigmoid challenge with gpt-4o-mini

# python -u -m MLAgentBench.runner \
#     --python $(which python) \
#     --challenge-dir ../fhe_challenge/black_box/challenge_sigmoid \
#     --device 0 \
#     --log-dir logs/fhe_sigmoid/gpt_4o_mini \
#     --work-dir workspace \
#     --llm-name gpt-4o-mini \
#     --fast-llm-name gpt-4o-mini \
#     --edit-script-llm-name gpt-4o-mini \
#     --docker-timeout 600 \
#     --docker-build-timeout 300 \
#     --agent-max-steps 10 \
#     --actions-remove-from-prompt "Copy File" "Execute Script" "Edit Script (AI)"

# For sign challenge with gpt-5-mini-2025-08-07

python -u -m MLAgentBench.runner \
    --python $(which python) \
    --challenge-dir ../fhe_challenge/black_box/challenge_sign \
    --device 0 \
    --log-dir logs/fhe_sign/gpt-5-mini-2025-08-07 \
    --work-dir workspace \
    --llm-name gpt-5-mini-2025-08-07 \
    --fast-llm-name gpt-4o-mini \
    --edit-script-llm-name gpt-5-mini-2025-08-07 \
    --docker-timeout 600 \
    --docker-build-timeout 300 \
    --agent-max-steps 10 \
    --actions-remove-from-prompt "Copy File" "Execute Script" "Edit Script (AI)"

# For relu challenge with gpt-5-mini-2025-08-07

# python -u -m MLAgentBench.runner \
#     --python $(which python) \
#     --challenge-dir ../fhe_challenge/black_box/challenge_relu \
#     --device 0 \
#     --log-dir logs/fhe_relu/gpt-5-mini-2025-08-07 \
#     --work-dir workspace \
#     --llm-name gpt-5-mini-2025-08-07 \
#     --fast-llm-name gpt-4o-mini \
#     --edit-script-llm-name gpt-5-mini-2025-08-07 \
#     --docker-timeout 600 \
#     --docker-build-timeout 300 \
#     --agent-max-steps 10 \
#     --actions-remove-from-prompt "Copy File" "Execute Script" "Edit Script (AI)"

# For sigmoid challenge with gpt-5-mini-2025-08-07

# python -u -m MLAgentBench.runner \
#     --python $(which python) \
#     --challenge-dir ../fhe_challenge/black_box/challenge_sigmoid \
#     --device 0 \
#     --log-dir logs/fhe_sigmoid/gpt-5-mini-2025-08-07 \
#     --work-dir workspace \
#     --llm-name gpt-5-mini-2025-08-07 \
#     --fast-llm-name gpt-4o-mini \
#     --edit-script-llm-name gpt-5-mini-2025-08-07 \
#     --docker-timeout 600 \
#     --docker-build-timeout 300 \
#     --agent-max-steps 10 \
#     --actions-remove-from-prompt "Copy File" "Execute Script" "Edit Script (AI)"
