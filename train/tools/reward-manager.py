"""
Reward function for LGC-V2 environment in VERL.

This module provides the reward computation function that VERL's reward manager
will call to create reward tensors during training.
"""

from typing import Any, Dict, Optional

import os
import sys

# Ensure this directory is on sys.path so we can import sibling modules
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

from env_adapter import SimpleChallenge, get_env_adapter


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str],
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute reward score for a task, routed by env.
    
    This function is called by VERL's reward manager (e.g., NaiveRewardManager)
    for each model response. It evaluates the response against the challenge
    and returns 0.0 (incorrect) or 1.0 (correct).
    
    Args:
        data_source: Data source identifier. In many VERL datasets this is used as an
                    env/task identifier. In this repo we also pass `extra_info["env"]`
                    to route correctly.
        solution_str: The model's response text (decoded from token IDs)
        ground_truth: Optional ground truth (may be unused depending on env).
        extra_info: Dictionary containing challenge metadata:
            - "task_id": Task ID used to generate challenge (optional)
            - "env": env name ("lgc-v2", "trace") (recommended)
            - optional serialized challenge:
              - "prompt": str
              - "challenge_extra": dict
    
    Returns:
        float: Reward score (0.0 for incorrect, 1.0 for correct)
    """
    if extra_info is None:
        extra_info = {}
    
    env = (extra_info.get("env") or "").strip().lower() or "lgc-v2"
    adapter = get_env_adapter(env, config=extra_info.get("adapter_config"))

    # Prefer evaluating against a serialized challenge when available (needed for non-deterministic envs).
    prompt = extra_info.get("prompt")
    challenge_extra = extra_info.get("challenge_extra")

    try:
        if isinstance(prompt, str) and isinstance(challenge_extra, dict):
            ch = SimpleChallenge(env=env, prompt=prompt, extra=challenge_extra)
            score = adapter.evaluate(solution_str, ch)
            return float(score)

        # Fallback: regenerate from task_id if possible (works for deterministic envs like lgc-v2).
        task_id = extra_info.get("task_id")
        if task_id is None:
            # Log warning but don't fail - this is expected for some scenarios
            return 0.0
        
        ch = adapter.generate(int(task_id))
        score = adapter.evaluate(solution_str, ch)
        return float(score)
    except (ValueError, TypeError) as e:
        # Invalid input - log and return 0
        print(f"Invalid input for evaluation (env={env}, task_id={extra_info.get('task_id')}): {e}", file=__import__('sys').stderr)
        return 0.0
    except Exception as e:
        # Unexpected error - log with more context
        print(
            f"Error evaluating response (env={env}, task_id={extra_info.get('task_id')}, "
            f"data_source={data_source}): {e}",
            file=__import__('sys').stderr
        )
        return 0.0


# Alternative: Return dict for additional metrics
def compute_lgc_v2_score_with_metrics(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str],
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute reward score with additional metrics.
    
    Returns a dictionary with score and metadata, which can be useful
    for logging and analysis during training.
    
    Returns:
        dict: {
            "score": float,  # 0.0 or 1.0 (required by VERL)
            "task_type": str,  # Task type
            "task_id": int,  # Task ID
            "response_length": int,  # Length of response
        }
    """
    if extra_info is None:
        extra_info = {}
    
    env = (extra_info.get("env") or "").strip().lower() or "lgc-v2"
    score = compute_score(data_source, solution_str, ground_truth, extra_info)
    return {
        "score": float(score),
        "task_type": extra_info.get("task_type", data_source),
        "task_id": extra_info.get("task_id"),
        "response_length": len(solution_str),
        "env": env,
    }


# Backwards-compatible alias (older configs/scripts may refer to this name)
compute_lgc_v2_score = compute_score

