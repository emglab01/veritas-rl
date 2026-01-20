"""
Add reasoning to dataset using OpenAI.

This script reads a dataset JSONL file and generates reasoning for each sample
using OpenAI. The reasoning is combined with the original answer and placed
in the "answer" field. The answer field contains: reasoning + final answer.

Usage:
    python data_processing/add_reasoning.py \
      --input data_processing/dataset/swe-synth.jsonl \
      --output data_processing/dataset/swe-synth-with-reasoning.jsonl \
      --model gpt-4o-mini \
      --max-retries 3
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv is optional, but recommended
    load_dotenv = None
    import warnings
    warnings.warn(
        "python-dotenv not installed. .env file will not be loaded. "
        "Install with: pip install python-dotenv or activate virtual environment.",
        UserWarning
    )

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai package not installed. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)

# Load .env file if it exists (doesn't override existing env vars)
if load_dotenv is not None:
    # Try loading from project root
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Fallback to default location (current directory)
        load_dotenv()

from validation import validate_file_path, validate_directory_path

# Constants
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 60
PROGRESS_INTERVAL = 10
RATE_LIMIT_RETRY_DELAY = 5  # seconds


def detect_language(prompt: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Detect the language of the prompt.
    
    Args:
        prompt: The problem/prompt text
        metadata: Optional metadata dict that may contain language info
    
    Returns:
        str: "chinese" or "english"
    """
    # First check metadata for language field
    if metadata and isinstance(metadata, dict):
        lang = metadata.get("language", "").lower()
        if lang in ("chinese", "zh", "zh-cn", "zh-tw"):
            return "chinese"
        if lang in ("english", "en"):
            return "english"
    
    # Detect by checking for Chinese characters
    # Chinese characters are typically in the range 0x4E00-0x9FFF
    chinese_char_count = sum(1 for char in prompt[:500] if '\u4e00' <= char <= '\u9fff')
    total_chars = len([c for c in prompt[:500] if c.isalnum() or '\u4e00' <= c <= '\u9fff'])
    
    # If more than 10% of characters are Chinese, consider it Chinese
    if total_chars > 0 and chinese_char_count / total_chars > 0.1:
        return "chinese"
    
    # Default to English
    return "english"


def get_openai_client() -> AsyncOpenAI:
    """Get OpenAI client from environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Check if .env file exists and provide helpful message
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        env_hint = ""
        if env_file.exists():
            env_hint = (
                f"\nNote: .env file exists at {env_file}, but OPENAI_API_KEY was not loaded. "
                "Make sure to:\n"
                "  1. Activate the virtual environment: source .venv/bin/activate\n"
                "  2. Or install python-dotenv: pip install python-dotenv\n"
                "  3. Or set the environment variable: export OPENAI_API_KEY=your_key_here"
            )
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it with: export OPENAI_API_KEY=your_key_here"
            + env_hint
        )
    return AsyncOpenAI(api_key=api_key)


async def generate_reasoning(
    client: AsyncOpenAI,
    prompt: str,
    original_answer: str,
    model: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    language: str = "english",
) -> str:
    """
    Generate reasoning using OpenAI that explains how to arrive at the answer.
    The reasoning will be generated in the same language as the problem.
    
    Args:
        client: OpenAI async client
        prompt: The problem/prompt text
        original_answer: The original answer (final solution)
        model: OpenAI model to use
        max_retries: Maximum number of retries on failure
        timeout: Request timeout in seconds
        language: Language of the prompt ("chinese" or "english")
    
    Returns:
        str: Generated reasoning text in the same language as the problem
    """
    # Generate prompts in the appropriate language
    if language == "chinese":
        system_prompt = """你是一个有用的助手，能够为解决问题提供逐步的推理过程。
你的任务是提供清晰、逻辑严密的推理，引导出给定的答案。
推理应该：
1. 将问题分解为步骤
2. 清楚地解释每个步骤
3. 展示步骤如何导向最终答案
4. 简洁但完整

请以纯文本格式提供推理，最后清楚地标记最终答案。"""
        
        user_prompt = f"""问题：
{prompt}

最终答案：
{original_answer}

请提供逐步推理，解释如何得出这个答案。
推理应该清晰且逻辑严密，逐步展示思考过程。
在推理的最后清楚地说明最终答案。"""
    else:
        # English (default)
        system_prompt = """You are a helpful assistant that explains step-by-step reasoning for solving problems.
Your task is to provide clear, logical reasoning that leads to the given answer.
The reasoning should:
1. Break down the problem into steps
2. Explain each step clearly
3. Show how the steps lead to the final answer
4. Be concise but complete

Format your response as plain text reasoning, ending with the final answer clearly marked."""
        
        user_prompt = f"""Problem:
{prompt}

Final Answer:
{original_answer}

Please provide step-by-step reasoning that explains how to arrive at this answer. 
The reasoning should be clear and logical, showing the thought process step by step.
End your reasoning with a clear statement of the final answer."""

    # Some models don't support custom temperature (e.g., gpt-5)
    # Try with temperature first, fall back to default if it fails
    use_temperature = True
    temperature_value = 0.3
    
    for attempt in range(max_retries):
        try:
            request_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            
            # Only add temperature if it's supported (skip on retry if we got temperature error)
            if use_temperature:
                request_params["temperature"] = temperature_value
            
            response = await asyncio.wait_for(
                client.chat.completions.create(**request_params),
                timeout=timeout,
            )
            
            reasoning = response.choices[0].message.content.strip()
            if not reasoning:
                raise ValueError("Empty reasoning generated")
            
            return reasoning
            
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                print(
                    f"Timeout on attempt {attempt + 1}/{max_retries}, retrying...",
                    file=sys.stderr
                )
                await asyncio.sleep(RATE_LIMIT_RETRY_DELAY)
                continue
            raise RuntimeError(f"Request timed out after {timeout}s")
            
        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            
            # Check for temperature unsupported error
            if "temperature" in error_str and ("unsupported" in error_str or "does not support" in error_str):
                if use_temperature:
                    # Retry without temperature parameter
                    use_temperature = False
                    print(
                        f"Model {model} doesn't support custom temperature, "
                        f"retrying without temperature parameter...",
                        file=sys.stderr
                    )
                    continue
                else:
                    # Already tried without temperature, this is a different error
                    pass
            
            # Check for rate limit errors
            if "rate limit" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    wait_time = RATE_LIMIT_RETRY_DELAY * (attempt + 1)
                    print(
                        f"Rate limit hit on attempt {attempt + 1}/{max_retries}, "
                        f"waiting {wait_time}s before retry...",
                        file=sys.stderr
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(f"Rate limit exceeded after {max_retries} attempts")
            
            # For other errors, retry once more
            if attempt < max_retries - 1:
                print(
                    f"Error on attempt {attempt + 1}/{max_retries}: {e}, retrying...",
                    file=sys.stderr
                )
                await asyncio.sleep(RATE_LIMIT_RETRY_DELAY)
                continue
            
            raise RuntimeError(f"Failed to generate reasoning after {max_retries} attempts: {e}") from e
    
    raise RuntimeError(f"Failed to generate reasoning after {max_retries} attempts")


def combine_reasoning_and_answer(reasoning: str, original_answer: str, language: str = "english") -> str:
    """
    Combine reasoning and answer into a single string.
    The reasoning and answer are combined into one cohesive answer field.
    
    Args:
        reasoning: The generated reasoning text
        original_answer: The original answer
        language: Language of the reasoning ("chinese" or "english")
    
    Returns:
        str: Combined text with reasoning followed by answer (all in answer field)
    """
    # Ensure reasoning ends properly
    reasoning = reasoning.rstrip()
    if language == "chinese":
        if not reasoning.endswith(('。', '！', '？', '.', '!', '?')):
            reasoning += "。"
        # Use Chinese separator
        separator = "\n\n最终答案：\n"
    else:
        # English
        if not reasoning.endswith(('.', '!', '?')):
            reasoning += "."
        separator = "\n\nFinal Answer:\n"
    
    # Combine reasoning and answer - both are part of the answer field
    # Format: reasoning text, then clear separator, then final answer
    combined = f"{reasoning}{separator}{original_answer}"
    return combined


async def process_sample(
    client: AsyncOpenAI,
    sample: Dict[str, Any],
    model: str,
    max_retries: int,
    timeout: int,
) -> Dict[str, Any]:
    """
    Process a single sample to add reasoning.
    
    Args:
        client: OpenAI async client
        sample: Sample dictionary with prompt and answer
        model: OpenAI model to use
        max_retries: Maximum retries
        timeout: Request timeout
    
    Returns:
        dict: Sample with updated answer field containing reasoning + answer
    """
    prompt = sample.get("prompt", "")
    original_answer = sample.get("answer", "")
    metadata = sample.get("metadata", {})
    
    if not prompt:
        raise ValueError("Sample missing 'prompt' field")
    if not original_answer:
        raise ValueError("Sample missing 'answer' field")
    
    # Detect language of the prompt
    language = detect_language(prompt, metadata)
    
    # Generate reasoning in the same language as the prompt
    reasoning = await generate_reasoning(
        client, prompt, original_answer, model, max_retries, timeout, language
    )
    
    # Combine reasoning and answer - this becomes the new answer field
    enhanced_answer = combine_reasoning_and_answer(reasoning, original_answer, language)
    
    # Create new sample with reasoning+answer in the answer field only
    new_sample = dict(sample)
    new_sample["answer"] = enhanced_answer  # Reasoning + final answer combined
    
    return new_sample


async def process_dataset(
    input_path: Path,
    output_path: Path,
    model: str,
    max_retries: int,
    timeout: int,
    max_concurrent: int = 5,
) -> None:
    """
    Process entire dataset to add reasoning.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        model: OpenAI model to use
        max_retries: Maximum retries per sample
        timeout: Request timeout
        max_concurrent: Maximum concurrent requests
    """
    client = get_openai_client()
    
    # Load samples
    samples = []
    print(f"Loading samples from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Skipping invalid JSON at line {line_no}: {e}",
                    file=sys.stderr
                )
                continue
    
    if not samples:
        raise ValueError(f"No valid samples found in {input_path}")
    
    print(f"Processing {len(samples)} samples with model {model}...")
    
    # Process samples with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    processed = 0
    errors = 0
    
    async def process_with_semaphore(sample: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        nonlocal processed, errors
        async with semaphore:
            try:
                result = await process_sample(client, sample, model, max_retries, timeout)
                processed += 1
                if processed % PROGRESS_INTERVAL == 0:
                    print(
                        f"Processed {processed}/{len(samples)} samples...",
                        file=sys.stderr
                    )
                return result
            except Exception as e:
                errors += 1
                print(
                    f"Error processing sample {index} (task_id={sample.get('task_id', 'unknown')}): {e}",
                    file=sys.stderr
                )
                return None
    
    # Process all samples
    tasks = [
        process_with_semaphore(sample, i)
        for i, sample in enumerate(samples)
    ]
    results = await asyncio.gather(*tasks)
    
    # Write results
    print(f"Writing {len([r for r in results if r is not None])} samples to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"✓ Successfully processed {processed} samples")
    if errors > 0:
        print(f"⚠ {errors} samples failed to process", file=sys.stderr)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add reasoning to dataset using OpenAI"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries per sample (default: {DEFAULT_MAX_RETRIES})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent requests (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    validate_file_path(input_path, must_exist=True, extensions=(".jsonl", ".json"))
    
    output_path = Path(args.output)
    validate_directory_path(output_path.parent, must_exist=False, create=True)
    
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be at least 1")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be at least 1")
    
    # Process dataset
    await process_dataset(
        input_path,
        output_path,
        args.model,
        args.max_retries,
        args.timeout,
        args.max_concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())

