# Dataset Generation from Environments

Both **LGC-V2** and **Trace** environments can generate datasets with **problems (prompts) and answers**.

## Overview

### LGC-V2 Environment
- **Problem**: Logic puzzle prompts (e.g., bracket matching, game of 24, cryptarithmetic)
- **Answer**: Stored in `game_data.answer` field, formatted according to task type requirements
- **Task Types**: 
  - `dyck_language` - Bracket matching completion (answer: full sequence)
  - `dyck_language2` - Reverse bracket matching (answer: full sequence in backticks)
  - `boolean_expressions` - Boolean logic evaluation (answer: `\boxed{result}`)
  - `operation` - Custom operator evaluation (answer: `\boxed{result}`)
  - `cryptarithm` - Letter-to-digit puzzles (answer: "The answer is ..." or "答案是 ...")
  - `sudoku` - Sudoku puzzles (answer: grid in triple backticks)
  - `game_of_24` - Arithmetic puzzle (no single reference answer, skipped by default)

### Trace Environment
- **Problem**: Python code execution prediction prompts
- **Answer**: Expected stdout output (stored in `ground_truth`)
- **Source**: Uses HuggingFace dataset `satpalsr/rl-python` by default
- **Dataset Size**: 23,303 items (indices 0-23302)
- **Generation Modes**: 
  - Sequential: Wraps around at 23303
  - Random: Randomly selects from 0-23302 (use `--random-selection` flag)

> **Implementation note (updated):** In the current code, Trace uses a **logical task_id range**
> of `[0, 1,000,000,000)` (PRINT range from the Affine config at
> [`https://api.affine.io/api/v1/config/environments`](https://api.affine.io/api/v1/config/environments)).
> Any logical `task_id` is mapped to the underlying HuggingFace dataset via
> `task_id % len(dataset)`, so the effective dataset index always stays within
> `[0, len(dataset) - 1]` while still supporting a large logical ID space.

## 1. Generating Datasets

A script is provided at `data_processing/generate_dataset.py` to generate datasets:

### Basic Usage

```bash
# Generate LGC-V2 dataset (all 7 task types with random selection)
python data_processing/generate_dataset.py \
  --env lgc-v2 \
  --output data_processing/dataset/lgc-v2.jsonl \
  --num-samples 100 \
  --seed 42 \
  --concurrency 16

# Generate Trace dataset (sequential, wraps around at 23303)
python data_processing/generate_dataset.py \
  --env trace \
  --output data_processing/dataset/trace.jsonl \
  --num-samples 1000 \
  --start-task-id 0 \
  --concurrency 4

# Generate Trace dataset with random selection (0-23302)
python data_processing/generate_dataset.py \
  --env trace \
  --output data_processing/dataset/trace-random.jsonl \
  --num-samples 10 \
  --random-selection \
  --seed 42 \
  --concurrency 8

# Generate specific task types from LGC-V2 (random selection among specified types)
python data_processing/generate_dataset.py \
  --env lgc-v2 \
  --task-types dyck_language game_of_24 operation \
  --output data_processing/dataset/mixed.jsonl \
  --num-samples 500 \
  --seed 42 \
  --concurrency 4
```

> **Persistence behavior:** `generate_dataset.py` writes each sample to the JSONL file **immediately as it is generated** (line-by-line streaming).  
> This means:
> - If the process is interrupted (Ctrl+C, crash, timeout), all samples generated so far are already saved.
> - You can safely inspect or resume from a **partially generated** file without losing work.

### Command-Line Options

**Common Options:**
- `--env`: Environment to use (`lgc-v2` or `trace`) (required)
- `--output`: Output JSONL file path (required)
- `--num-samples`: Number of samples to generate (required)
- `--seed`: Random seed for reproducibility (defaults to `--start-task-id` if not provided)
- `--start-task-id`: Starting task_id/seed for deterministic generation (default: 0)
- `--adapter-config`: JSON string for adapter configuration (e.g., `'{"dataset_name": "satpalsr/rl-python"}'`)
- `--concurrency`: Number of concurrent generations (default: 4). Higher values speed up generation but use more resources.

**LGC-V2 Specific:**
- `--task-types`: List of task types to generate (e.g., `dyck_language game_of_24`). 
  - If not specified, uses **all 7 task types** with **random selection** for each sample
  - Available types: `dyck_language`, `game_of_24`, `operation`, `cryptarithm`, `dyck_language2`, `boolean_expressions`, `sudoku`

**Trace Specific:**
- `--random-selection`: Randomly select task_ids from valid range (0-23302) instead of sequential
  - Trace dataset has 23,303 items (indices 0-23302)
  - Without this flag: sequential generation wraps around at 23303
  - With this flag: each sample randomly picks from 0-23302

> **Updated behavior:** The generator now derives the dataset size dynamically from the
> underlying `TraceTask` and uses a logical PRINT range `[0, 1e9)` for `task_id`. In practice:
> - Sequential mode uses `(start_task_id + i) % len(dataset)` for each sample.
> - Random mode samples logical ids uniformly (as if from `[0, 1e9)`) and then maps each to
>   an index via `task_id % len(dataset)`.

### Output Format

The generated JSONL files contain:

**LGC-V2 format:**
```json
{
  "task_id": 500,
  "task_type": "dyck_language",
  "prompt": "Complete the bracket sequence: ( [ { ...",
  "answer": "( [ { } ] )",
  "raw_answer": "( [ { } ] )",
  "seed": 500,
  "metadata": {
    "n_types": 3,
    "total_length": 40,
    "full_sequence": "( [ { } ] )",
    ...
  }
}
```

**Note on Answer Formatting:**
- Answers are automatically formatted according to each task type's requirements:
  - `dyck_language`: Full sequence (question + closing brackets)
  - `dyck_language2`: Full sequence wrapped in backticks (e.g., `` `( [ { } ] )` ``)
  - `boolean_expressions`: Result in `\boxed{}` format
  - `operation`: Result in `\boxed{}` format
  - `cryptarithm`: "The answer is ..." format (English) or "答案是 ..." (Chinese)
  - `sudoku`: Grid in triple backticks
  - `game_of_24`: Python code block (if answer exists)

**Trace format:**
```json
{
  "task_id": 0,
  "prompt": "Predict the exact and complete standard output...",
  "answer": "__DBG_0__ 42\n__DBG_1__ 'hello'\n...",
  "transformed_code": "def f():\n    x = 42\n    print('__DBG_0__', x)\n...",
  "inputs": "",
  "seed": 12345,
  "dataset_index": 0
}
```

## 2. Adding Reasoning to Datasets

You can enhance existing datasets by adding step-by-step reasoning to each sample's answer field using OpenAI. This is useful for training models that need to learn reasoning patterns.

### Overview

The `add_reasoning.py` script:
- Reads an existing dataset (JSONL format)
- Generates reasoning for each sample using OpenAI
- Combines reasoning + original answer into the `answer` field
- **Language-aware**: Automatically detects and uses the same language as the problem (English or Chinese)

### Basic Usage

```bash

# Add reasoning to existing dataset
python data_processing/add_reasoning.py \
  --input data_processing/dataset/trace-random.jsonl \
  --output data_processing/dataset/trace-random-with-reasoning.jsonl \
  --model gpt-4o-mini \
  --max-retries 3 \
  --max-concurrent 5
```

### Command-Line Options

- `--input`: Input JSONL file path (required)
- `--output`: Output JSONL file path (required)
- `--model`: OpenAI model to use (default: `gpt-4o-mini`)
- `--max-retries`: Maximum retries per sample on failure (default: 3)
- `--timeout`: Request timeout in seconds (default: 60)
- `--max-concurrent`: Maximum concurrent requests (default: 5)

### Language Detection

The script automatically detects the language of each problem:

1. **Metadata check**: First checks `metadata.language` field if available
2. **Character analysis**: Analyzes the prompt text for Chinese characters
3. **Language-specific reasoning**: Generates reasoning in the same language as the problem
   - **Chinese problems** → Chinese reasoning with `最终答案：` separator
   - **English problems** → English reasoning with `Final Answer:` separator

### Output Format

**Input (original dataset):**
```json
{
  "task_id": 1,
  "prompt": "Evaluate: 2+2",
  "answer": "4"
}
```

**Output (with reasoning):**
```json
{
  "task_id": 1,
  "prompt": "Evaluate: 2+2",
  "answer": "To solve 2+2, I add the two numbers together. 2 plus 2 equals 4.\n\nFinal Answer:\n4"
}
```

**For Chinese problems:**
```json
{
  "task_id": 1,
  "prompt": "定义 ⊙，规则如下：...",
  "answer": "要解决这个表达式，我们需要...\n\n最终答案：\n\\boxed{102}"
}
```

### Features

1. **Automatic Language Detection**: Detects Chinese vs English from metadata or text analysis
2. **Language-Matched Reasoning**: Generates reasoning in the same language as the problem
3. **Rate Limit Handling**: Automatically handles rate limits with exponential backoff
4. **Error Recovery**: Retries failed requests with configurable max attempts
5. **Concurrent Processing**: Processes multiple samples in parallel for faster generation
6. **Temperature Handling**: Automatically handles models that don't support custom temperature

### Troubleshooting

**Rate Limit Errors:**
- Reduce `--max-concurrent` (e.g., `--max-concurrent 2`)
- Increase `--timeout` if requests are timing out
- The script automatically retries with exponential backoff

**Temperature Errors:**
- Some models (e.g., `gpt-5`) don't support custom temperature
- The script automatically detects this and retries without temperature parameter
- No action needed - handled automatically

**Timeout Errors:**
- Increase `--timeout` value (default: 60 seconds)
- Some complex problems may need more time to generate reasoning

**Model Compatibility:**
- Recommended: `gpt-4o-mini` (fast, cost-effective)
- Also works: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Some newer models may have different parameter support

## 3. Evaluating Dataset Answers

After generating a dataset (with or without reasoning), you can evaluate the answers using the evaluation script:

### Basic Usage

```bash
# Evaluate LGC-V2 dataset
python data_processing/evaluate_dataset.py \
  --input data_processing/dataset/lgc-v2.jsonl \
  --env lgc-v2 \
  --output data_processing/results/lgc-v2-eval.json

# Evaluate Trace dataset
python data_processing/evaluate_dataset.py \
  --input data_processing/dataset/trace-random-with-reasoning.jsonl \
  --env trace \
  --output data_processing/results/trace-eval.json

# Show detailed per-sample results
python data_processing/evaluate_dataset.py \
  --input data_processing/dataset/lgc-v2.jsonl \
  --env lgc-v2 \
  --verbose
```

### Evaluation Output

The evaluation script provides:
- **Overall accuracy**: Percentage of correct answers
- **Per-task-type accuracy**: Breakdown by task type (LGC-V2 only)
- **Error statistics**: Count of evaluation errors
- **Detailed results**: Per-sample scores (optional with `--verbose`)

**Example output:**
```
============================================================
Evaluation Results
============================================================
Environment: lgc-v2
Total Samples: 1000
Correct: 850
Incorrect: 150
Accuracy: 85.00%
Errors: 0

Per-Task-Type Accuracy:
  boolean_expressions: 80/100 (80.00%)
  dyck_language: 90/100 (90.00%)
  game_of_24: 75/100 (75.00%)
  ...
```

### Command-Line Options

**Common Options:**
- `--input`: Input JSONL file path (required)
- `--env`: Environment type (`lgc-v2` or `trace`) (required)
- `--output`: Output JSON file path (optional, prints to stdout if not provided)
- `--verbose`: Show detailed per-sample results
- `--adapter-config`: JSON string for adapter configuration

### Evaluation with Reasoning-Enhanced Datasets

The evaluation script automatically extracts the final answer from reasoning-enhanced datasets:

- Recognizes both English (`Final Answer:`) and Chinese (`最终答案：`) markers
- Extracts only the final answer for evaluation
- Maintains 100% accuracy when the final answer is correct

```bash
# Evaluate reasoning-enhanced dataset
python data_processing/evaluate_dataset.py \
  --input data_processing/dataset/lgc-v2-with-reasoning.jsonl \
  --env lgc-v2 \
  --output data_processing/results/lgc-v2-eval.json
```

### Use Cases

1. **Validate generated datasets**: Ensure all answers in your dataset are correct
2. **Test model responses**: Evaluate model outputs against ground truth
3. **Quality control**: Check dataset quality before training
4. **Benchmarking**: Compare different models or approaches

### Troubleshooting

**Common Issues:**

1. **Generation Failures**: If you see "Failed to generate valid sequence" errors:
   - This is normal for some `dyck_language` tasks with difficult parameter combinations
   - The script automatically retries with different task_ids
   - If you consistently get fewer samples than requested, try a different seed

2. **Evaluation Accuracy < 100%**: 
   - Check that answers are properly formatted (the generator handles this automatically)
   - For `dyck_language` and `dyck_language2`, ensure full sequences are used
   - Run with `--verbose` to see which specific samples failed
   - For reasoning-enhanced datasets, ensure the final answer is correctly formatted after the separator

3. **Trace Dataset Index Errors**:
   - Logical task_ids live in a large PRINT range; the implementation always maps to a valid
     dataset index using `task_id % len(dataset)`
   - Use `--random-selection` for random sampling over the logical range
   - Sequential mode automatically wraps around via modulo indexing

## 4. Converting to VERL Format

After generating and optionally enhancing your dataset with reasoning, convert it to VERL parquet format for training. The converter automatically handles generated datasets:

### SFT Format

```bash
# For SFT training (prompt + response)
# The converter automatically detects "answer" field as response
python data_processing/convert_to_parquet.py \
  --input data_processing/dataset/lgc-v2.jsonl \
  --out-dir train/parquet \
  --env lgc-v2 \
  --format sft \
  --val-ratio 0.4 \
  --shuffle
```

**SFT Format Features:**
- **Automatic field detection**: `answer` → `response` (auto-detected)
- **Metadata preservation**: All `metadata` fields preserved in `extra_info`
- **Additional fields**: Fields like `raw_answer`, `transformed_code`, `inputs`, `dataset_index` are preserved in `extra_info`
- **Backward compatible**: Still works with custom JSONL files that use different field names

### RL Format (For Reinforcement Learning)

```bash
# For RL training (prompt + ground_truth)
# The converter automatically uses the "answer" field as ground_truth
python data_processing/convert_to_parquet.py \
  --input data_processing/dataset/lgc-v2.jsonl \
  --out-dir train/parquet \
  --env lgc-v2 \
  --format rl
```

**RL Format Features:**
- **Automatic field detection**: `answer` → `reward_model.ground_truth`
- **Chat format**: Prompt converted to chat messages format
- **Metadata preservation**: All fields preserved in `extra_info`

**Explicit field specification** (optional, usually not needed):
```bash
# Explicitly specify response field (if auto-detection fails)
python data_processing/convert_to_parquet.py \
  --input data_processing/dataset/lgc-v2.jsonl \
  --out-dir train/parquet_sft \
  --env lgc-v2 \
  --format sft \
  --response-field answer
```

### Command-Line Options

**Common Options:**
- `--input`: Input .jsonl/.json file or directory (default: "dataset")
- `--out-dir`: Output directory for parquet files (default: "parquet")
- `--env`: Value for extra_info.env (e.g., lgc-v2, trace) (default: "lgc-v2")
- `--format`: Output format - `rl` (prompt as chat messages) or `sft` (prompt/response strings) (default: "rl")
- `--response-field`: SFT only - input field name to use as response (auto-detected if omitted from: response, answer, completion, output, assistant, label)
- `--prompt-field`: Field name in input containing prompt/messages (default: "prompt")
- `--data-source-field`: Optional field name to use as data_source (default: data_source or task_type)
- `--val-ratio`: If >0, split into train/val with this ratio (default: 0.0)
- `--seed`: Random seed for splitting/shuffling (default: 42)
- `--shuffle`: Shuffle before splitting (flag)

## Example: Full Pipeline

### Complete Workflow with Reasoning

```bash
# 1. Generate base dataset
python data_processing/generate_dataset.py \
  --env lgc-v2 \
  --output data_processing/dataset/lgc-v2.jsonl \
  --num-samples 1000 \
  --seed 42 \
  --concurrency 4

# 2. Add reasoning to dataset (optional)

python data_processing/add_reasoning.py \
  --input data_processing/dataset/lgc-v2.jsonl \
  --output data_processing/dataset/lgc-v2-with-reasoning.jsonl \
  --model gpt-4o-mini \
  --max-concurrent 5

# 3. Evaluate dataset (validate correctness)
python data_processing/evaluate_dataset.py \
  --input data_processing/dataset/lgc-v2-with-reasoning.jsonl \
  --env lgc-v2 \
  --output data_processing/results/lgc-v2-eval.json

# 4. Convert to VERL format for training
python data_processing/convert_to_parquet.py \
  --input data_processing/dataset/lgc-v2-with-reasoning.jsonl \
  --out-dir train/parquet \
  --env lgc-v2 \
  --format sft \
  --val-ratio 0.1 \
  --shuffle
```

### LGC-V2 Example (Without Reasoning)

```bash
# 1. Generate dataset with all task types (random selection)
python data_processing/generate_dataset.py \
  --env lgc-v2 \
  --output data_processing/dataset/lgc-v2-training.jsonl \
  --num-samples 10000 \
  --seed 42 \
  --concurrency 4

# 2. Evaluate dataset
python data_processing/evaluate_dataset.py \
  --input data_processing/dataset/lgc-v2-training.jsonl \
  --env lgc-v2 \
  --output data_processing/results/lgc-v2-eval.json

# 3. Convert to SFT format (answer field auto-detected)
python data_processing/convert_to_parquet.py \
  --input data_processing/dataset/lgc-v2-training.jsonl \
  --out-dir train/parquet_sft \
  --env lgc-v2 \
  --format sft \
  --val-ratio 0.1 \
  --shuffle
```

### Trace Example

```bash
# 1. Generate Trace dataset with random selection
python data_processing/generate_dataset.py \
  --env trace \
  --output data_processing/dataset/trace-training.jsonl \
  --num-samples 10000 \
  --random-selection \
  --seed 42 \
  --concurrency 4

# 2. Evaluate dataset
python data_processing/evaluate_dataset.py \
  --input data_processing/dataset/trace-training.jsonl \
  --env trace \
  --output data_processing/results/trace-eval.json

# 3. Convert to SFT format (answer field auto-detected)
python data_processing/convert_to_parquet.py \
  --input data_processing/dataset/trace-training.jsonl \
  --out-dir train/parquet_sft \
  --env trace \
  --format sft \
  --val-ratio 0.1 \
  --shuffle
```

### Mixed Task Types Example

```bash
# Generate specific task types from LGC-V2
python data_processing/generate_dataset.py \
  --env lgc-v2 \
  --task-types dyck_language game_of_24 operation \
  --output data_processing/dataset/mixed-task-types.jsonl \
  --num-samples 5000 \
  --seed 42 \
  --concurrency 4
```

## Programmatic Usage

You can also generate datasets programmatically:

```python
import asyncio
from tools.env_adapter import get_env_adapter

async def generate_samples():
    # LGC-V2
    adapter = get_env_adapter("lgc-v2")
    challenge = adapter.generate(task_id=500)
    
    # Extract answer
    game_data = json.loads(challenge.extra["game_data"])
    answer = game_data["answer"]
    
    print(f"Prompt: {challenge.prompt}")
    print(f"Answer: {answer}")
    
    # Trace
    trace_adapter = get_env_adapter("trace")
    trace_challenge = trace_adapter.generate(task_id=0)
    
    print(f"Prompt: {trace_challenge.prompt}")
    print(f"Answer: {trace_challenge.extra['ground_truth']}")

asyncio.run(generate_samples())
```

## Key Points

1. **Task Type Selection**:
   - **LGC-V2**: When `--task-types` is not specified, **randomly selects** from all 7 task types for each sample
   - **LGC-V2**: When `--task-types` is specified, randomly selects among those types
   - **Trace**: Use `--random-selection` to randomly sample over the logical PRINT id space,
     which is then mapped to concrete dataset indices via modulo

2. **Deterministic Generation**: Both environments use `task_id` for deterministic generation
   - LGC-V2: `task_id = task_type_id * 100M + seed` (overall `[0, 899,999,999]` across all task types)
   - Trace: logical `task_id` is in a large PRINT range and the effective index is
     always `task_id % len(dataset)`

3. **Answer Availability**: 
   - ✅ LGC-V2: Answers are in `game_data.answer`
   - ✅ Trace: Answers are in `ground_truth` (expected stdout)

4. **Reproducibility**: Same `task_id` + same `seed` → same problem + answer

5. **Scalability**: 
   - LGC-V2: 100M unique tasks per task type (7 task types + reasoning-errors = 800M total logical ids)
   - Trace: 23,303 unique underlying items, but a large logical PRINT id space that is mapped via modulo
   - Trace sequential mode wraps around automatically using modulo indexing

6. **Error Handling and Retry Logic**:
   - The generator includes automatic retry logic for failed generations
   - If a task fails to generate (e.g., `dyck_language` sequence generation fails), the script automatically retries up to 5 times with different task_ids
   - This ensures you get the requested number of samples even if some generations fail
   - Failed generations are logged to stderr for debugging

7. **Answer Format Requirements**:
   - Answers are formatted to match each task type's prompt requirements
   - For `dyck_language` and `dyck_language2`, the answer is the **full sequence** (not just the missing part), as required by the verifiers
   - Formatting ensures compatibility with environment verifiers for accurate evaluation

8. **Reasoning-Enhanced Datasets**:
   - Reasoning is included directly in the `answer` field
   - The evaluation script automatically extracts the final answer from reasoning-enhanced answers
   - Language detection ensures reasoning matches the problem's language (English or Chinese)
