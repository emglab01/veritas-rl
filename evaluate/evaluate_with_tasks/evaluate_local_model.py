#!/usr/bin/env python3
"""
Universal evaluation script for affine environments

Supports two evaluation modes:
1. Using Chutes service via --uid
2. Using local model via --base-url (recommended: http://172.17.0.1:30000/v1 for Docker)

Evaluation methods:
- Single task: --task-id <id> (with optional --samples for multiple runs)
- Range: --task-id-range <start> <end> (one sample per task)
- Task IDs from file: --use-task-ids (loads from task_ids.json)

Usage:
    # Recommended: Use Docker bridge gateway for container access
    ./evaluate_local_model.py --env LGC-V2 --base-url http://172.17.0.1:30000/v1 --use-task-ids
    
    # Using Chutes service
    ./evaluate_local_model.py --env ABD-V2 --uid 7
    
    # Single task evaluation
    ./evaluate_local_model.py --env ABD --base-url http://172.17.0.1:30000/v1 --task-id 5 --samples 10
    
    # Range evaluation
    ./evaluate_local_model.py --env PRINT --base-url http://172.17.0.1:30000/v1 --task-id-range 0 9

Setup:
    - Virtual environment must be in affine-cortex folder (shared with scripts)
    - Use uv venv: cd affine-cortex && uv venv && source .venv/bin/activate
    - Install: uv pip install -e . (in affine-cortex) && uv pip install -r requirements.txt (in scripts)
    - See README.md for detailed setup instructions
"""
import asyncio
import argparse
import sys
import os
import json
import random
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
import wandb
import time
import hashlib
import httpx
from urllib.parse import urlparse

# ============================================================================
# Constants
# ============================================================================

# Evaluation defaults
DEFAULT_TIMEOUT = 300  # seconds
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CONNECTION_TIMEOUT = 5.0  # seconds
ERROR_MESSAGE_MAX_LENGTH = 200  # characters
ERROR_DISPLAY_MAX_LENGTH = 100  # characters for console output

# File paths
ROLLOUT_FILE = "rollout.jsonl"
TASK_IDS_FILE = "task_ids.json"

# Supported environment names (will be mapped to actual classes after argparse)
ENVIRONMENT_NAMES = ['SAT', 'ABD', 'ABD-V2', 'DED', 'DED-V2', "LGC", "SWE_PRO", "CDE", "GAME", "PRINT", "LGC-V2"]

# Connection error keywords for detection
CONNECTION_ERROR_KEYWORDS = ['connection', 'connect', 'refused', 'unreachable', 'timeout', 'network']

# ============================================================================
# Environment Setup
# ============================================================================

# Load environment variables from .env file
# The script loads WANDB_API_KEY and CHUTES_API_KEY from .env file in project root
# Try multiple locations: project root (recommended), script directory, current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up from scripts/ to evaluate/ to project root

# Try loading .env from project root first (where env.example is), then script dir, then current dir
env_loaded = False
env_file_path = None
for env_path in [
    os.path.join(project_root, '.env'),  # Recommended: /root/veritas-rl/.env
    os.path.join(script_dir, '.env'),
    '.env'
]:
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)
        env_loaded = True
        env_file_path = env_path
        break

# If no .env found, try default load_dotenv() behavior (current directory)
if not env_loaded:
    load_dotenv()
    env_file_path = '.env' if os.path.exists('.env') else None

# After load_dotenv(), both WANDB_API_KEY and CHUTES_API_KEY will be available via os.getenv()
# They are checked and used in main() function

# ============================================================================
# Helper Functions
# ============================================================================

def format_error_message(error: Exception, max_length: int = ERROR_MESSAGE_MAX_LENGTH) -> Tuple[str, str]:
    """
    Extract concise error information from an exception.
    
    Args:
        error: The exception to format
        max_length: Maximum length for error message
        
    Returns:
        Tuple of (error_type, concise_error_message)
    """
    error_type = type(error).__name__
    error_message = str(error)
    # Extract first line of error message (before newline) or truncate
    first_line = error_message.split('\n')[0] if '\n' in error_message else error_message[:max_length]
    concise_error = f"{error_type}: {first_line}"
    return error_type, concise_error


def is_connection_error(error_message: str) -> bool:
    """Check if an error message indicates a connection error."""
    error_lower = error_message.lower()
    return any(keyword in error_lower for keyword in CONNECTION_ERROR_KEYWORDS)


def create_error_result_dict(error_type: str, concise_error: str, task_id: Optional[int] = None, 
                            seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a standardized error result dictionary.
    
    Args:
        error_type: Type of the error
        concise_error: Concise error message
        task_id: Optional task ID
        seed: Optional seed value
        
    Returns:
        Dictionary representing a failed evaluation result
    """
    result = {
        'score': 0.0,
        'success': False,
        'error': concise_error,
        'error_type': error_type,
        'latency_seconds': 0.0
    }
    if task_id is not None:
        result['task_id'] = task_id
    if seed is not None:
        result['seed'] = seed
    return result


def log_to_wandb(env_name: str, score: float, avg_score: float, count: int) -> None:
    """Log metrics to W&B."""
    log_dict = {
        f"{env_name}/score": score,
        f"{env_name}/avg_score": avg_score,
        f"{env_name}/count": count,
    }
    wandb.log(log_dict)


# ============================================================================
# Main Functions
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate models on affine environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended: Evaluate using task IDs with Docker bridge gateway
  %(prog)s --env LGC-V2 --base-url http://172.17.0.1:30000/v1 --use-task-ids
  %(prog)s --env PRINT --base-url http://172.17.0.1:30000/v1 --use-task-ids --output results.json
  
  # Evaluate using Chutes service (requires CHUTES_API_KEY)
  %(prog)s --env ABD-V2 --uid 7
  
  # Single task evaluation
  %(prog)s --env ABD-V2 --base-url http://172.17.0.1:30000/v1 --task-id 5 --samples 10
  
  # Evaluate with specific task and seed
  %(prog)s --env ALFWORLD --task-id 2 --seed 42 --base-url http://172.17.0.1:30000/v1 --samples 5

  # Evaluate across a range of task IDs (one sample per task)
  %(prog)s --env ALFWORLD --base-url http://172.17.0.1:30000/v1 --task-id-range 0 9
  %(prog)s --env ABD --task-id-range 0 4 --uid 7 --output results.json

  # Alternative: Use host.docker.internal (Docker Desktop)
  %(prog)s --env ABD --base-url http://host.docker.internal:30000/v1 --use-task-ids

Supported environments: """ + ', '.join(ENVIRONMENT_NAMES)
    )

    parser.add_argument('--env', '--envname', required=True, type=str,
                        dest='env',
                        help='Environment name (case-insensitive)')
    
    # Mode selection: either uid or base-url
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--uid', type=int,
                           help='Miner UID (use Chutes service)')
    
    parser.add_argument('--base-url', default='http://localhost:30000/v1',
                       help='Model service URL. Recommended: http://172.17.0.1:30000/v1 (Docker bridge gateway). '
                            'Default: http://localhost:30000/v1 (only works if NOT in Docker). '
                            'Alternative: http://host.docker.internal:30000/v1 (Docker Desktop)')
    
    # Evaluation method selection (mutually exclusive)
    eval_method_group = parser.add_mutually_exclusive_group()
    eval_method_group.add_argument('--task-id', type=int,
                       help='Single task ID for evaluation (can use with --samples for multiple runs)')
    eval_method_group.add_argument('--task-id-range', type=int, nargs=2, metavar=('START', 'END'),
                       help='Range of task IDs to evaluate (one sample per task)')
    eval_method_group.add_argument('--use-task-ids', action='store_true',
                       help=f'Load task IDs from {TASK_IDS_FILE} file')
    
    parser.add_argument('--seed', type=int,
                       help='Random seed for evaluation (only used with --task-id)')
    parser.add_argument('--samples', type=int, default=1,
                       help='Number of evaluation samples per task (default: 1, only used with --task-id)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Sampling temperature (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--output', '-o',
                       help='Output file path for JSON results (optional)')
    parser.add_argument('--wandb-project', type=str, default=None,
                       help='W&B project name (default: {env}-Benchmark)')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='W&B run name (default: auto-generated)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B logging')

    args = parser.parse_args()

    # Normalize environment name to uppercase for case-insensitive matching
    args.env = args.env.upper()
    
    # Validate environment name
    if args.env not in ENVIRONMENT_NAMES:
        parser.error(f'invalid environment: {args.env!r} (choose from {", ".join(ENVIRONMENT_NAMES)})')
    
    # Validate seed usage
    if args.seed and not args.task_id:
        parser.error('--seed can only be used with --task-id')

    return args


def generate_random_task_ids(count: int = 1000, category_count: int = 6, bucket_size: int = 100_000_000):
    """Generate task IDs where the leading digit encodes category buckets."""
    ids = []
    for _ in range(count):
        category = random.randint(0, category_count - 1)
        start = category * bucket_size
        end = start + bucket_size - 1
        ids.append(random.randint(start, end))
    return ids


async def evaluate_with_uid(env_instance, uid: int, task_id: Optional[int], seed: Optional[int], samples: int, temperature: float, af):
    """Evaluate using Chutes service via miner UID"""
    print(f"\nFetching miner info for UID {uid}...")
    miner = await af.miners(uid)
    if not miner:
        raise ValueError(f"Unable to get miner info for UID {uid}")

    print(f"Miner found: {miner.get(uid).model}")

    total_score = 0.0
    total_time = 0.0
    results = []

    for i in range(samples):
        print(f"\rProgress: {i+1}/{samples}", end="", flush=True)

        eval_kwargs = {'temperature': temperature}
        if task_id is not None:
            eval_kwargs['task_id'] = task_id
        if seed is not None:
            eval_kwargs['seed'] = seed
        
        result = await env_instance.evaluate(miner, **eval_kwargs)

        # Result is a dict with uid as key
        eval_result = result[uid]
        total_score += eval_result.score
        total_time += eval_result.latency_seconds
        
        results.append(result[uid].model_dump())

    print()  # New line after progress
    return total_score, total_time, results

def generate_deterministic_seed(env_name: str, task_id: int) -> int:
    # Combine env_name and task_id to create a unique string
    seed_string = f"{env_name}:{task_id}"
    # Use SHA256 hash to generate deterministic seed
    hash_object = hashlib.sha256(seed_string.encode())
    # Convert first 8 bytes of hash to integer and modulo to fit in 32-bit range
    seed = int.from_bytes(hash_object.digest()[:8], byteorder='big') % (2**32)
    return seed


async def check_connection(base_url: str, timeout: float = DEFAULT_CONNECTION_TIMEOUT) -> bool:
    """
    Check if the model service is accessible.
    
    Args:
        base_url: Base URL of the model service
        timeout: Connection timeout in seconds
        
    Returns:
        True if service is accessible, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Most OpenAI-compatible APIs have a /models endpoint
            models_url = f"{base_url.rstrip('/')}/models"
            try:
                response = await client.get(models_url)
                return response.status_code < 500  # Any non-server-error is fine
            except httpx.RequestError:
                return False
    except Exception:
        return False
    

async def evaluate_with_model(env_instance, base_url: str, task_id: Optional[int], seed: Optional[int], 
                              samples: int, temperature: float, use_wandb: bool = True) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Evaluate using direct model endpoint with single task ID.
    
    Args:
        env_instance: Environment instance
        base_url: Model service base URL
        task_id: Optional task ID
        seed: Optional random seed
        samples: Number of evaluation samples
        temperature: Sampling temperature
        use_wandb: Whether to log to W&B
        
    Returns:
        Tuple of (total_score, total_time, results_list)
    """
    print(f"\nService URL: {base_url}")

    total_score = 0.0
    total_time = 0.0
    results = []

    for sample_idx in range(samples):
        print(f"\rProgress: {sample_idx+1}/{samples}", end="", flush=True)
        
        eval_kwargs = {
            'base_url': base_url,
            'temperature': temperature,
            'timeout': DEFAULT_TIMEOUT
        }
        if task_id is not None:
            eval_kwargs['task_id'] = task_id
        if seed is not None:
            eval_kwargs['seed'] = seed
        
        try:
            result = await env_instance.evaluate(**eval_kwargs)
            total_score += result.score
            total_time += result.latency_seconds
            results.append(result.model_dump())
            
            if use_wandb:
                log_to_wandb(
                    env_instance.env_name,
                    result.score,
                    total_score / (sample_idx + 1),
                    sample_idx + 1
                )
        except Exception as e:
            error_type, concise_error = format_error_message(e)
            print(f"\nError evaluating task {task_id}: {concise_error}")
            
            error_result = create_error_result_dict(error_type, concise_error)
            results.append(error_result)
            
            if use_wandb:
                log_to_wandb(
                    env_instance.env_name,
                    0.0,
                    total_score / (sample_idx + 1),
                    sample_idx + 1
                )

    print()  # New line after progress
    return total_score, total_time, results


async def evaluate_with_task_ids(env_instance, base_url: str, temperature: float, 
                                  task_ids_path: Optional[str] = None, use_wandb: bool = True) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Evaluate using task IDs from JSON file.
    
    Args:
        env_instance: Environment instance
        base_url: Model service base URL
        temperature: Sampling temperature
        task_ids_path: Path to task IDs JSON file (defaults to {TASK_IDS_FILE} in script directory)
        use_wandb: Whether to log to W&B
        
    Returns:
        Tuple of (total_score, total_time, results_list)
    """
    print(f"\nService URL: {base_url}")

    # Load task IDs from JSON file
    if task_ids_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        task_ids_path = os.path.join(script_dir, TASK_IDS_FILE)
    
    if not os.path.exists(task_ids_path):
        raise FileNotFoundError(f"Task IDs file not found: {task_ids_path}")
    
    with open(task_ids_path, 'r') as f:
        task_ids = json.load(f)
    
    if not isinstance(task_ids, list):
        raise ValueError(f"{TASK_IDS_FILE} must contain a list of task IDs, got {type(task_ids)}")
    
    print(f"Loaded {len(task_ids)} task IDs from {task_ids_path}")

    total_score = 0.0
    total_time = 0.0
    results = []
    
    # Open rollout file once for appending (more efficient than opening repeatedly)
    with open(ROLLOUT_FILE, "a", encoding='utf-8') as rollout_file:
        for task_idx, task_id in enumerate(task_ids):
            seed = generate_deterministic_seed(env_instance.env_name.lower(), task_id)
            print(f"\rProgress: {task_idx+1}/{len(task_ids)} (Task ID: {task_id}, Seed: {seed})", end="", flush=True)
            
            eval_kwargs = {
                'base_url': base_url,
                'temperature': temperature,
                'task_id': task_id,
                'timeout': DEFAULT_TIMEOUT
            }
            
            try:
                result = await env_instance.evaluate(**eval_kwargs)
                total_score += result.score
                total_time += result.latency_seconds
                result_dict = result.model_dump()
                result_dict['task_id'] = task_id
                result_dict['seed'] = seed
                results.append(result_dict)
                
                if use_wandb:
                    log_to_wandb(
                        env_instance.env_name,
                        result.score,
                        total_score / (task_idx + 1),
                        task_idx + 1
                    )
                
                # Append to rollout file
                rollout_file.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                rollout_file.flush()  # Ensure data is written immediately

            except Exception as e:
                error_type, concise_error = format_error_message(e)
                error_message = str(e)
                
                # Check if it's a connection error
                if is_connection_error(error_message):
                    print(f"\n⚠️  Connection error for task {task_id}: {concise_error[:ERROR_DISPLAY_MAX_LENGTH]}...")
                    print(f"   Make sure the model service is running at {base_url}")
                else:
                    print(f"\nError evaluating task {task_id}: {concise_error[:ERROR_DISPLAY_MAX_LENGTH]}...")
                
                error_result = create_error_result_dict(error_type, concise_error, task_id=task_id, seed=seed)
                results.append(error_result)
                
                if use_wandb:
                    log_to_wandb(
                        env_instance.env_name,
                        0.0,
                        total_score / (task_idx + 1),
                        task_idx + 1
                    )

    print()  # New line after progress
    return total_score, total_time, results


async def evaluate_with_model_range(env_instance, base_url: str, task_id_start: int, task_id_end: int, 
                                     temperature: float, use_wandb: bool = True) -> Tuple[float, float, List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Evaluate using direct model endpoint across a range of task IDs (one sample per task).
    
    Args:
        env_instance: Environment instance
        base_url: Model service base URL
        task_id_start: Starting task ID (inclusive)
        task_id_end: Ending task ID (inclusive)
        temperature: Sampling temperature
        use_wandb: Whether to log to W&B
        
    Returns:
        Tuple of (total_score, total_time, all_results_list, task_results_dict)
    """
    print(f"\nService URL: {base_url}")

    total_score = 0.0
    total_time = 0.0
    all_results = []
    task_results = {}

    task_count = task_id_end - task_id_start + 1

    for task_idx, task_id in enumerate(range(task_id_start, task_id_end + 1), start=1):
        print(f"\rProgress: {task_idx}/{task_count} (Task {task_id})", end="", flush=True)

        eval_kwargs = {
            'base_url': base_url,
            'temperature': temperature,
            'task_id': task_id,
            'timeout': DEFAULT_TIMEOUT
        }

        try:
            result = await env_instance.evaluate(**eval_kwargs)
            total_score += result.score
            total_time += result.latency_seconds

            sample_result = result.model_dump()
            sample_result['task_id'] = task_id
            all_results.append(sample_result)
            task_results[task_id] = {
                'score': result.score,
                'success': result.success,
                'time': result.latency_seconds,
                'error': result.error
            }
            
            if use_wandb:
                log_to_wandb(
                    env_instance.env_name,
                    result.score,
                    total_score / task_idx,
                    task_idx
                )
        except Exception as e:
            error_type, concise_error = format_error_message(e)
            print(f"\nError evaluating task {task_id}: {concise_error}")
            
            error_result = create_error_result_dict(error_type, concise_error, task_id=task_id)
            all_results.append(error_result)
            task_results[task_id] = {
                'score': 0.0,
                'success': False,
                'time': 0.0,
                'error': concise_error
            }
            
            if use_wandb:
                log_to_wandb(
                    env_instance.env_name,
                    0.0,
                    total_score / task_idx,
                    task_idx
                )

    print()  # New line after progress
    return total_score, total_time, all_results, task_results

async def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Get API keys from environment (loaded from .env file)
    # Both keys should be in /root/veritas-rl/.env
    wandb_api_key = os.getenv("WANDB_API_KEY")
    chutes_api_key = os.getenv("CHUTES_API_KEY")
    
    # Initialize W&B if enabled
    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            print("⚠️  WANDB_API_KEY not found in .env file - W&B logging may not work")
            print(f"   Expected location: {os.path.join(project_root, '.env')}")
            print("   Continuing without W&B authentication...")
        
        # Generate run name
        if args.wandb_name:
            run_name = args.wandb_name
        else:
            if args.uid:
                run_name = f"uid-{args.uid}-{time.strftime('%Y%m%d-%H%M%S')}"
            else:
                # Extract hostname from base_url for run name
                from urllib.parse import urlparse
                parsed = urlparse(args.base_url)
                hostname = parsed.hostname or 'localhost'
                run_name = f"{hostname}-{time.strftime('%Y%m%d-%H%M%S')}"
        
        # Determine project name
        project_name = args.wandb_project or f"{args.env}-Benchmark"
        
        wandb_run = wandb.init(project=project_name, config=vars(args), name=run_name)
   
    # Import affine AFTER argparse to avoid bittensor hijacking
    # Add affine-cortex to Python path if not already installed
    affine_cortex_path = os.path.join(project_root, 'evaluate', 'affine-cortex')
    if os.path.exists(affine_cortex_path) and affine_cortex_path not in sys.path:
        sys.path.insert(0, affine_cortex_path)
    
    import affine as af
    af.trace()
    
    # Map environment names to actual classes
    ENVIRONMENTS = {
        'sat': af.SAT,
        'abd': af.ABD,
        'abd-v2': af.ABD_V2,
        'ded': af.DED,
        'ded-v2': af.DED_V2,
        'lgc': af.LGC,
        'swe_pro': af.SWEPRO,
        'cde': af.CDE,
        'game': af.GAME,
        'print': af.PRINT,
        'lgc-v2': af.LGC_V2,
    }
    
    # Check API key for Chutes service (already loaded from .env above)
    if args.uid and not chutes_api_key:
        print("\n❌ CHUTES_API_KEY environment variable not found")
        print("   Please add CHUTES_API_KEY to your .env file")
        print(f"   Expected location: {os.path.join(project_root, '.env')}")
        print("   Or set: export CHUTES_API_KEY='your-key'")
        sys.exit(1)
    
    # Set fake API key for local model testing (required by Docker env)
    if args.uid is None and not chutes_api_key:
        os.environ["CHUTES_API_KEY"] = "fake-test-key-for-local-testing"
        print("⚠️  CHUTES_API_KEY not found in .env, using temporary test key for local evaluation")
    
    print("=" * 60)
    print("Evaluation Configuration:")
    print(f"  Environment: {args.env}")
    if args.uid:
        print(f"  Mode: Chutes service (UID: {args.uid})")
    else:
        print(f"  Mode: Direct model")
        print(f"  Base URL: {args.base_url}")
        
        # Check connection before starting
        print("\nChecking connection to model service...")
        if not await check_connection(args.base_url):
            print(f"\n⚠️  WARNING: Cannot verify connection to model service at {args.base_url}")
            print("   This may indicate:")
            print("   - The model service is not running")
            print("   - The service is not accessible at the specified URL")
            print("   - Network/firewall issues")
            
            # Check if running in Docker and suggest host.docker.internal
            if 'localhost' in args.base_url or '127.0.0.1' in args.base_url:
                print("\n   💡 DOCKER DETECTION:")
                print("   If you're running this script inside a Docker container,")
                print("   'localhost' won't work. Try one of these instead:")
                print(f"   - http://host.docker.internal:30000/v1 (Docker Desktop)")
                print(f"   - http://172.17.0.1:30000/v1 (Docker bridge gateway)")
                print(f"   - Use the host machine's IP address")
                print("\n   Example:")
                print(f"   ./evaluate_local_model.py --env {args.env} --base-url http://172.17.0.1:30000/v1 --use-task-ids")
            
            print("\n   You can test the connection manually with:")
            print(f"   curl {args.base_url}/models")
            print("\n   Continuing anyway - connection errors will be logged in results...")
        else:
            print("✓ Connection check passed")
    
    # Print evaluation method
    if args.use_task_ids:
        print(f"  Method: Task IDs from {TASK_IDS_FILE}")
    elif args.task_id_range:
        print(f"  Method: Range ({args.task_id_range[0]} to {args.task_id_range[1]})")
    elif args.task_id is not None:
        print(f"  Method: Single task (ID: {args.task_id})")
        print(f"  Samples: {args.samples}")
    else:
        print(f"  Method: Single task (no task_id specified)")
        print(f"  Samples: {args.samples}")
    
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    print(f"  Temperature: {args.temperature}")
    print("=" * 60)
    
    try:
        # Create environment instance
        print(f"\nLoading {args.env} environment...")
        env_class = ENVIRONMENTS[args.env.lower()]
        env_instance = env_class()
        print("✓ Environment loaded")
        
        # Run evaluation based on method
        task_results = None
        
        if args.uid:
            # UID mode - only supports single task or range
            if args.task_id_range:
                raise NotImplementedError("UID mode with task-id-range not yet implemented")
            else:
                print(f"\nStarting evaluation ({args.samples} sample(s))...")
                total_score, total_time, results = await evaluate_with_uid(
                    env_instance, args.uid, args.task_id, args.seed, args.samples, args.temperature, af
                )
        else:
            # Direct model mode
            if args.use_task_ids:
                print(f"\nStarting evaluation with task IDs from file...")
                total_score, total_time, results = await evaluate_with_task_ids(
                    env_instance, args.base_url, args.temperature, use_wandb=use_wandb
                )
            elif args.task_id_range:
                print(f"\nStarting evaluation across task range...")
                task_id_start, task_id_end = args.task_id_range
                total_score, total_time, results, task_results = await evaluate_with_model_range(
                    env_instance, args.base_url, task_id_start, task_id_end, args.temperature, use_wandb=use_wandb
                )
            else:
                print(f"\nStarting evaluation ({args.samples} sample(s))...")
                total_score, total_time, results = await evaluate_with_model(
                    env_instance, args.base_url, args.task_id, args.seed, args.samples, args.temperature, use_wandb=use_wandb
                )
        
        # Calculate total samples
        if args.use_task_ids:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            task_ids_path = os.path.join(script_dir, TASK_IDS_FILE)
            with open(task_ids_path, 'r') as f:
                task_ids = json.load(f)
            total_samples = len(task_ids)
        elif args.task_id_range:
            total_samples = args.task_id_range[1] - args.task_id_range[0] + 1
        else:
            total_samples = args.samples

        summary = {
            'environment': args.env,
            'mode': 'chutes' if args.uid else 'direct',
            'evaluation_method': 'task_ids' if args.use_task_ids else ('range' if args.task_id_range else 'single'),
            'total_samples': total_samples,
            'total_score': total_score,
            'average_score': total_score / total_samples if total_samples > 0 else 0.0,
            'total_time': total_time,
            'average_time': total_time / total_samples if total_samples > 0 else 0.0,
            'results': results
        }

        # Add mode-specific info
        if args.uid:
            summary['uid'] = args.uid
        else:
            summary['base_url'] = args.base_url

        if args.task_id is not None:
            summary['task_id'] = args.task_id
        if args.task_id_range:
            summary['task_id_range'] = args.task_id_range
        if args.use_task_ids:
            summary['task_ids_source'] = TASK_IDS_FILE
        if args.seed is not None:
            summary['seed'] = args.seed
        
        summary['temperature'] = args.temperature
        
        # Save to JSON file if specified
        if args.output:
            output_path = args.output
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"\n✓ Results saved to: {output_path}")
            except Exception as e:
                print(f"\n✗ Failed to save results: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Evaluation Summary:")
        print(f"  Environment: {args.env}")

        if args.use_task_ids:
            print(f"  Method: Task IDs from file ({total_samples} tasks)")
        elif args.task_id_range:
            print(f"  Method: Range ({args.task_id_range[0]} - {args.task_id_range[1]}, {total_samples} tasks)")
        else:
            print(f"  Method: Single task")
            print(f"  Total Samples: {total_samples}")

        print(f"  Total Score: {total_score:.4f}")
        print(f"  Average Score: {summary['average_score']:.4f}")
        print(f"  Total Time: {total_time:.2f} seconds")
        print(f"  Average Time: {summary['average_time']:.2f} seconds/sample")

        # Show task-by-task results for range mode
        if args.task_id_range and task_results:
            print(f"\nResults by Task ID:")
            for task_id in sorted(task_results.keys()):
                task_info = task_results[task_id]
                status = "✓" if task_info.get('success', False) else "✗"
                score = task_info.get('score', 0.0)
                task_time = task_info.get('time', 0.0)
                print(f"  [{status}] Task {task_id}: score={score:.4f}, time={task_time:.2f}s")
                if task_info.get('error'):
                    print(f"      Error: {task_info['error']}")

        # Show individual results for single task mode with multiple samples
        elif args.samples > 1 and not args.task_id_range and not args.use_task_ids:
            print(f"\nDetailed Results:")
            for idx, r in enumerate(results):
                status = "✓" if r.get('success', False) else "✗"
                score = r.get('score', 0.0)
                latency = r.get('latency_seconds', r.get('latency', 0.0))
                print(f"  [{status}] Sample {idx+1}: score={score:.4f}, time={latency:.2f}s")
                if r.get('error'):
                    print(f"      Error: {r['error']}")

        print("=" * 60)
        
        # Finalize W&B
        if wandb_run:
            wandb_run.finish()
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        if wandb_run:
            wandb_run.finish()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        if wandb_run:
            wandb_run.finish()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
