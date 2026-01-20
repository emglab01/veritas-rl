import asyncio
import affine as af
from dotenv import load_dotenv
import os
import sys
import logging
import wandb
import time
import datetime
import json
af.trace()

# Load .env file from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up from benchmark/ to evaluate/ to project root
env_file = os.path.join(project_root, '.env')
if os.path.exists(env_file):
    load_dotenv(env_file)
else:
    load_dotenv()  # Fallback to default location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    model_name = input("Enter model name: ")
    model_name += time.strftime("-%Y%m%d%H%M%S")
    logging.info(f"Running benchmark for model: {model_name}")
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    
    # Load credentials from .env file
    api_key = os.getenv("CHUTES_API_KEY")
    chutes_user = os.getenv("CHUTES_USER")
    hf_user = os.getenv("HF_USER")
    hf_repo = os.getenv("HF_REPO")
    
    # Validate required environment variables
    missing_vars = []
    if not api_key:
        missing_vars.append("CHUTES_API_KEY")
    if not chutes_user:
        missing_vars.append("CHUTES_USER")
    if not hf_user:
        missing_vars.append("HF_USER")
    if not hf_repo:
        missing_vars.append("HF_REPO")
    
    if missing_vars:
        print("\n   ❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"      - {var}")
        print("\n   Please add them to your .env file:")
        print("      CHUTES_API_KEY=your-api-key")
        print("      CHUTES_USER=your-chutes-username")
        print("      HF_USER=your-huggingface-username")
        print("      HF_REPO=your-huggingface-repo")
        print("\n   Or set them as environment variables:")
        for var in missing_vars:
            print(f"      export {var}='your-value'")
        sys.exit(1)
    
    run = wandb.init(project="LLM-Benchmark", config={}, name=f"{model_name}")
    slug = f"{chutes_user}-{hf_user}-{hf_repo}".lower()
    print(f"slug: {slug}")

    # Get environment names from .env (AFFINE_ENV_LIST) or fallback to default
    env_list_str = os.getenv("AFFINE_ENV_LIST", "")
    if env_list_str:
        env_names = [name.strip().split(":")[-1].upper() for name in env_list_str.split(",") if name.strip()]
        print(f"env_names: {env_names}")
    else:
        # Default set if none specified
        env_names = [
            "affine:abd", "affine:ded", "affine:sat",
            "agentgym:alfworld", "agentgym:sciworld",
            "agentgym:webshop", "agentgym:textcraft", "agentgym:babyai"
        ]

    envs = []
    for name in env_names:
        # Parse prefix and env short name
        if ":" in name:
            prefix, env_key = name.split(":", 1)
            class_name = env_key.strip().upper()
            # Map prefix to namespace (if "affine", use af.ABD, etc.; if "agentgym", use af.ALFWORLD, etc.)
            # Try dynamic lookup, fallback to logging
            try:
                env_cls = getattr(af, class_name)
                envs.append(env_cls())
            except AttributeError:
                logging.warning(f"No environment class found in affine for: {class_name}")
        else:
            # If no prefix, assume agentgym style
            class_name = name.strip().upper()
            try:
                env_cls = getattr(af, class_name)
                envs.append(env_cls())
            except AttributeError:
                logging.warning(f"No environment class found in affine for: {class_name}")

    env_stats = {env.__class__.__name__: {"score_sum":0.0,"dt_sum":0.0,"latency_sum":0.0,"count":0,"success_sum":0.0} for env in envs}
    task_idx = 2000
    while task_idx < 2500:
        for env in envs:
            env_name = env.__class__.__name__
            if env_name == "ABD" or env_name == "DED" or env_name == "SAT":
                env_name = "affine:" + env_name.lower()
            else:
                env_name = "agentgym:" + env_name.lower()
            evaluation = await env.evaluate(
            model=f"{hf_user}/{hf_repo}",
            base_url=f"https://{slug}.chutes.ai/v1",
            temperature=0.0,
            task_id=task_idx
            )
            with open(f"evaluation.txt", "a") as f:
                f.write(f"{task_idx}: {evaluation.score}\n")
            env_stats[env.__class__.__name__]["score_sum"] += evaluation.score
            env_stats[env.__class__.__name__]["count"] += 1
            env_stats[env.__class__.__name__]["avg_score"] = env_stats[env.__class__.__name__]["score_sum"] / env_stats[env.__class__.__name__]["count"]
            logging.info(f"\n{env_name} Evaluation Result:")
            logging.info(evaluation)
            log_dict = {
                    f"{env_name}/score": evaluation.score,
                    f"{env_name}/avg_score": env_stats[env.__class__.__name__]["avg_score"],
                    f"{env_name}/count": env_stats[env.__class__.__name__]["count"],
                }
            wandb.log(log_dict)
        task_idx += 1
if __name__ == "__main__":
    asyncio.run(main())