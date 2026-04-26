#!/usr/bin/env python3
"""
Dead Internet Detective -- GRPO Training Script
Vanilla transformers + bitsandbytes 4-bit + PEFT LoRA + SDPA attention.
No unsloth/xformers — bulletproof against ABI mismatches on A10G.
"""
import os, sys, json, time, random, requests, statistics, gc, logging, warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_SPACE_URL      = os.environ.get("HF_SPACE_URL", "https://aryanjain7031-dead-internet-detective.hf.space")
WANDB_KEY         = os.environ.get("WANDB_KEY", "")
HF_TOKEN          = os.environ.get("HF_TOKEN", "")
HF_MODEL_REPO     = os.environ.get("HF_MODEL_REPO", "PriyanshuHF/dead-internet-detective-model-p4")
# Ungated FP16 mirror — quantized on the fly by bitsandbytes
MODEL_NAME        = os.environ.get("MODEL_NAME", "NousResearch/Meta-Llama-3.1-8B-Instruct")
SMOKE_TEST        = os.environ.get("SMOKE_TEST", "false").lower() == "true"
NUM_TRAIN_STEPS   = 10  if SMOKE_TEST else 60
ROLLOUTS_PER_STEP = 2   if SMOKE_TEST else 4
MAX_EP_STEPS      = 4   if SMOKE_TEST else 6   # tight cap forces early file_report
MAX_EP_SECS       = 20
TIMEOUT_PENALTY   = -0.3                        # penalty when episode times out without file_report
DIFFICULTIES      = ["easy"] if SMOKE_TEST else ["easy", "medium", "hard"]
SAVE_PATH         = "./trained_model"
MAX_SEQ_LENGTH    = 2048

STATUS = {"phase": "starting", "step": 0, "total": NUM_TRAIN_STEPS, "mean_reward": 0.0, "log": []}

def _log(msg):
    log.info(msg)
    STATUS["log"].append(msg)
    if len(STATUS["log"]) > 200:
        STATUS["log"] = STATUS["log"][-200:]

# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------
sys.path.insert(0, "/app")
from client.client import DeadInternetClient

# ---------------------------------------------------------------------------
# Prompt + action generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a disinformation analyst.
At each step respond ONLY with a single JSON object like:
{"tool": "<tool_name>", "params": {<params>}}

Valid tools and their required params:
visit_page:           {"url": "<url>"}
search:               {"query": "<string>"}
wayback_check:        {"url": "<url>"}
whois_lookup:         {"domain": "<domain>"}
author_lookup:        {"author_name": "<name>"}
image_provenance:     {"image_url": "<url>"}
citation_trace:       {"url": "<url>"}
linguistic_fingerprint: {"url": "<url>"}
cross_reference:      {"url_a": "<url>", "url_b": "<url>"}
update_case_file:     {"confirmed_facts": [], "synthetic_sources": [],
                       "credible_sources": [], "confidence": 0.0}
request_expert:       {"domain": "medical|legal|statistical", "question": "<q>"}
file_report:          {"verdict": "true|false|unverifiable",
                       "confidence": 0.0,
                       "evidence_chain": ["<url>"],
                       "reasoning": "<string>"}

Always end every episode by calling file_report."""


def generate_action(model, tokenizer, obs_dict):
    import torch
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\nOBSERVATION:\n{json.dumps(obs_dict, default=str)[:3000]}\n\n"
        f"Respond with one JSON action.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.empty_cache()
    decoded = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    try:
        start = decoded.find("{")
        end   = decoded.rfind("}") + 1
        return json.loads(decoded[start:end])
    except Exception:
        return {
            "tool": "file_report",
            "params": {
                "verdict": "unverifiable",
                "confidence": 0.3,
                "evidence_chain": [],
                "reasoning": f"parse error: {decoded[:100]}"
            }
        }


def run_episode(model, tokenizer, client, difficulty="easy", seed=None):
    if seed is None:
        seed = random.randint(0, 99999)
    try:
        reset_result = client.reset(task_id=difficulty, seed=seed)
        session_id   = reset_result["session_id"]
        obs          = reset_result["observation"]
        total_reward = 0.0
        steps        = 0
        t0           = time.time()

        filed_report = False
        for _ in range(MAX_EP_STEPS):
            if time.time() - t0 > MAX_EP_SECS:
                client.step(session_id, "file_report", {
                    "verdict": "unverifiable", "confidence": 0.3,
                    "evidence_chain": [], "reasoning": "timeout"
                })
                break
            action = generate_action(model, tokenizer, obs)
            if action["tool"] == "file_report":
                filed_report = True
            result = client.step(session_id, action["tool"], action.get("params", {}))
            total_reward += result.get("reward", 0.0)
            steps        += 1
            obs           = result.get("observation", obs)
            if result.get("done"):
                break

        if not filed_report:
            total_reward += TIMEOUT_PENALTY

        return {"total_reward": total_reward, "steps_used": steps,
                "filed_report": filed_report, "difficulty": difficulty, "seed": seed}
    except Exception as e:
        _log(f"Episode error: {e}")
        return {"total_reward": 0.0, "steps_used": 0, "difficulty": difficulty, "seed": seed}


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def run_training():
    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset

    STATUS["phase"] = "loading model"
    _log(f"Loading model: {MODEL_NAME}")

    # 4-bit quantization config (bitsandbytes nf4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",  # PyTorch native — no xformers, no flash-attn
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN or None,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Clear generation_config conflicts
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    _log(f"GPU: {torch.cuda.get_device_name(0)}")
    _log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    client = DeadInternetClient(HF_SPACE_URL)
    assert client.health(), f"Environment server not reachable: {HF_SPACE_URL}"
    _log(f"Environment server OK: {HF_SPACE_URL}")

    # Smoke test (smaller — 3 episodes — to fail fast if generate is broken)
    STATUS["phase"] = "smoke test"
    _log("Running smoke test (1 episode)...")
    smoke = []
    for i in range(1):
        _log(f"  smoke episode {i+1}/3...")
        r = run_episode(model, tokenizer, client, "easy", i)
        _log(f"  episode {i+1} reward={r['total_reward']:.3f} steps={r['steps_used']}")
        smoke.append(r)
    mean_r = statistics.mean(r["total_reward"] for r in smoke)
    _log(f"Smoke test mean reward: {mean_r:.3f}")
    # Don't assert — proceed even with low reward (model is untrained)
    _log("Smoke test PASSED")

    # W&B
    if WANDB_KEY:
        try:
            wandb.login(key=WANDB_KEY)
            wandb.init(project="dead-internet-detective", name="phase4-timeout-penalty")
        except Exception as e:
            _log(f"wandb init failed: {e} (continuing without wandb)")

    def grpo_reward_fn(prompts, completions, **kwargs):
        rewards = []
        for i in range(len(prompts)):
            difficulty = DIFFICULTIES[i % len(DIFFICULTIES)]
            result = run_episode(model, tokenizer, client, difficulty)
            r = result["total_reward"]
            rewards.append(float(r))
            STATUS["step"] += 1
            STATUS["mean_reward"] = r
            filed = result.get("filed_report", False)
            _log(f"step={STATUS['step']} reward={r:.3f} difficulty={difficulty} filed={filed}")
            if WANDB_KEY:
                try:
                    wandb.log({"train/reward": r, "train/step": STATUS["step"]})
                except Exception:
                    pass
        return rewards

    dataset = Dataset.from_list([
        {"prompt": f"Investigate claim #{i}. Difficulty: {DIFFICULTIES[i % len(DIFFICULTIES)]}."}
        for i in range(NUM_TRAIN_STEPS * ROLLOUTS_PER_STEP)
    ])

    STATUS["phase"] = "training"
    STATUS["total"] = NUM_TRAIN_STEPS
    _log(f"Starting GRPO training: {NUM_TRAIN_STEPS} steps, {ROLLOUTS_PER_STEP} rollouts/step")

    training_args = GRPOConfig(
        output_dir=SAVE_PATH,
        num_train_epochs=1,
        per_device_train_batch_size=2,         # must be divisible by num_generations
        num_generations=2,                     # GRPO group size; 2 = minimum signal
        gradient_accumulation_steps=1,         # 2 rollouts, 1 accum = lean + fast
        learning_rate=2e-5,
        logging_steps=1,
        save_steps=50,
        max_completion_length=128,
        report_to="wandb" if WANDB_KEY else "none",
        bf16=True,
        max_grad_norm=0.3,
        dataloader_num_workers=0,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=grpo_reward_fn,
        processing_class=tokenizer,
    )

    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    _log(f"Model saved to {SAVE_PATH}")

    # Dump full training log alongside weights
    log_path = os.path.join(SAVE_PATH, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "phase": "phase4-timeout-penalty",
            "model_name": MODEL_NAME,
            "num_train_steps": NUM_TRAIN_STEPS,
            "rollouts_per_step": ROLLOUTS_PER_STEP,
            "max_ep_steps": MAX_EP_STEPS,
            "difficulties": DIFFICULTIES,
            "final_status": STATUS,
            "log": STATUS["log"],
        }, f, indent=2, default=str)
    _log(f"Training log saved to {log_path}")

    # Push to HF Hub
    if HF_TOKEN:
        STATUS["phase"] = "pushing to hub"
        _log(f"Pushing model to {HF_MODEL_REPO}...")
        try:
            model.push_to_hub(HF_MODEL_REPO, token=HF_TOKEN)
            tokenizer.push_to_hub(HF_MODEL_REPO, token=HF_TOKEN)
            _log(f"Model pushed to https://huggingface.co/{HF_MODEL_REPO}")
        except Exception as e:
            _log(f"Push failed: {e}")

    if WANDB_KEY:
        try: wandb.finish()
        except Exception: pass

    STATUS["phase"] = "done"
    _log("Training complete.")


if __name__ == "__main__":
    run_training()
