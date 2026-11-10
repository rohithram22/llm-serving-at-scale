"""Smoke test + theory demo: tokenization, chat template, prefill vs decode,
and the KV cache (with vs without), all measured on this machine.

Usage: python scripts/smoke_test.py [--max-new-tokens 64]
Writes: results/smoke_test.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# PyTorch defaulted to 1 thread in this container; use every core we have.
torch.set_num_threads(os.cpu_count() or 1)


def section(title: str) -> None:
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def eos_ids(model, tok) -> set[int]:
    ids = set()
    for v in (model.generation_config.eos_token_id, tok.eos_token_id):
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(v)
    return ids


@torch.inference_mode()
def generate_with_kv_cache(model, input_ids, max_new_tokens, stop_ids):
    """Prefill once, then feed ONE new token per step, reusing cached K/V."""
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)          # PREFILL: whole prompt
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    ttft = time.perf_counter() - t0

    generated = [next_id.item()]
    t1 = time.perf_counter()
    while len(generated) < max_new_tokens and generated[-1] not in stop_ids:
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)  # DECODE: 1 token
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_id.item())
    decode_s = time.perf_counter() - t1
    return generated, ttft, decode_s


@torch.inference_mode()
def generate_without_cache(model, input_ids, max_new_tokens, stop_ids):
    """Re-run the ENTIRE growing sequence through the model every step."""
    seq = input_ids
    generated, step_times = [], []
    while len(generated) < max_new_tokens and (not generated or generated[-1] not in stop_ids):
        t = time.perf_counter()
        out = model(input_ids=seq, use_cache=False)
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        step_times.append(time.perf_counter() - t)
        generated.append(next_id.item())
        seq = torch.cat([seq, next_id], dim=-1)
    return generated, step_times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    results = {"model_id": MODEL_ID, "torch_threads": torch.get_num_threads()}

    # ---------------------------------------------------------------- load
    section("1. Load model (fp32 on CPU)")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    weights_gb = n_params * 4 / 1024**3
    print(f"Loaded in {load_s:.1f}s | params: {n_params / 1e6:.0f}M | fp32 weights ~{weights_gb:.2f} GB")
    print(f"torch threads: {torch.get_num_threads()}")
    results.update(load_seconds=round(load_s, 2), params_millions=round(n_params / 1e6, 1),
                   weights_gb_fp32=round(weights_gb, 2))

    # ------------------------------------------------------------- KV math
    section("2. KV cache size (from config.json)")
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    per_token_fp16 = 2 * cfg.num_hidden_layers * kv_heads * head_dim * 2
    print(f"layers={cfg.num_hidden_layers} q_heads={cfg.num_attention_heads} "
          f"kv_heads={kv_heads} head_dim={head_dim}")
    print(f"KV cache per token (fp16): {per_token_fp16 / 1024:.1f} KB")
    for ctx in (1_000, 8_000, 32_000):
        print(f"  one request with {ctx:>6,} tokens -> {per_token_fp16 * ctx / 1024**2:8.1f} MB")
    results["kv_bytes_per_token_fp16"] = per_token_fp16

    # -------------------------------------------------------- tokenization
    section("3. Tokenization")
    text = "Serving LLMs at scale is mostly a memory-bandwidth problem."
    ids = tok(text)["input_ids"]
    print(f"{len(text.split())} words -> {len(ids)} tokens")
    print("pieces:", [tok.decode([i]) for i in ids])

    # ------------------------------------------------------- chat template
    section("4. Chat template (what the model actually sees)")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a KV cache is in two sentences."},
    ]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt_text)
    input_ids = tok(prompt_text, return_tensors="pt")["input_ids"]
    print(f"prompt length: {input_ids.shape[-1]} tokens")
    stop = eos_ids(model, tok)

    # ----------------------------------------------------- with KV cache
    section(f"5. Generate WITH KV cache (greedy, up to {args.max_new_tokens} tokens)")
    gen_c, ttft, decode_s = generate_with_kv_cache(model, input_ids, args.max_new_tokens, stop)
    n_dec = max(len(gen_c) - 1, 1)
    tps = n_dec / decode_s if decode_s > 0 else float("nan")
    print(tok.decode(gen_c, skip_special_tokens=True))
    print(f"\nTTFT (prefill): {ttft * 1000:.0f} ms")
    print(f"decode: {n_dec} tokens in {decode_s:.2f}s -> {tps:.1f} tokens/s "
          f"({decode_s / n_dec * 1000:.0f} ms/token)")
    print(f"effective weight-read bandwidth ~{weights_gb * tps:.0f} GB/s "
          f"(each decode step reads all {weights_gb:.1f} GB of weights)")
    results.update(prompt_tokens=int(input_ids.shape[-1]), ttft_ms=round(ttft * 1000, 1),
                   decode_tokens_per_s=round(tps, 2), generated_tokens=len(gen_c))

    # -------------------------------------------------- without KV cache
    section("6. Generate WITHOUT KV cache (recompute everything each step)")
    print("(this one is slow on purpose: about 1 minute)")
    gen_n, steps = generate_without_cache(model, input_ids, args.max_new_tokens, stop)
    total_n = sum(steps)
    total_c = ttft + decode_s
    print(f"step 1: {steps[0] * 1000:.0f} ms | last step: {steps[-1] * 1000:.0f} ms "
          f"(gets slower as the sequence grows)")
    print(f"total: no-cache {total_n:.2f}s vs cache {total_c:.2f}s -> "
          f"KV cache speedup {total_n / total_c:.1f}x")
    same = gen_n == gen_c
    print(f"outputs identical: {same}  (greedy: should be True; rare float noise can flip a token)")
    results.update(no_cache_total_s=round(total_n, 2), cache_total_s=round(total_c, 2),
                   kv_cache_speedup=round(total_n / total_c, 2), outputs_identical=same)

    out_path = Path("results/smoke_test.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()