"""Meta-AttnLRP token-token interactions for generative LMs.

For each generation step t, recompute AttnLRP relevance explaining the predicted
token y_t. Sample coalitions over real input tokens; apply masking; refit
relevance; then for each real token i, fit an XGBRegressor with target
= relevance-at-i and extract max_order=1 Shapley values via
InterventionalTreeExplainer — yielding SV[i, j]@t. Average across t to obtain
a directed n x n interaction matrix (the Meta-AttnLRP interaction index).
Also compute the first-order AttnLRP heatmap (mean relevance across steps).
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from lxt.efficient import monkey_patch
from lxt.utils import clean_tokens
from transformers.models.gemma3 import modeling_gemma3

from xgboost import XGBRegressor
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer

from sampler import CoalitionSampler  # noqa: E402


def set_seeds(seed_value=42):
    """Set seeds for random, numpy, and torch."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SPECIAL_TOKENS = {
    "<bos>", "<start_of_turn>", "<end_of_turn>",
    r"<start\_of\_turn>", r"<end\_of\_turn>",
    "user", "model", r"\#", "\\#", r"\#\#", r"\\#\\#",
    "\n", r"\\n",
}


def apply_prompt_template(tokenizer, prompt, model_id=None):
    """Returns (text, used_template). `used_template=True` means a chat template
    already injected BOS, so the caller should tokenize with add_special_tokens=False
    to avoid a double BOS."""
    is_pt = model_id is not None and model_id.endswith("-pt")
    if not is_pt and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text, True
    return prompt, False


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

def identify_real_tokens(raw_tokens, cleaned_tokens):
    """Return indices of tokens that are not special / structural."""
    real = []
    for i, (raw, cln) in enumerate(zip(raw_tokens, cleaned_tokens)):
        stripped = cln.strip()
        if raw in SPECIAL_TOKENS or cln in SPECIAL_TOKENS or stripped in SPECIAL_TOKENS:
            continue
        if stripped == "":
            continue
        real.append(i)
    return real


# ---------------------------------------------------------------------------
# Relevance computation
# ---------------------------------------------------------------------------

def forward_backward(model_lxt, input_embeds, target_id=None):
    """One forward+backward. If target_id is None, pick argmax of next-token logits.

    Returns (relevance over sequence positions, target_id).
    """
    input_embeds = input_embeds.detach().requires_grad_(True)
    logits = model_lxt(inputs_embeds=input_embeds, use_cache=False).logits
    if target_id is None:
        target_id = int(logits[0, -1, :].detach().argmax().item())
    logits[0, -1, target_id].backward()
    relevance = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()[0]
    return relevance, target_id


def forward_backward_batch(model_lxt, input_embeds_batch, target_id):
    """Batched forward+backward explaining the same target_id across K samples.

    Summing per-sample target logits gives per-sample grads via autograd independence
    (sample k's logit only depends on sample k's embeds).

    Returns relevance tensor of shape (K, L) on CPU.
    """
    input_embeds_batch = input_embeds_batch.detach().requires_grad_(True)
    logits = model_lxt(inputs_embeds=input_embeds_batch, use_cache=False).logits
    logits[:, -1, target_id].sum().backward()
    relevance = (input_embeds_batch * input_embeds_batch.grad).float().sum(-1).detach().cpu()
    return relevance


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def build_masked_embeds(input_ids, embed_layer, real_idx, coalition, mode,
                        mask_token_id):
    """Apply a coalition mask.

    Args:
        input_ids: (1, L) tensor — full current sequence (input + generated).
        real_idx: sequence of positions within input_ids[0] that are players.
        coalition: bool array of length len(real_idx); True = keep.
        mode: "drop" | "zero_embed" | "mask_token".

    Returns:
        embeds: (1, L', D) input embeddings after masking.
        kept_to_orig: np.ndarray of length L' mapping new position -> original position.
    """
    L = input_ids.shape[1]
    real_idx = np.asarray(real_idx)
    masked_positions = real_idx[~np.asarray(coalition, dtype=bool)]

    if mode == "drop":
        keep = np.ones(L, dtype=bool)
        keep[masked_positions] = False
        kept_to_orig = np.where(keep)[0]
        embeds = embed_layer(input_ids[:, kept_to_orig])
        return embeds, kept_to_orig

    if mode == "mask_token":
        new_ids = input_ids.clone()
        new_ids[0, masked_positions] = mask_token_id
        return embed_layer(new_ids), np.arange(L)

    if mode == "zero_embed":
        embeds = embed_layer(input_ids).clone()
        embeds[0, masked_positions, :] = 0.0
        return embeds, np.arange(L)

    raise ValueError(f"Unknown mask mode: {mode}")


def build_masked_embeds_batch(seq_ids, embed_layer, real_idx, coalitions_batch,
                              mode, mask_token_id):
    """Batched version of build_masked_embeds for fixed-length modes.

    Args:
        seq_ids: (1, L) tensor.
        coalitions_batch: (K, n_real) bool numpy array.
        mode: "zero_embed" | "mask_token" (NOT "drop" — lengths would vary).

    Returns embeds of shape (K, L, D).
    """
    K = coalitions_batch.shape[0]
    L = seq_ids.shape[1]

    if mode == "mask_token":
        new_ids = seq_ids.repeat(K, 1)  # (K, L) — .repeat allocates fresh memory
        for k in range(K):
            masked_positions = real_idx[~coalitions_batch[k]]
            if len(masked_positions):
                new_ids[k, torch.as_tensor(masked_positions, device=new_ids.device)] = mask_token_id
        return embed_layer(new_ids)

    if mode == "zero_embed":
        out = embed_layer(seq_ids).repeat(K, 1, 1)  # (K, L, D)
        for k in range(K):
            masked_positions = real_idx[~coalitions_batch[k]]
            if len(masked_positions):
                out[k, torch.as_tensor(masked_positions, device=out.device), :] = 0.0
        return out

    raise ValueError(f"build_masked_embeds_batch does not support mode={mode}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def shapley_sampling_weights(n):
    w = np.zeros(n + 1, dtype=float)
    for k in range(1, n):
        w[k] = 1.0 / (k * (n - k))
    return w


def compute_shapley_attnlrp(model_lxt, tokenizer, prompt, budget=None,
                            mask_mode="mask_token", max_new_tokens=100,
                            coalition_batch_size=32, random_state=0,
                            normalize_per_step=False, verbose=True, model_id=None,
                            checkpoint_dir=None):
    """Main routine. Returns a dict with relevance_mean and interaction_matrix."""
    device = next(model_lxt.parameters()).device
    input_text, used_template = apply_prompt_template(tokenizer, prompt, model_id=model_id)
    input_ids = tokenizer(input_text, return_tensors="pt",
                          add_special_tokens=not used_template).input_ids.to(device)
    n_input = input_ids.shape[1]

    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    cleaned = clean_tokens(raw_tokens)
    real_idx = np.asarray(identify_real_tokens(raw_tokens, cleaned))
    n_real = len(real_idx)
    if verbose:
        print(f"[info] input len={n_input}, real tokens={n_real}", flush=True)

    if n_real < 2:
        raise ValueError(f"Need at least 2 real input tokens to compute interactions; got {n_real}.")
    n_minus = n_real - 1
    if budget is None or budget <= 0:
        budget = 32 * n_minus
    budget = min(budget, 2 ** n_minus)

    # Disable gradient checkpointing during Shapley: batched forward already replicates
    # activations K times, and re-checkpointing on top would roughly double backward cost.
    was_checkpointing = getattr(model_lxt, "is_gradient_checkpointing", False)
    model_lxt.gradient_checkpointing_disable()

    embed_layer = model_lxt.get_input_embeddings()
    mask_token_id = tokenizer.mask_token_id or tokenizer.unk_token_id or tokenizer.pad_token_id or 0
    if mask_mode == "mask_token" and tokenizer.mask_token_id is None and verbose:
        if tokenizer.unk_token_id is not None:
            fallback = f"unk_token_id={mask_token_id}"
        elif tokenizer.pad_token_id is not None:
            fallback = f"pad_token_id={mask_token_id}"
        else:
            fallback = "0"
        print(f"[warn] tokenizer has no mask_token; falling back to {fallback}", flush=True)

    # ---- Greedy generation with per-step relevance on original input ----
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for tok in ("<end_of_turn>", "<eos>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    current_ids = input_ids.clone()
    generated_ids = []
    step_relevances = []  # each: (n_input,) float array restricted to original input
    step_norm_scalars = []  # per-step max-abs for matching Shapley normalization
    gen_bar = tqdm(range(max_new_tokens), desc="generate", disable=not verbose,
                   mininterval=10, miniters=10)
    for t in gen_bar:
        embeds = embed_layer(current_ids)
        relevance, target_id = forward_backward(model_lxt, embeds, target_id=None)
        step_rel = relevance[:n_input].numpy()
        max_abs = float(np.abs(step_rel).max())
        step_norm_scalars.append(max_abs if max_abs > 0 else 1.0)
        if normalize_per_step and max_abs > 0:
            step_rel = step_rel / max_abs
        step_relevances.append(step_rel)
        generated_ids.append(target_id)
        gen_bar.set_postfix_str(repr(tokenizer.decode([target_id])[:20]))

        if target_id in stop_ids:
            break
        current_ids = torch.cat(
            [current_ids, torch.tensor([[target_id]], device=device)], dim=1)
    gen_bar.close()

    n_steps = len(generated_ids)
    if n_steps == 0:
        raise ValueError("No tokens generated (max_new_tokens=0 or immediate stop).")
    relevance_mean = np.stack(step_relevances, 0).mean(0)  # (n_input,)
    # Preallocate once so the Shapley step loop can slice instead of re-allocating.
    generated_ids_tensor = torch.tensor([generated_ids], device=device)  # (1, n_steps)

    # ---- Coalition sampling over (n_real - 1) players ----
    # For each target i, player i is always present; the remaining n_minus players
    # are sampled with proper Shapley weights for an n_minus-player game. A single
    # sample is reused across targets (the distribution does not depend on i).
    coalition_sampler = CoalitionSampler(
        n_players=n_minus,
        sampling_weights=shapley_sampling_weights(n_minus),
        enforce_empty_full=True,
        pairing_trick=True,
        random_state=random_state,
    )
    coalition_sampler.sample(budget)
    coalitions_minus = coalition_sampler.coalitions_matrix.astype(bool)  # (B, n_minus)
    n_coalitions = coalitions_minus.shape[0]
    if verbose:
        if n_coalitions != budget:
            print(f"[info] requested budget={budget}, actual n_coalitions={n_coalitions} "
                  f"(pairing_trick / enforce_empty_full may adjust)", flush=True)
        else:
            print(f"[info] sampled {n_coalitions} coalitions (budget={budget})", flush=True)
        print(f"[info] expected forward passes: n_steps * n_real * B = "
              f"{n_steps} * {n_real} * {n_coalitions} = {n_steps * n_real * n_coalitions}",
              flush=True)

    # ---- Per-step relevance under each coalition; fit XGB per target token ----
    interaction_matrix = np.zeros((n_real, n_real), dtype=np.float64)
    interaction_matrix_abs = np.zeros((n_real, n_real), dtype=np.float64)
    proxy = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4,
                         random_state=random_state, verbosity=0)
    ref = np.zeros((1, n_minus))
    query = np.ones((1, n_minus))

    coalitions_int8 = coalitions_minus.astype(np.int8)
    non_i_cols = [np.delete(np.arange(n_real), i) for i in range(n_real)]
    can_batch = mask_mode in ("zero_embed", "mask_token") and coalition_batch_size > 1

    try:
        step_bar = tqdm(list(enumerate(generated_ids)), desc="shapley-steps",
                        disable=not verbose, mininterval=30, miniters=1)
        for t, target_id in step_bar:
            seq_ids = torch.cat([input_ids, generated_ids_tensor[:, :t]], dim=1)
            # Scale batch ~1/L_t to keep peak activation memory roughly constant.
            batch_k = max(1, min(n_coalitions,
                                 int(coalition_batch_size * n_input / (n_input + t))))

            step_matrix = np.zeros((n_real, n_real), dtype=np.float64)
            for i in range(n_real):
                # Full n_real-wide coalitions for target i, with column i forced present.
                full_coalitions = np.ones((n_coalitions, n_real), dtype=bool)
                full_coalitions[:, non_i_cols[i]] = coalitions_minus
                y_i = np.zeros(n_coalitions, dtype=np.float32)

                if can_batch:
                    for b_start in range(0, n_coalitions, batch_k):
                        b_end = min(b_start + batch_k, n_coalitions)
                        batch_embeds = build_masked_embeds_batch(
                            seq_ids, embed_layer, real_idx, full_coalitions[b_start:b_end],
                            mask_mode, mask_token_id)
                        relevance_batch = forward_backward_batch(
                            model_lxt, batch_embeds, target_id).numpy()
                        if normalize_per_step:
                            relevance_batch = relevance_batch / step_norm_scalars[t]
                        y_i[b_start:b_end] = relevance_batch[:, real_idx[i]]
                else:
                    # Fallback: one-coalition-at-a-time (drop mode has variable seq length).
                    for b in range(n_coalitions):
                        embeds, kept_to_orig = build_masked_embeds(
                            seq_ids, embed_layer, real_idx, full_coalitions[b],
                            mask_mode, mask_token_id)
                        relevance, _ = forward_backward(model_lxt, embeds, target_id=target_id)
                        relevance_np = relevance.numpy()
                        if normalize_per_step:
                            relevance_np = relevance_np / step_norm_scalars[t]
                        if mask_mode == "drop":
                            orig_to_new = -np.ones(seq_ids.shape[1], dtype=np.int64)
                            orig_to_new[kept_to_orig] = np.arange(len(kept_to_orig))
                            # i is always present, so real_idx[i] is never dropped.
                            y_i[b] = relevance_np[orig_to_new[real_idx[i]]]
                        else:
                            y_i[b] = relevance_np[real_idx[i]]

                torch.cuda.empty_cache()

                if np.allclose(y_i, y_i[0]):
                    continue
                proxy.fit(coalitions_int8, y_i)
                explainer = InterventionalTreeExplainer(
                    proxy, data=ref, class_index=None,
                    index="SV", max_order=1, bool_tree=True,
                )
                interaction_values = explainer.explain_function(query)
                step_matrix[i, non_i_cols[i]] = [
                    float(interaction_values[(j,)]) for j in range(n_minus)
                ]
                # step_matrix[i, i] stays 0 by construction (target always present).

            interaction_matrix += step_matrix
            interaction_matrix_abs += np.abs(step_matrix)

            completed = t + 1
            if (checkpoint_dir is not None and completed >= 50
                    and completed % 10 == 0 and completed < n_steps):
                np.savez(
                    os.path.join(checkpoint_dir, "results_partial.npz"),
                    interaction_matrix=interaction_matrix / completed,
                    interaction_matrix_abs=interaction_matrix_abs / completed,
                    relevance_mean=relevance_mean,
                    real_idx=np.array(real_idx),
                    tokens_clean=np.array(cleaned, dtype=object),
                    generated_ids=np.array(generated_ids),
                    n_steps_completed=completed,
                    n_steps_total=n_steps,
                )
                if verbose:
                    print(f"[info] saved partial results at step {completed}/{n_steps}", flush=True)

        interaction_matrix /= n_steps
        interaction_matrix_abs /= n_steps
    finally:
        if was_checkpointing:
            model_lxt.gradient_checkpointing_enable()

    return {
        "tokens_clean": cleaned,
        "tokens_raw": raw_tokens,
        "real_idx": real_idx,
        "relevance_mean": relevance_mean,                # (n_input,)
        "interaction_matrix": interaction_matrix,        # (n_real, n_real)
        "interaction_matrix_abs": interaction_matrix_abs,  # mean_t |step_matrix|
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids),
        "n_input": n_input,
        "n_real": n_real,
        "budget_requested": budget,
        "budget_actual": n_coalitions,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CACHE_DIR = os.environ.get(
    "HF_HOME",
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
)

DEFAULT_PROMPT = "Is Sydney a good place to live? \nDo not include the following keywords:\n Sydney, good, place."

COALITION_BATCH_SIZE_BY_MODEL = {
    "google/gemma-3-1b-it": 256,
    "google/gemma-3-4b-it": 128,
    "google/gemma-3-12b-it": 144,
    "google/gemma-3-27b-it": 144,
    "google/gemma-3-1b-pt": 256,
    "google/gemma-3-4b-pt": 128,
    "google/gemma-3-12b-pt": 144,
    "google/gemma-3-27b-pt": 144,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--budget", type=int, default=0,
                        help="0 = auto (32 * (n-real-tokens - 1))")
    parser.add_argument("--mask-mode", type=str, default="mask_token",
                        choices=["zero_embed", "mask_token", "drop"])
    parser.add_argument("--coalition-batch-size", type=int, default=None)
    parser.add_argument("--normalize-relevance", action="store_true",
                        help="Enable per-step max-abs normalization of relevance (affects both heatmap and interactions).")
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--output-root", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                        help="Root dir for outputs; results go to <output_root>/<model_name>/<run_name>/")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    set_seeds(args.random_state)

    coalition_batch_size = args.coalition_batch_size
    if coalition_batch_size is None:
        coalition_batch_size = COALITION_BATCH_SIZE_BY_MODEL.get(args.model_id, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_id.split("/")[-1]
    out_dir = os.path.join(args.output_root, model_name, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] output dir: {out_dir}", flush=True)

    monkey_patch(modeling_gemma3, verbose=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=CACHE_DIR)
    model_lxt = modeling_gemma3.Gemma3ForCausalLM.from_pretrained(
        args.model_id, device_map=device, torch_dtype=torch.bfloat16,
        attn_implementation="eager", cache_dir=CACHE_DIR,
    )
    model_lxt.train()
    model_lxt.gradient_checkpointing_enable()
    for p in model_lxt.parameters():
        p.requires_grad = False
    print(f"[info] loaded {args.model_id}", flush=True)

    result = compute_shapley_attnlrp(
        model_lxt, tokenizer, args.prompt,
        budget=args.budget, mask_mode=args.mask_mode,
        max_new_tokens=args.max_new_tokens,
        coalition_batch_size=coalition_batch_size,
        random_state=args.random_state,
        normalize_per_step=args.normalize_relevance,
        model_id=args.model_id,
        checkpoint_dir=out_dir,
    )

    tokens_clean = result["tokens_clean"]
    real_idx = result["real_idx"]
    matrix = result["interaction_matrix"]
    matrix_abs = result["interaction_matrix_abs"]

    np.savez(
        os.path.join(out_dir, "results.npz"),
        interaction_matrix=matrix,
        interaction_matrix_abs=matrix_abs,
        relevance_mean=result["relevance_mean"],
        real_idx=np.array(real_idx),
        tokens_clean=np.array(tokens_clean, dtype=object),
        generated_ids=np.array(result["generated_ids"]),
    )

    print(f"\n[info] generated text: {result['generated_text']!r}", flush=True)
    print(f"[info] matrix shape: {matrix.shape}, |.| mean={np.abs(matrix).mean():.4g}, "
          f"diag mean={np.diag(matrix).mean():.4g}", flush=True)
    print("\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()
