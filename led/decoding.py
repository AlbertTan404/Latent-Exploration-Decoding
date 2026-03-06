from typing import List, Dict
import torch
from dataclasses import dataclass
import numpy as np
import torch.nn.functional as F


@dataclass
class LatentDecodingParams:
    # LED:
    method: str = None
    think_only: int = None
    d: int = None
    k: int = None
    ln: int = None
    do_exploit: int = None

    # DoLa:
    sub_method: str = None
    candidate_premature_layers: List[int] = None


global_decoding_params = LatentDecodingParams()


def set_global_decoding_params(_decoding_params: LatentDecodingParams):
    global global_decoding_params
    global_decoding_params = _decoding_params


def get_global_decoding_params():
    return global_decoding_params


def latent_exploration_decoding(
    logits_output,
    decoding_params: LatentDecodingParams,
    is_all_greedy: bool,
    temperatures: torch.Tensor,
    think_only_mask: torch.Tensor=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    d_logits: torch.Tensor = logits_output.next_token_logits
    B, d, V = d_logits.shape

    if not is_all_greedy:
        if isinstance(temperatures, torch.Tensor):
            d_logits.div_(temperatures.unsqueeze(1))
        else:
            d_logits.div_(temperatures)
    d_logits[:] = torch.softmax(d_logits, dim=-1)
    d_probs = d_logits
    del d_logits

    origin_probs = d_probs[:, 0, :]  # (B, V)

    k = decoding_params.k
    d = decoding_params.d

    origin_topk_probs, origin_topk_ids = origin_probs.topk(k=k, dim=-1)  # (B, k)
    dk_probs = d_probs.gather(dim=-1, index=origin_topk_ids.unsqueeze(1).expand(-1, d, -1)).clamp_min_(eps)  # (B, d, k)
    
    if is_all_greedy:
        batch_next_origin_indices = origin_topk_probs.argmax(dim=-1).view(B, 1)  # (B, 1)
    else:
        batch_next_origin_indices = torch.multinomial(origin_topk_probs, num_samples=1)  # (B, 1)
    origin_probs_top1 = origin_topk_probs[:, :1]  # (B, 1, k)
    if decoding_params.do_exploit:
        exploration_mask = torch.bernoulli(1 - origin_probs_top1).bool()  # (B, 1)
    else:
        exploration_mask = torch.ones_like(think_only_mask).bool()  # (B, 1)
    new_probs = torch.cumsum(dk_probs, dim=1)  # (B, d, k)
    new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)  # (B, d, k)
    entropy = -torch.sum(new_probs * torch.log(new_probs.clamp_min_(1e-9)), dim=-1)  # (B, d)

    target_layer = torch.argmax(entropy, dim=-1).view(B, 1, 1).repeat(1, 1, k)  # (B, 1, k)
    new_probs = new_probs.gather(dim=1, index=target_layer).squeeze(1)  # (B, k)
    if is_all_greedy:
        batch_next_topk_indices = new_probs.argmax(dim=-1).view(B, 1)
    else:
        batch_next_topk_indices = torch.multinomial(new_probs, num_samples=1).view(B, 1)
    batch_next_topk_indices = torch.where(exploration_mask.logical_and(think_only_mask), batch_next_topk_indices, batch_next_origin_indices)

    batch_next_token_ids = origin_topk_ids.gather(dim=1, index=batch_next_topk_indices).squeeze(1)
    return batch_next_token_ids
    

def dola_decoding(
    logits_output,
    decoding_params: LatentDecodingParams,
    is_all_greedy: bool,
    temperatures: torch.Tensor,
    topks: torch.Tensor,  # not used since dola has its own filtering
    topps: torch.Tensor,  # same above
) -> torch.Tensor:
    layerwise_logits = logits_output.next_token_logits
    B, n_all_layers, vocab_size = layerwise_logits.shape

    if not is_all_greedy:
        if isinstance(temperatures, torch.Tensor):
            layerwise_logits.div_(temperatures.unsqueeze(1))
        else:
            layerwise_logits.div_(temperatures)

    final_logits = layerwise_logits[:, -1, :]

    candidate_premature_layers = decoding_params.candidate_premature_layers
    candidate_premature_logits = {
        layer: layerwise_logits[:, idx, :] for idx, layer in enumerate(candidate_premature_layers)
    }

    del layerwise_logits
    logits = _dola_select_contrast(
        candidate_premature_layers, candidate_premature_logits, final_logits
    )
    if is_all_greedy:
        batch_next_token_ids = logits.argmax(dim=-1).view(B)
    else:
        batch_next_token_ids = torch.multinomial(logits.softmax(dim=-1), num_samples=1).view(B)
    return batch_next_token_ids


def _dola_select_contrast(
    candidate_premature_layers: List[int],
    candidate_premature_logits: Dict[int, torch.FloatTensor],
    final_logits: torch.FloatTensor,
) -> torch.FloatTensor:
    if len(candidate_premature_layers) == 1:
        base_logits = candidate_premature_logits[candidate_premature_layers[0]]
        final_logits, base_logits = _relative_top_filter(final_logits, base_logits)
        logits = final_logits - base_logits
        return logits

    # 1. Stacking all premature_layers into a new dimension
    stacked_premature_layers = torch.stack([candidate_premature_logits[i] for i in candidate_premature_layers], dim=0)

    # 2. Calculate the softmax values for mature_layer and all premature_layers
    # shape: (batch_size, vocab_size)
    softmax_mature_layer = F.softmax(final_logits, dim=-1)
    # shape: (num_premature_layers, batch_size, vocab_size)
    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)

    # 3. Calculate the average distribution
    # shape: (num_premature_layers, batch_size, vocab_size)
    avg_dist = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)

    # 4. Calculate log-softmax for the KL divergence
    # shape: (batch_size, vocab_size)
    log_softmax_mature_layer = F.log_softmax(final_logits, dim=-1)
    # shape: (num_premature_layers, batch_size, vocab_size)
    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)

    # 5. Calculate the KL divergences and then the JS divergences
    # shape: (num_premature_layers, batch_size)
    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], avg_dist, reduction="none").mean(-1)
    # shape: (num_premature_layers, batch_size)
    kl2 = F.kl_div(log_softmax_premature_layers, avg_dist, reduction="none").mean(-1)
    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

    # 6. Reduce the batchmean
    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
    premature_layer = candidate_premature_layers[int(js_divs.argmax().item())]

    base_logits = candidate_premature_logits[premature_layer]
    final_logits, base_logits = _relative_top_filter(final_logits, base_logits)
    logits = final_logits - base_logits
    return logits


def _relative_top_filter(
    scores: torch.FloatTensor,
    baseline_scores: torch.FloatTensor,
    relative_top: float = 0.1,
    filter_value: float = -float("Inf"),
    base_filter_value=-1e-3,
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1)
    baseline_scores_normalized = baseline_scores.log_softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    baseline_scores_normalized[scores_normalized < probs_thresh] = base_filter_value
    scores_normalized[scores_normalized < probs_thresh] = filter_value
    return scores_normalized, baseline_scores_normalized
