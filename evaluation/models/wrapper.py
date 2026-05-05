# evaluation/models/wrapper.py
import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm
from transformers.cache_utils import DynamicCache


def _get_attention_pre_hook():
    def pre_hook(module, args, kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")

        if hidden_states is not None and hidden_states.shape[1] == 1:
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = None

        return args, kwargs

    return pre_hook


def _get_attention_hook(cache_obj, layer_idx):
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) > 1:
            attn_weights = outputs[1]
            if attn_weights is not None:
                with torch.no_grad():
                    accumulated_score = cache_obj.reduce_attention(attn_weights)
                    cache_obj.current_attention_scores[layer_idx] = accumulated_score

                attn_weights.untyped_storage().resize_(0)

                new_outputs = list(outputs)
                new_outputs[1] = None
                return tuple(new_outputs)

        return outputs

    return hook


class EvaluatorHFLM(HFLM):
    def __init__(self, pretrained: str, cache_class=None, cache_kwargs=None,
                 prefill_fraction=0.1, max_length=4096, **kwargs):

        kwargs["attn_implementation"] = kwargs.get("attn_implementation", "eager")
        super().__init__(pretrained=pretrained, max_length=max_length, **kwargs)

        self.prefill_fraction = prefill_fraction
        self.cache_class = cache_class
        self.cache_kwargs = cache_kwargs or {}
        self._hooks = []

    def _setup_cache_and_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        if self.cache_class is None:
            return DynamicCache()

        cache_instance = self.cache_class(**self.cache_kwargs)
        needs_attn = getattr(cache_instance, "requires_attention", True)

        layers = self._model.model.layers
        for layer_idx, layer in enumerate(layers):
            if needs_attn:
                hook_handle = layer.self_attn.register_forward_hook(
                    _get_attention_hook(cache_instance, layer_idx)
                )
                self._hooks.append(hook_handle)

            pre_hook_handle = layer.self_attn.register_forward_pre_hook(
                _get_attention_pre_hook(), with_kwargs=True
            )
            self._hooks.append(pre_hook_handle)

        return cache_instance

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        iterator = requests if disable_tqdm else tqdm(requests, desc="Autoregressive PPL")

        self._model.eval()
        device = self._model.device

        with torch.no_grad():
            for req in iterator:
                text = req.args[0]
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)

                if getattr(self.tokenizer, "bos_token_id", None) is not None:
                    if len(token_ids) == 0 or token_ids[0] != self.tokenizer.bos_token_id:
                        token_ids = [self.tokenizer.bos_token_id] + token_ids

                if len(token_ids) < 3:
                    results.append(0.0)
                    continue

                split_idx = max(1, int(len(token_ids) * self.prefill_fraction))
                split_idx = min(split_idx, len(token_ids) - 1)

                prefix_ids = torch.tensor([token_ids[:split_idx]], device=device)
                target_ids = torch.tensor([token_ids[split_idx:]], device=device)

                print(f"\n" + "=" * 55)
                print(f"[Doc Monitor] Processing New Document")
                print(f"[Doc Monitor] Total Tokens   : {len(token_ids)}")
                print(f"[Doc Monitor] Prefill Tokens : {prefix_ids.shape[1]}")
                print(f"[Doc Monitor] Decode Steps   : {target_ids.shape[1]}")
                print("=" * 55)
                print()

                past_key_values = self._setup_cache_and_hooks()
                needs_attn = getattr(past_key_values, "requires_attention", False)

                # 1. 执行 Prefill
                # (注意：不再需要在这里显式打印监控，因为 Cache 内部在 update 期间会自动打印)
                outputs = self._model(
                    input_ids=prefix_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_attentions=needs_attn,
                    return_dict=True
                )

                last_logit = outputs.logits[:, -1:, :]
                total_logprob = 0.0
                decode_seq_len = target_ids.shape[1]

                # 2. 模拟单步 Decode (逐 Token 生成)
                for i in range(decode_seq_len):
                    target_token = target_ids[:, i:i + 1]

                    log_probs = F.log_softmax(last_logit, dim=-1)
                    token_logprob = log_probs.gather(-1, target_token.unsqueeze(-1)).squeeze()
                    total_logprob += token_logprob.item()

                    if i == decode_seq_len - 1:
                        break

                    outputs = self._model(
                        input_ids=target_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                        position_ids=torch.tensor([[split_idx + i]], device=device),
                        output_attentions=needs_attn,
                        return_dict=True
                    )

                    last_logit = outputs.logits

                print()
                results.append(total_logprob)

        for h in self._hooks:
            h.remove()

        return results