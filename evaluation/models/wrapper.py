# evaluation/models/wrapper.py
import types
import torch
import torch.nn.functional as F
from tqdm import tqdm
from lm_eval.models.huggingface import HFLM

from evaluation.models.cache_manager import KVCacheManager


# evaluation/models/wrapper.py
def universal_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                                output_attentions=False, use_cache=False, **kwargs):
    # 1. 阶段判定与参数劫持
    q_len = hidden_states.shape[1]
    is_prefill = (q_len > 1 and (past_key_value is None or self._get_cache_len(past_key_value) == 0))

    needs_attention = getattr(self, "cache_manager", None) and self.cache_manager.policy is not None
    should_output_attentions = output_attentions or needs_attention

    # 调用原生底层 forward
    outputs = self._original_forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=should_output_attentions,
        use_cache=use_cache,
        **kwargs
    )

    if not use_cache or getattr(self, "cache_manager", None) is None or self.cache_manager.policy is None:
        # 如果未启用策略，必须根据 caller 期望的 output_attentions 进行退回
        if should_output_attentions and not output_attentions:
            attn_output = outputs[0]
            current_cache = outputs[2] if len(outputs) > 2 else (
                outputs[1] if len(outputs) == 2 and not isinstance(outputs[1], torch.Tensor) else None)
            if current_cache is None: current_cache = past_key_value
            return (attn_output, current_cache) if use_cache else (attn_output,)
        return outputs

    # ==========================================
    # [核心修复 1] 极其稳健的特征提取
    # 无论底层是 SDPA 还是 Eager，安全抽离 Weights 和 Cache
    # ==========================================
    attn_output = outputs[0]
    attn_weights = None
    current_kv_cache = None

    if len(outputs) == 2:
        if isinstance(outputs[1], torch.Tensor):
            attn_weights = outputs[1]
        else:
            current_kv_cache = outputs[1]
    elif len(outputs) >= 3:
        attn_weights = outputs[1]
        current_kv_cache = outputs[2]

    # 安全后备：如果 HF 模型是 inplace 更新且未返回 cache
    if current_kv_cache is None and use_cache:
        current_kv_cache = past_key_value

    # ==========================================
    # 2. 将 Cache 交给管家处理
    # ==========================================
    modified_kv_cache = self.cache_manager.on_layer_forward_end(
        past_key_values=current_kv_cache,
        layer_idx=self.layer_idx,
        is_prefill=is_prefill,
        attention_scores=attn_weights
    )

    # ==========================================
    # [核心修复 2] 严格遵循 Caller 协议返回
    # 彻底解决 LlamaDecoderLayer 读取错位的问题
    # ==========================================
    if output_attentions and use_cache:
        return attn_output, attn_weights, modified_kv_cache
    elif output_attentions and not use_cache:
        return attn_output, attn_weights
    elif not output_attentions and use_cache:
        return attn_output, modified_kv_cache
    else:
        return (attn_output,)


def apply_universal_patch(model, cache_manager: KVCacheManager):
    """遍历模型的所有层，注入通用的 forward 拦截器"""
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer.self_attn, "_original_forward"):
            layer.self_attn._original_forward = layer.self_attn.forward

        layer.self_attn.layer_idx = layer_idx
        layer.self_attn.cache_manager = cache_manager

        # 挂载辅助函数来安全获取 cache 长度 (适配不同的 transformers 版本)
        layer.self_attn._get_cache_len = lambda past_kv: past_kv[0].shape[-2] if isinstance(past_kv, tuple) and len(
            past_kv) > 0 else 0

        layer.self_attn.forward = types.MethodType(universal_attention_forward, layer.self_attn)
    print(">>> [System] Universal KV Cache Interceptor injected successfully.")
    return model


class EvaluatorHFLM(HFLM):
    """
    继承自 lm-eval 的 HFLM。将具体的压缩 Policy 在初始化时传入。
    """

    def __init__(self, pretrained: str, policy=None, prefill_fraction=0.1, max_length=4096, **kwargs):
        # 考虑到某些 Policy (如H2O) 强依赖 Eager Attention，在这里做一层安全检查
        if policy is not None and "attn_implementation" not in kwargs:
            kwargs["attn_implementation"] = "eager"

        super().__init__(pretrained=pretrained,
                         max_length=max_length,
                         **kwargs)

        self.prefill_fraction = prefill_fraction

        # 实例化管家并注入模型
        self.cache_manager = KVCacheManager(policy=policy)
        self._model = apply_universal_patch(self._model, self.cache_manager)

    # （注：在这里你可以保留你原来重写的 loglikelihood 等方法，
    # 但里面的 self._h2o_teacher_forced_loglikelihood 等硬编码应该被替换为统一逻辑）

    def loglikelihood(self, requests, disable_tqdm=False):
        """
        重写 lm-eval 的 loglikelihood。
        拦截请求，强制将原始请求切分成 Context (触发 Prefill) 和 Continuation (触发 Decode 驱逐)。
        这样无论底层挂载的是什么 Policy，都能在 PPL 测试中被正确调用。
        """
        modified_requests = []
        for req in requests:
            original_context = req.args[0]
            original_continuation = req.args[1]
            full_text = original_context + original_continuation

            # 使用 tokenizer 计算精确 token 长度
            tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

            # 保证至少有少量的 Token 作为前置 Prefill
            prefill_len = max(int(len(tokens) * self.prefill_fraction), 1)
            # 防止 prefill 把文本全占了
            prefill_len = min(prefill_len, len(tokens) - 1)

            # 解码回文本，重新构造 request
            new_context = self.tokenizer.decode(tokens[:prefill_len])
            new_continuation = self.tokenizer.decode(tokens[prefill_len:])

            req.args = (new_context, new_continuation)
            modified_requests.append(req)

        # 调回父类原生的执行逻辑，底层的 universal_attention_forward 会自动触发 Policy
        return super().loglikelihood(modified_requests, disable_tqdm=disable_tqdm)

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        """
        [关键重写] 强制自回归模拟的 Rolling PPL。
        为了确保 KV Cache 压缩策略在评估 PPL 时能触发单步 Decode 的淘汰逻辑，
        我们放弃 lm-eval 的并行 Teacher Forcing，改为手动的逐 Token 前向传播。
        """


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

                # 1. 划分 Prefill 和 Decode 区间
                split_idx = max(1, int(len(token_ids) * getattr(self, 'prefill_fraction', 0.1)))
                split_idx = min(split_idx, len(token_ids) - 1)

                prefix_ids = torch.tensor([token_ids[:split_idx]], device=device)
                target_ids = torch.tensor([token_ids[split_idx:]], device=device)

                # 2. 强制执行一次 Prefill
                # 这里 q_len > 1，会安全触发 CacheManager 的 process_prefill
                outputs = self._model(input_ids=prefix_ids, use_cache=True)
                past_key_values = outputs.past_key_values
                # 保存最后一个 token 的 logits，用于预测 Decode 阶段的第一个真实 Token
                last_logit = outputs.logits[:, -1:, :]

                total_logprob = 0.0
                decode_seq_len = target_ids.shape[1]

                # 3. 模拟真实的单步 Decode
                for i in range(decode_seq_len):
                    target_token = target_ids[:, i:i+1]

                    # 计算交叉熵并累加对数似然
                    log_probs = F.log_softmax(last_logit, dim=-1)
                    token_logprob = log_probs.gather(-1, target_token.unsqueeze(-1)).squeeze(-1).squeeze(-1)
                    total_logprob += token_logprob.item()

                    # 如果已经是最后一个目标 token，无需再向前推理
                    if i == decode_seq_len - 1:
                        break

                    # [核心] 单步输入！q_len == 1
                    # 完美触发 CacheManager 的 process_decode_step 进行缓存压缩
                    outputs = self._model(
                        input_ids=target_token,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    last_logit = outputs.logits

                results.append(total_logprob)

        return results