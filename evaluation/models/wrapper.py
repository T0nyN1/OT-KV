# evaluation/models/wrapper.py
import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm
from transformers.cache_utils import DynamicCache


def _get_attention_hook(cache_obj, layer_idx):
    """
    PyTorch Forward Hook: 负责将 Attention 模块输出的权重静默传递给 Cache 对象。
    """

    def hook(module, inputs, outputs):
        # 在 HuggingFace 模型中，Attention 层的输出通常是一个 tuple:
        # (attn_output, attn_weights, past_key_value)
        # 我们只需要第二个元素 attn_weights
        if isinstance(outputs, tuple) and len(outputs) > 1:
            attn_weights = outputs[1]
            if attn_weights is not None:
                # 将权重存入 Cache 实例的暂存区，供其内部的 update/process 方法使用
                # 使用 detach() 避免梯度残留
                cache_obj.current_attention_scores[layer_idx] = attn_weights.detach()
        return outputs

    return hook


class EvaluatorHFLM(HFLM):
    """
    继承自 lm-eval 的 HFLM。
    通过挂载 Hook 和 注入自定义 Cache 类来实现 KV Cache 压缩策略。
    """

    def __init__(self, pretrained: str, cache_class=None, cache_kwargs=None,
                 prefill_fraction=0.1, max_length=4096, **kwargs):

        # 强制要求模型输出 Attention，否则 Hook 拿不到数据
        kwargs["attn_implementation"] = kwargs.get("attn_implementation", "eager")

        # 初始化父类（加载分词器和模型）
        super().__init__(pretrained=pretrained, max_length=max_length, **kwargs)

        self.prefill_fraction = prefill_fraction
        self.cache_class = cache_class
        self.cache_kwargs = cache_kwargs or {}

        # 用于存储 hook 句柄，方便后续清理
        self._hooks = []

    def _setup_cache_and_hooks(self):
        """
        为当前的推理任务准备环境：
        1. 实例化具体的 Cache 对象
        2. 注册 Forward Hooks
        """
        # 清理之前可能存在的 hooks
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # 1. 如果没有指定策略，则使用原生的 DynamicCache (Baseline)
        if self.cache_class is None:
            return DynamicCache()

        # 2. 实例化自定义压缩 Cache (如 H2OCache)
        cache_instance = self.cache_class(**self.cache_kwargs)

        # 3. 遍历模型所有层，在 Attention 模块上挂载 Hook
        # 这里的路径 'model.layers' 适用于 Llama/Mistral，其他模型可能需微调
        layers = self._model.model.layers
        for layer_idx, layer in enumerate(layers):
            # 找到每一层的 self_attn 模块
            hook_handle = layer.self_attn.register_forward_hook(
                _get_attention_hook(cache_instance, layer_idx)
            )
            self._hooks.append(hook_handle)

        return cache_instance

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        """
        重写 PPL 评测核心逻辑。
        通过手动循环确保每一步都能触发 Cache 对象的压缩逻辑。
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

                # 划分 Prefill (Context) 和 Decode (Continuation) 区域
                split_idx = max(1, int(len(token_ids) * self.prefill_fraction))
                split_idx = min(split_idx, len(token_ids) - 1)

                prefix_ids = torch.tensor([token_ids[:split_idx]], device=device)
                target_ids = torch.tensor([token_ids[split_idx:]], device=device)
                
                decode_seq_len = target_ids.shape[1]

                # ==========================================
                # [新增] 宏观文档级别监控
                # ==========================================
                print(f"\n" + "=" * 55)
                print(f"[Doc Monitor] Processing New Document")
                print(f"[Doc Monitor] Total Tokens   : {len(token_ids)}")
                print(f"[Doc Monitor] Prefill Tokens : {prefix_ids.shape[1]}")
                print(f"[Doc Monitor] Decode Steps   : {decode_seq_len}")
                print("=" * 55)

                past_key_values = self._setup_cache_and_hooks()

                # 1. 执行 Prefill
                # 在 forward 过程中，HF 模型会自动调用 past_key_values.update(...)
                outputs = self._model(
                    input_ids=prefix_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_attentions=True,  # 确保输出以供 Hook 抓取
                    return_dict=True
                )

                last_logit = outputs.logits[:, -1:, :]
                total_logprob = 0.0
                decode_seq_len = target_ids.shape[1]

                # 2. 模拟单步 Decode (逐 Token 生成)
                for i in range(decode_seq_len):
                    target_token = target_ids[:, i:i + 1]

                    # 计算对数似然
                    log_probs = F.log_softmax(last_logit, dim=-1)
                    token_logprob = log_probs.gather(-1, target_token.unsqueeze(-1)).squeeze()
                    total_logprob += token_logprob.item()

                    if i == decode_seq_len - 1:
                        break

                    # 单步执行：输入 1 个 token，Cache 会自动进行驱逐/压缩
                    outputs = self._model(
                        input_ids=target_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,
                        return_dict=True
                    )
                    last_logit = outputs.logits

                results.append(total_logprob)

        # 评测结束后清理 Hook 句柄
        for h in self._hooks:
            h.remove()

        return results