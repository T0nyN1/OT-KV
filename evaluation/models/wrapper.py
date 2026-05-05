# evaluation/models/wrapper.py
import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm
from transformers.cache_utils import DynamicCache


def _get_attention_pre_hook():
    """
    [新增] 在进入 self_attn 之前拦截并处理 kwargs
    消除 Decode 阶段由于 KV Cache 压缩导致的 attention_mask 形状不匹配问题。
    """

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
                    # 1. 让具体的 Cache 策略决定如何降维 attention 矩阵
                    #    （SnapKV 会在此处先做 observation-window 切片，再求和）。
                    accumulated_score = cache_obj.reduce_attention(attn_weights)
                    cache_obj.current_attention_scores[layer_idx] = accumulated_score

                # 2. 【核心！显存物理粉碎】
                # 在 Hook 内部直接剥夺这个 4GB 矩阵的底层显存，
                # 这样哪怕 HF 外部还有循环在引用它，拿到的也只是一个 0 字节的空壳！
                attn_weights.untyped_storage().resize_(0)

                # 3. 替换并切断链条
                new_outputs = list(outputs)
                new_outputs[1] = None
                return tuple(new_outputs)

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

        # 3. 仅当策略需要 attention 分数时，才挂载 attention hook
        #    （StreamingLLM / EchoKV / Baseline 等无需，省下大量显存与拷贝开销）
        needs_attn = getattr(cache_instance, "requires_attention", True)

        # 4. 遍历模型所有层，挂载必要的 Hook
        # 这里的路径 'model.layers' 适用于 Llama/Mistral，其他模型可能需微调
        layers = self._model.model.layers
        for layer_idx, layer in enumerate(layers):
            if needs_attn:
                hook_handle = layer.self_attn.register_forward_hook(
                    _get_attention_hook(cache_instance, layer_idx)
                )
                self._hooks.append(hook_handle)

            # pre-hook 与 attention 是否需要无关，仅修复 decode 阶段 mask 形状问题
            pre_hook_handle = layer.self_attn.register_forward_pre_hook(
                _get_attention_pre_hook(), with_kwargs=True
            )
            self._hooks.append(pre_hook_handle)

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

                print(f"\n" + "=" * 55)
                print(f"[Doc Monitor] Processing New Document")
                print(f"[Doc Monitor] Total Tokens   : {len(token_ids)}")
                print(f"[Doc Monitor] Prefill Tokens : {prefix_ids.shape[1]}")
                print(f"[Doc Monitor] Decode Steps   : {target_ids.shape[1]}")
                print("=" * 55)

                past_key_values = self._setup_cache_and_hooks()
                needs_attn = getattr(past_key_values, "requires_attention", False)

                # --- [监控] Prefill 阶段 ---
                q_len_prefill = prefix_ids.shape[1]

                # 1. 执行 Prefill
                # 在 forward 过程中，HF 模型会自动调用 past_key_values.update(...)
                outputs = self._model(
                    input_ids=prefix_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_attentions=needs_attn,  # 仅在策略需要时输出 attention，节省显存
                    return_dict=True
                )
                past_key_values.on_prefill_end()

                # 获取压缩后的长度 (以第0层为例)
                post_prefill_len = self._get_phys_length(past_key_values, 0)
                budget = getattr(past_key_values, "budget", "N/A")
                sink = getattr(past_key_values, "sink_size", "N/A")
                recent = getattr(past_key_values, "recent_size", "N/A")
                middle = getattr(past_key_values, "middle_budget", "N/A")

                print(
                    f"\n[KV Monitor] Stage: Prefill | "
                    f"Input Tokens: {q_len_prefill:<4} | "
                    f"Cache: {post_prefill_len:<5} | "
                    f"Budget: {budget:<4} (Sink:{sink} Middle:{middle} Recent:{recent})")

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
                        position_ids=torch.tensor([[split_idx + i]], device=device),
                        output_attentions=needs_attn,
                        return_dict=True
                    )
                    post_decode_len = self._get_phys_length(past_key_values, 0)

                    budget = getattr(past_key_values, "budget", "N/A")
                    sink = getattr(past_key_values, "sink_size", "N/A")
                    recent = getattr(past_key_values, "recent_size", "N/A")
                    middle = getattr(past_key_values, "middle_budget", "N/A")

                    last_logit = outputs.logits
                    # 动态刷新显示长度变化
                    log_str = (f"[KV Monitor] Stage: Decode  | Step: {i:<4} | "
                               f"Cache: {post_decode_len:<4} | "
                               f"Budget: {budget:<4} (Sink:{sink} Middle:{middle} Recent:{recent})")
                    print(f"\r{log_str}\033[K", end="", flush=True)

                print()
                results.append(total_logprob)

        # 评测结束后清理 Hook 句柄
        for h in self._hooks:
            h.remove()

        return results

    def _get_phys_length(self, cache_obj, layer_idx=0):
        if hasattr(cache_obj, "layers"):
            if layer_idx < len(cache_obj.layers):
                k_tensor = cache_obj.layers[layer_idx].keys
                if k_tensor is not None and k_tensor.numel() > 0:
                    return k_tensor.shape[-2]

        elif hasattr(cache_obj, "key_cache"):
            k_cache = cache_obj.key_cache
            if layer_idx < len(k_cache) and k_cache[layer_idx] is not None:
                if k_cache[layer_idx].numel() > 0:
                    return k_cache[layer_idx].shape[-2]

        return cache_obj.get_seq_length(layer_idx)