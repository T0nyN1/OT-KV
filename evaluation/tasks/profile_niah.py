from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from .registry import register_task


@register_task("profile_niah")
class ProfileNIAHEvaluator(BaseEvaluator):

    def evaluate(self) -> Dict[str, Any]:
        import torch
        import time
        import os
        import glob
        from transformers import LogitsProcessorList, LogitsProcessor

        print("\n[*] Running System Profiler with NIAH Corpus...")

        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model

        prompt_length = self.args.get('max_length', 4000)
        generate_length = self.args.get('profiler_gen_length', 128)
        haystack_dir = self.args.get('haystack_dir', "./datasets/PaulGrahamEssays")

        text_files = glob.glob(os.path.join(haystack_dir, "*.txt"))
        if not text_files:
            print(f"[!] Error: No .txt files found in {haystack_dir}.")
            return {"profile_niah": {}}

        full_text = ""
        for f in text_files:
            with open(f, 'r', encoding='utf-8') as file:
                full_text += file.read() + "\n\n"

        full_text_tokens = tokenizer.encode(full_text, add_special_tokens=False)

        while len(full_text_tokens) < prompt_length:
            full_text_tokens += full_text_tokens

        context_tokens = full_text_tokens[:prompt_length]
        real_prompt = tokenizer.decode(context_tokens)
        inputs = tokenizer(real_prompt, return_tensors="pt").to(model.device)

        print(f"-> Target Prefill Length: {inputs.input_ids.shape[1]} tokens")
        print(f"-> Target Generate Length: {generate_length} tokens")

        print("-> Performing CUDA Warm-up...")
        with torch.no_grad():
            _ = model.generate(
                inputs.input_ids[:, :128],
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        class TTFTTracker(LogitsProcessor):
            def __init__(self):
                self.start_time = None
                self.ttft = None

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                if self.ttft is None and self.start_time is not None:
                    torch.cuda.synchronize()
                    self.ttft = time.time() - self.start_time
                return scores

        ttft_tracker = TTFTTracker()
        logits_processor = LogitsProcessorList([ttft_tracker])

        print("-> Running Benchmark...")

        custom_cache = self.model_wrapper._setup_cache_and_hooks()

        torch.cuda.synchronize()
        start_time = time.time()
        ttft_tracker.start_time = start_time

        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=generate_length,
                min_new_tokens=generate_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=custom_cache,
                use_cache=True,
                logits_processor=logits_processor
            )
        self._cleanup_cache_and_hooks(custom_cache)

        torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        ttft = ttft_tracker.ttft if ttft_tracker.ttft else total_time
        decode_time = total_time - ttft
        decode_tokens = max(generate_length - 1, 1)

        decode_tps = decode_tokens / decode_time if decode_time > 0 else 0
        overall_tps = generate_length / total_time
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        print(f"\n--- Profiling Results ---")
        print(f"-> Context Length: {inputs.input_ids.shape[1]} tokens")
        print(f"-> Total Time: {total_time:.4f} s")
        print(f"-> Time To First Token (TTFT): {ttft:.4f} s")
        print(f"-> Pure Decode Time ({decode_tokens} tokens): {decode_time:.4f} s")
        print(f"-> Pure Decode Throughput: {decode_tps:.2f} tokens/s")
        print(f"-> Overall Throughput: {overall_tps:.2f} tokens/s")
        print(f"-> Peak VRAM Usage: {peak_memory_mb:.2f} MB")
        print("---------------------------------------")

        return {
            "profile_niah": {
                "context_length_tokens": inputs.input_ids.shape[1],
                "total_time_s": total_time,
                "ttft_s": ttft,
                "pure_decode_time_s": decode_time,
                "pure_decode_throughput_tps": decode_tps,
                "overall_throughput_tps": overall_tps,
                "peak_memory_mb": peak_memory_mb,
            }
        }
