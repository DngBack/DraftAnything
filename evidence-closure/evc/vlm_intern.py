"""Minimal InternVL (transformers-native -hf checkpoint) wrapper for prompt-level experiments.

Only what scripts/run_decide.py needs: model, proc, tok, device, eos, dtype. No token pruning or attention scoring.
"""
from __future__ import annotations

import torch


class VLMI:
    dtype = torch.bfloat16

    def __init__(self, path: str, device: str = "cuda:0"):
        from transformers import AutoProcessor, InternVLForConditionalGeneration

        self.device = device
        self.model = InternVLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        # the checkpoint default squeezes the whole page into one 448 px tile; tile it, as InternVL's own usage does
        self.proc.image_processor.crop_to_patches = True
        self.tok = self.proc.tokenizer
        self.eos = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
