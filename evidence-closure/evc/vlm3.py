"""Qwen3-VL variant of evc.vlm.VLM (same interface: prepare / encode / scores / answer_batch).

Differences handled:
  - DeepStack: visual features from intermediate ViT layers are added to the LLM hidden states at
    visual positions in the first len(deepstack) decoder layers. When tokens are pruned, the
    deepstack features are pruned with exactly the same index set (row-major over the batch).
  - attention uses q_norm / k_norm before RoPE (RoPE cos/sin already interleaved -> 1-D rotary).
  - patch 16 px, merge 2 -> one LLM visual token per 32x32 px cell (Grid reads inp.grid).
"""
from __future__ import annotations

import math

import torch

from .vlm import LAYERS, PROMPT, StopForward, VLM


class VLM3(VLM):
    def __init__(self, path: str, device: str = "cuda:0"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        self.tok = self.proc.tokenizer
        cfg = self.model.config
        self.image_token_id = cfg.image_token_id
        self.eos = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
        self.core = self.model.model
        self.lm = self.core.language_model
        tc = cfg.text_config
        self.n_heads = tc.num_attention_heads
        self.n_kv = tc.num_key_value_heads
        self.head_dim = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)

    @torch.no_grad()
    def encode(self, inp):
        ids = inp["input_ids"]
        emb = self.lm.get_input_embeddings()(ids)
        feats, ds = self.core.get_image_features(inp["pixel_values"], inp["image_grid_thw"])
        feats = torch.cat(list(feats), 0)
        mask = ids == self.image_token_id
        assert int(mask.sum()) == feats.shape[0], (int(mask.sum()), feats.shape)
        emb = emb.clone()
        emb[mask] = feats.to(emb.dtype)
        pos, _ = self.core.get_rope_index(ids, inp["image_grid_thw"], None, attention_mask=inp["attention_mask"])
        self._ds = [d.to(emb.dtype) for d in ds]  # list[(N_img, D)] for the current prompt
        return emb, pos

    # ------------------------------------------------------------ relevance
    @torch.no_grad()
    def scores(self, inp, emb, pos):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        layers = self.lm.layers
        acc, hooks = [], []
        qpos, img_pos = inp.qpos, inp.img_pos

        def mk(li):
            def hook(mod, args, kwargs):
                h = kwargs.get("hidden_states", args[0] if args else None)
                cos, sin = kwargs["position_embeddings"]
                B, L, _ = h.shape
                q = mod.q_norm(mod.q_proj(h).view(B, L, -1, self.head_dim)).transpose(1, 2)
                k = mod.k_norm(mod.k_proj(h).view(B, L, -1, self.head_dim)).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
                logits = q[:, :, qpos].float() @ k.float().transpose(-1, -2) / math.sqrt(self.head_dim)
                causal = torch.arange(L, device=h.device)[None, :] > qpos[:, None]
                a = logits.masked_fill(causal[None, None], float("-inf")).softmax(-1)[..., img_pos]
                acc.append(a.mean((0, 1, 2)))
                if li == max(LAYERS):
                    raise StopForward
            return hook

        for li in LAYERS:
            hooks.append(layers[li].self_attn.register_forward_pre_hook(mk(li), with_kwargs=True))
        vmask = inp["input_ids"] == self.image_token_id
        try:
            # explicit all-ones mask: without it transformers infers "packed sequences" from the
            # temporal M-RoPE ids (constant over image tokens) and blocks text->image attention
            self.lm(inputs_embeds=emb, position_ids=pos, use_cache=False,
                    attention_mask=torch.ones_like(inp["input_ids"]),
                    visual_pos_masks=vmask, deepstack_visual_embeds=self._ds)
        except StopForward:
            pass
        finally:
            for hk in hooks:
                hk.remove()
        return torch.stack(acc).mean(0).cpu().numpy()

    # ------------------------------------------------------------ decoding
    @torch.no_grad()
    def answer_batch(self, emb, pos, inp, keeps, max_new: int = 48, bs: int = 12) -> list[str]:
        L = emb.shape[1]
        dev = emb.device
        n_img = len(inp.img_pos)
        sels, kimg = [], []
        for kp in keeps:
            if kp is None:
                ki = torch.arange(n_img, device=dev)
            else:
                ki = torch.as_tensor(sorted(set(kp)), device=dev, dtype=torch.long)
            keep = torch.ones(L, dtype=torch.bool, device=dev)
            keep[inp.img_pos] = False
            keep[inp.img_pos[ki]] = True
            sels.append(keep.nonzero().squeeze(1))
            kimg.append(ki)
        order = sorted(range(len(sels)), key=lambda i: len(sels[i]))
        outs = [None] * len(sels)
        embed = self.lm.get_input_embeddings()
        img_flag = torch.zeros(L, dtype=torch.bool, device=dev)
        img_flag[inp.img_pos] = True
        eos = torch.tensor(sorted(self.eos), device=dev)
        for c in range(0, len(order), bs):
            idx = order[c:c + bs]
            T = max(len(sels[i]) for i in idx)
            B = len(idx)
            E = emb.new_zeros(B, T, emb.shape[-1])
            P = pos.new_zeros(3, B, T)
            M = torch.zeros(B, T, dtype=torch.long, device=dev)
            V = torch.zeros(B, T, dtype=torch.bool, device=dev)
            for b, i in enumerate(idx):
                s = sels[i]
                E[b, T - len(s):] = emb[0, s]
                P[:, b, T - len(s):] = pos[:, 0, s]
                M[b, T - len(s):] = 1
                V[b, T - len(s):] = img_flag[s]
            # deepstack rows in row-major order of V (batch, then ascending position)
            ds = [torch.cat([d[kimg[i]] for i in idx], 0) for d in self._ds]
            out = self.lm(inputs_embeds=E, position_ids=P, attention_mask=M, use_cache=True,
                          visual_pos_masks=V, deepstack_visual_embeds=ds)
            pkv = out.past_key_values
            nxt = self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1)
            cur = int(pos.max()) + 1
            toks = [[] for _ in idx]
            done = torch.zeros(B, dtype=torch.bool, device=dev)
            for _ in range(max_new):
                done |= torch.isin(nxt, eos)
                if bool(done.all()):
                    break
                for b in range(B):
                    if not done[b]:
                        toks[b].append(int(nxt[b]))
                M = torch.cat([M, torch.ones(B, 1, dtype=M.dtype, device=dev)], 1)
                p1 = torch.full((3, B, 1), cur, device=dev, dtype=pos.dtype)
                out = self.lm(inputs_embeds=embed(nxt[:, None]), position_ids=p1, attention_mask=M,
                              past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                nxt = self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1)
                cur += 1
            for b, i in enumerate(idx):
                outs[i] = self.tok.decode(toks[b], skip_special_tokens=True).strip()
        return outs
