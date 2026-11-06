"""Qwen2.5-VL wrapper: post-merger visual-token pruning with original M-RoPE positions.

Pipeline per question:
  prepare()  -> processor inputs (full-resolution page)
  encode()   -> inputs_embeds with image features spliced in, 3-D rope positions
  scores()   -> question->image attention per merged visual token (layers LAYERS)
  answer()   -> greedy decode keeping all text tokens + a chosen subset of image tokens
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

PROMPT = ("Answer the question using the document image. Give only the final answer, no explanation. "
          "If the answer is a number, write the number followed by its scale or unit if one applies "
          "(thousand, million, billion, or percent). If there are several answers, separate them with '; '.\n"
          "Question: {q}")
LAYERS = (16, 20, 24)  # chosen on dev by fact recall, same for all models (deviation #12)


class StopForward(Exception):
    pass


class VLM:
    def __init__(self, path: str, device: str = "cuda:0", max_pixels: int = 2400 * 28 * 28 * 2):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        self.tok = self.proc.tokenizer
        cfg = self.model.config
        self.image_token_id = cfg.image_token_id
        self.eos = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
        self.core = self.model.model
        self.lm = self.core.language_model
        tc = getattr(cfg, "text_config", cfg)
        self.n_heads = tc.num_attention_heads
        self.n_kv = tc.num_key_value_heads
        self.head_dim = tc.hidden_size // tc.num_attention_heads

    # ------------------------------------------------------------ inputs
    @torch.no_grad()
    def prepare(self, image, question: str):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": PROMPT.format(q=question)}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[image], return_tensors="pt").to(self.device)
        t, h, w = inp["image_grid_thw"][0].tolist()
        m = self.core.visual.spatial_merge_size if hasattr(self.core.visual, "spatial_merge_size") else 2
        inp.grid = (h // m, w // m)
        # positions of the question text tokens (after the image) for attention scoring
        ids = inp["input_ids"][0]
        img_pos = (ids == self.image_token_id).nonzero().squeeze(1)
        inp.img_pos = img_pos
        inp.qpos = self._question_span(ids, int(img_pos[-1]))
        return inp

    def _question_span(self, ids, last_img):
        """Token positions of the question text: after 'Question: ' up to <|im_end|>.

        (Bug fixed 2026-11-05: matching tokenizer(question) as a subsequence always failed because
        of the leading-space token in context, silently falling back to *all* text after the image,
        i.e. mostly the generic instruction. See docs/deviations.md #12.)
        """
        L = ids.numel()
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        tail = ids[last_img + 1:].tolist()
        start = end = None
        acc = ""
        for k, t in enumerate(tail):
            acc += self.tok.decode([t])
            if start is None and acc.rstrip().endswith("Question:"):
                start = last_img + 1 + k + 1
            if start is not None and t == im_end:
                end = last_img + 1 + k
                break
        if start is None or end is None or end <= start:
            raise RuntimeError("question span not found")
        return torch.arange(start, end, device=ids.device)

    @torch.no_grad()
    def encode(self, inp):
        ids = inp["input_ids"]
        emb = self.lm.get_input_embeddings()(ids)
        feats = self.core.get_image_features(inp["pixel_values"], inp["image_grid_thw"])
        if not torch.is_tensor(feats):
            feats = getattr(feats, "pooler_output", feats)
            feats = torch.cat(list(feats), 0) if not torch.is_tensor(feats) else feats
        mask = ids == self.image_token_id
        assert int(mask.sum()) == feats.shape[0], (int(mask.sum()), feats.shape)
        emb = emb.clone()
        emb[mask] = feats.to(emb.dtype)
        pos, _ = self.core.get_rope_index(ids, inp["image_grid_thw"], None, attention_mask=inp["attention_mask"])
        return emb, pos  # (1,L,D), (3,1,L)

    # ------------------------------------------------------------ relevance
    @torch.no_grad()
    def scores(self, inp, emb, pos):
        """Mean attention (heads x question tokens x LAYERS) from question tokens to each image token."""
        layers = self.lm.layers
        acc = []
        qpos = inp.qpos
        img_pos = inp.img_pos
        hooks = []

        def mk(li):
            def hook(mod, args, kwargs):
                h = kwargs.get("hidden_states", args[0] if args else None)
                cos, sin = kwargs["position_embeddings"]
                B, L, _ = h.shape
                q = mod.q_proj(h).view(B, L, -1, self.head_dim).transpose(1, 2)
                k = mod.k_proj(h).view(B, L, -1, self.head_dim).transpose(1, 2)
                q, k = self._rope(q, k, cos, sin, mod)
                rep = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(rep, 1)
                qs = q[:, :, qpos].float()
                logits = qs @ k.float().transpose(-1, -2) / math.sqrt(self.head_dim)
                causal = torch.arange(L, device=h.device)[None, :] > qpos[:, None]
                logits = logits.masked_fill(causal[None, None], float("-inf"))
                a = logits.softmax(-1)[..., img_pos]  # (B,H,Q,Nimg)
                acc.append(a.mean((0, 1, 2)))
                if li == max(LAYERS):
                    raise StopForward
            return hook

        for li in LAYERS:
            hooks.append(layers[li].self_attn.register_forward_pre_hook(mk(li), with_kwargs=True))
        try:
            self.lm(inputs_embeds=emb, position_ids=pos, use_cache=False,
                    attention_mask=torch.ones_like(inp["input_ids"]))
        except StopForward:
            pass
        finally:
            for hk in hooks:
                hk.remove()
        return torch.stack(acc).mean(0).cpu().numpy()

    def _rope(self, q, k, cos, sin, mod):
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as M

        sec = self.model.config.rope_scaling.get("mrope_section") if getattr(self.model.config, "rope_scaling", None) else None
        if sec is None:
            sec = getattr(self.model.config, "text_config", self.model.config).rope_scaling["mrope_section"]
        if cos.dim() == 4:  # (3,B,L,D) -> multimodal variant
            return M.apply_multimodal_rotary_pos_emb(q, k, cos, sin, sec)
        return M.apply_rotary_pos_emb(q, k, cos, sin)

    # ------------------------------------------------------------ group relevance
    REL_PROMPT = ("Only part of this document page is visible. Question: {q}\n"
                  "Does the visible part contain information needed to answer the question? Answer Yes or No.")

    @torch.no_grad()
    def prepare_rel(self, image, question: str):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": self.REL_PROMPT.format(q=question)}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[image], return_tensors="pt").to(self.device)
        ids = inp["input_ids"][0]
        inp.img_pos = (ids == self.image_token_id).nonzero().squeeze(1)
        return inp

    @torch.no_grad()
    def yes_scores(self, emb, pos, inp, keeps, bs: int = 32):
        """logit(Yes) - logit(No) at the answer position, one prefill per kept-token subset."""
        if not hasattr(self, "_yn"):
            def tid(s):
                return self.tok(s, add_special_tokens=False)["input_ids"][0]
            self._yn = ([tid("Yes"), tid("yes")], [tid("No"), tid("no")])
        L = emb.shape[1]
        dev = emb.device
        out = []
        for c in range(0, len(keeps), bs):
            chunk = keeps[c:c + bs]
            sels = []
            for kp in chunk:
                keep = torch.ones(L, dtype=torch.bool, device=dev)
                keep[inp.img_pos] = False
                keep[inp.img_pos[torch.as_tensor(sorted(set(kp)), device=dev, dtype=torch.long)]] = True
                sels.append(keep.nonzero().squeeze(1))
            T = max(len(s) for s in sels)
            B = len(sels)
            E = emb.new_zeros(B, T, emb.shape[-1])
            P = pos.new_zeros(3, B, T)
            M = torch.zeros(B, T, dtype=torch.long, device=dev)
            for b, s in enumerate(sels):
                E[b, T - len(s):] = emb[0, s]
                P[:, b, T - len(s):] = pos[:, 0, s]
                M[b, T - len(s):] = 1
            h = self.lm(inputs_embeds=E, position_ids=P, attention_mask=M, use_cache=False).last_hidden_state[:, -1]
            lg = self.model.lm_head(h).float()
            y, n = self._yn
            out += (torch.logsumexp(lg[:, y], -1) - torch.logsumexp(lg[:, n], -1)).tolist()
        return out

    # ------------------------------------------------------------ decoding
    @torch.no_grad()
    def answer_batch(self, emb, pos, inp, keeps, max_new: int = 48, bs: int = 12) -> list[str]:
        """Greedy decode many token subsets of the same prompt. keeps[i] None = full input.

        Left-padded batch; padded slots are masked and get position 0. All sequences share the
        same text suffix, so the next rope position is pos.max()+1 for every row.
        """
        L = emb.shape[1]
        dev = emb.device
        sels = []
        for kp in keeps:
            if kp is None:
                sels.append(torch.arange(L, device=dev))
                continue
            keep = torch.ones(L, dtype=torch.bool, device=dev)
            keep[inp.img_pos] = False
            keep[inp.img_pos[torch.as_tensor(sorted(set(kp)), device=dev, dtype=torch.long)]] = True
            sels.append(keep.nonzero().squeeze(1))
        order = sorted(range(len(sels)), key=lambda i: len(sels[i]))
        outs = [None] * len(sels)
        embed = self.lm.get_input_embeddings()
        for c in range(0, len(order), bs):
            idx = order[c:c + bs]
            T = max(len(sels[i]) for i in idx)
            B = len(idx)
            E = emb.new_zeros(B, T, emb.shape[-1])
            P = pos.new_zeros(3, B, T)
            M = torch.zeros(B, T, dtype=torch.long, device=dev)
            for b, i in enumerate(idx):
                s = sels[i]
                E[b, T - len(s):] = emb[0, s]
                P[:, b, T - len(s):] = pos[:, 0, s]
                M[b, T - len(s):] = 1
            out = self.lm(inputs_embeds=E, position_ids=P, attention_mask=M, use_cache=True)
            pkv = out.past_key_values
            nxt = self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1)
            cur = int(pos.max()) + 1
            toks = [[] for _ in idx]
            done = torch.zeros(B, dtype=torch.bool, device=dev)
            eos = torch.tensor(sorted(self.eos), device=dev)
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

    @torch.no_grad()
    def answer(self, emb, pos, inp, keep_img=None, max_new: int = 48) -> str:
        L = emb.shape[1]
        if keep_img is None:
            sel = torch.arange(L, device=emb.device)
        else:
            keep = torch.ones(L, dtype=torch.bool, device=emb.device)
            keep[inp.img_pos] = False
            kept = inp.img_pos[torch.as_tensor(sorted(keep_img), device=emb.device, dtype=torch.long)]
            keep[kept] = True
            sel = keep.nonzero().squeeze(1)
        e, p = emb[:, sel], pos[:, :, sel]
        out = self.lm(inputs_embeds=e, position_ids=p, use_cache=True)
        pkv = out.past_key_values
        nxt = int(self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1))
        cur = int(pos.max()) + 1
        toks = []
        for _ in range(max_new):
            if nxt in self.eos:
                break
            toks.append(nxt)
            e1 = self.lm.get_input_embeddings()(torch.tensor([[nxt]], device=emb.device))
            p1 = torch.full((3, 1, 1), cur, device=emb.device, dtype=pos.dtype)
            out = self.lm(inputs_embeds=e1, position_ids=p1, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            nxt = int(self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1))
            cur += 1
        return self.tok.decode(toks, skip_special_tokens=True).strip()


def load_vlm(path: str, device: str = "cuda:0"):
    """Pick the wrapper from the checkpoint family (Qwen3-VL needs DeepStack-aware pruning)."""
    if "internvl" in str(path).lower():
        from .vlm_intern import VLMI

        return VLMI(path, device=device)
    if "qwen3" in str(path).lower():
        from .vlm3 import VLM3

        return VLM3(path, device=device)
    return VLM(path, device=device)
