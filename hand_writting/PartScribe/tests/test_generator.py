"""Unit and integration tests for stages 2-4.  Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from partscribe.charset import ALPHABET  # noqa: E402
from partscribe.content import ContentEncoder, build_content_table, char_layout, fit_widths  # noqa: E402
from partscribe.datasets import PNL_FIRST_WRITER, Batcher, PackedWords, concat_words, load_pnl  # noqa: E402
from partscribe.dit import DiT  # noqa: E402
from partscribe.generator import Config, PartScribe  # noqa: E402
from partscribe.holdout import held_chars  # noqa: E402
from partscribe.layers import Attention, grad_reverse  # noqa: E402
from partscribe.style import StyleEncoder, supcon_loss  # noqa: E402
from partscribe.vae import VAE, clean_right_edge, recon_losses  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CHARS = [c for c in ALPHABET if not c.isspace()]
SMALL = dict(dim=64, depth=2, heads=4, context_depth=1, n_global=2)


def fake_table(n_fonts=2, slots=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    valid = torch.rand(n_fonts, len(CHARS), slots, generator=g) > 0.4
    valid[..., 0] = True
    return {'chars': ''.join(CHARS), 'codes': torch.randint(240, (n_fonts, len(CHARS), slots), generator=g),
            'where': torch.randn(n_fonts, len(CHARS), slots, 5, generator=g), 'valid': valid, 'n_codes': 240}


def fake_batch(b=4, h=32, w=96, k=3, sw=64):
    torch.manual_seed(0)
    return {'z0': torch.randn(b, 4, h // 4, w // 4, device=DEV),
            'lat_w': torch.tensor([w // 4, w // 4 - 4, w // 4 - 8, 10][:b], device=DEV).float(),
            'texts': ['ngày', 'Yes', 'quê hương', 'jq'][:b],
            'support': torch.rand(b, k, h, sw, device=DEV), 'support_w': torch.full((b, k), sw, device=DEV),
            'writers': torch.tensor([0, 0, 1, 1][:b], device=DEV),
            'support_text': [['abc', 'xyz'], ['ab'], ['hello'], ['đi']][:b]}


class Layers(unittest.TestCase):
    def test_grad_reverse(self):
        x = torch.ones(3, requires_grad=True)
        grad_reverse(x, 0.5).sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.full((3,), -0.5)))

    def test_attention_key_mask(self):
        att = Attention(32, 4)
        q, ctx = torch.randn(2, 5, 32), torch.randn(2, 7, 32)
        mask = torch.tensor([[True] * 4 + [False] * 3] * 2)
        y1 = att(q, ctx, key_mask=mask)
        ctx2 = ctx.clone()
        ctx2[:, 4:] = 100
        self.assertTrue(torch.allclose(y1, att(q, ctx2, key_mask=mask), atol=1e-5))


class Content(unittest.TestCase):
    def test_modes_and_space(self):
        for mode in ('parts', 'nfd', 'char'):
            enc = ContentEncoder(fake_table(), mode, dim=32, depth=1, heads=4)
            out = enc(['ab c', 'ễ'])
            s = enc.n_slots
            self.assertEqual(out['tokens'].shape, (2, 4 * s, 32))
            self.assertEqual(out['char_mask'].tolist(), [[True] * 4, [True, False, False, False]])
            # space has exactly one token; padding has none
            self.assertEqual(out['token_mask'].reshape(2, 4, s)[0, 2].sum().item(), 1)
            self.assertEqual(out['token_mask'].reshape(2, 4, s)[1, 1:].sum().item(), 0)

    def test_parts_share_codes(self):
        """Two chars with identical slot tables get identical tokens (sharing is in the code vocabulary)."""
        t = fake_table()
        i, j = CHARS.index('e'), CHARS.index('ễ')
        for k in ('codes', 'where', 'valid'):
            t[k][:, j] = t[k][:, i]
        enc = ContentEncoder(t, 'parts', dim=32, depth=0, heads=4)
        out = enc(['e', 'ễ'])
        self.assertTrue(torch.allclose(out['tokens'][0], out['tokens'][1]))

    def test_nfd_factorisation(self):
        enc = ContentEncoder(None, 'nfd', dim=16, depth=0, heads=4)
        cid = lambda c: enc.index[c]
        self.assertEqual(enc.nfd_letter[cid('ễ')], enc.nfd_letter[cid('Ể')])     # same letter e
        self.assertEqual(enc.nfd_upper[cid('Ể')].item(), 1)
        self.assertEqual(enc.nfd_marks[cid('ễ')].tolist()[0], enc.nfd_marks[cid('ê')].tolist()[0])   # circumflex
        self.assertEqual(enc.nfd_marks[cid('ễ')].tolist()[1], enc.nfd_marks[cid('ẽ')].tolist()[0])   # tilde
        out = enc(['ễ', 'e'])
        self.assertEqual(out['token_mask'][0, :3].tolist(), [True, True, True])
        self.assertEqual(out['token_mask'][1, :3].tolist(), [True, False, False])
        # every factor of every held-out character occurs in some training character
        held = held_chars('vn')
        seen = [c for c in enc.table_chars if c not in held]
        for c in held:
            i = cid(c)
            self.assertIn(enc.nfd_letter[i].item(), {enc.nfd_letter[cid(x)].item() for x in seen})
            for m in enc.nfd_marks[i].tolist():
                if m >= 0:
                    self.assertTrue(any(m in enc.nfd_marks[cid(x)].tolist() for x in seen))

    def test_layout_gradient_reaches_widths(self):
        w = torch.tensor([[2.0, 3.0, 5.0]], requires_grad=True)
        c, ww = char_layout(fit_widths(w, torch.tensor([20.0])))
        self.assertAlmostEqual(float(ww.sum()), 20.0, places=4)
        (c[0, 1] * 1.0).backward()
        self.assertGreater(w.grad.abs().sum().item(), 0)


class Denoiser(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = Config(**SMALL)
        self.model = PartScribe(self.cfg, fake_table()).to(DEV)

    def _call(self, z, lat_w, b, drop_c=None, drop_s=None):
        m = self.model
        c, s, widths = m.condition(b['texts'], b['support'], b['support_w'])
        lay = char_layout(fit_widths(widths, lat_w))
        n = z.shape[0]
        f = torch.zeros(n, dtype=torch.bool, device=DEV)
        return m.dit(z, torch.full((n,), 0.5, device=DEV), lat_w, s['g'], c, lay, s['memory'], s['memory_mask'],
                     f if drop_c is None else drop_c, f if drop_s is None else drop_s)

    def test_zero_init_output(self):
        b = fake_batch()
        v = self._call(b['z0'], b['lat_w'], b)
        self.assertEqual(v.abs().max().item(), 0.0)

    def test_padding_invariance(self):
        """Extra right padding must not change the output on valid columns."""
        for p in self.model.dit.final.parameters():
            torch.nn.init.normal_(p, std=0.02)
        for blk in self.model.dit.blocks:
            torch.nn.init.normal_(blk.ada[-1].weight, std=0.02)
            torch.nn.init.normal_(blk.gate_c, std=0.5)
            torch.nn.init.normal_(blk.gate_s, std=0.5)
        b = fake_batch()
        z = b['z0']
        z_pad = torch.cat([z, torch.randn(*z.shape[:3], 8, device=DEV)], -1)
        v1, v2 = self._call(z, b['lat_w'], b), self._call(z_pad, b['lat_w'], b)
        for i, w in enumerate(b['lat_w'].long().tolist()):
            w = (w // 2) * 2
            self.assertTrue(torch.allclose(v1[i, ..., :w], v2[i, ..., :w], atol=1e-4))

    def test_dropout_removes_content(self):
        for blk in self.model.dit.blocks:
            torch.nn.init.normal_(blk.gate_c, std=0.5)
            torch.nn.init.normal_(blk.ada[-1].weight, std=0.02)
        torch.nn.init.normal_(self.model.dit.final.weight, std=0.02)
        b = fake_batch(2)
        b2 = dict(b, texts=['zzz', 'qq'])
        drop = torch.ones(2, dtype=torch.bool, device=DEV)
        v1 = self._call(b['z0'], b['lat_w'], b, drop_c=drop)
        v2 = self._call(b['z0'], b['lat_w'], b2, drop_c=drop)
        self.assertTrue(torch.allclose(v1, v2, atol=1e-5), 'dropped content must not reach the output')


class EndToEnd(unittest.TestCase):
    def test_overfit_and_sample(self):
        torch.manual_seed(0)
        for mode in ('parts', 'nfd', 'char'):
            model = PartScribe(Config(content=mode, **SMALL), fake_table()).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            b = fake_batch()
            first = last = None
            for step in range(60):
                torch.manual_seed(step % 4)       # few fixed noise/t draws -> learnable
                loss, parts = model.loss(b['z0'], b['lat_w'], b['texts'], b['support'], b['support_w'],
                                         b['writers'], b['support_text'])
                opt.zero_grad()
                loss.backward()
                opt.step()
                first = parts['flow'] if first is None else first
                last = parts['flow']
            self.assertLess(last, first * 0.8, f'{mode}: flow loss did not fall ({first:.3f} -> {last:.3f})')
            model.eval()
            z, lw = model.sample(['ab', 'ễỡ'], b['support'][:2], b['support_w'][:2], 32, steps=3)
            self.assertEqual(z.shape[:3], (2, 4, 8))
            self.assertTrue(torch.isfinite(z).all())
            self.assertEqual(z.shape[-1] % 2, 0)


class Style(unittest.TestCase):
    def test_memory_mask(self):
        """Cells beyond each support's width are masked; padding is always zero ink at a fixed width."""
        enc = StyleEncoder(64, 2, 4, n_chars=10).to(DEV).eval()
        x = torch.rand(1, 2, 32, 64, device=DEV)
        out = enc(x, torch.tensor([[20.0, 64.0]], device=DEV))
        m = out['memory_mask'].reshape(2, 4, 8)          # (K, rows, cols) at stride 8
        self.assertEqual(m[0].sum(0).tolist(), [4, 4, 4, 0, 0, 0, 0, 0])   # ceil(20/8) = 3 columns
        self.assertTrue(m[1].all())

    def test_supcon(self):
        g = torch.randn(6, 16)
        self.assertGreater(supcon_loss(g, torch.tensor([0, 0, 1, 1, 2, 2])).item(), 0)
        self.assertEqual(supcon_loss(g, torch.arange(6)).item(), 0)


class VAETest(unittest.TestCase):
    def test_shapes(self):
        vae = VAE().to(DEV)
        x = torch.rand(2, 1, 64, 120, device=DEV)
        rec, mu, logvar = vae(x)
        self.assertEqual(rec.shape, x.shape)
        self.assertEqual(mu.shape, (2, 4, 16, 30))
        L = recon_losses(rec, x, mu, logvar, torch.tensor([120, 60], device=DEV))
        self.assertTrue(all(torch.isfinite(v) for v in L.values()))


class BatcherTest(unittest.TestCase):
    def test_support_rules(self):
        texts = ['a', 'b', 'a', 'c', 'd', 'e', 'f', 'g']
        writers = [0, 0, 0, 0, 1, 1, 1, 1]
        ds = PackedWords(np.full((32, 80), 255, np.uint8), np.arange(8) * 10, [10] * 8, texts, writers,
                         list(range(8)), 'toy')
        bt = Batcher(ds, lambda t: True, writers_per_batch=2, per_writer=2, n_support=3)
        for _ in range(50):
            sup = bt.support_for(0)
            self.assertNotIn(0, sup)
            self.assertNotIn(2, sup)                 # same transcript 'a'
            self.assertTrue(all(writers[j] == 0 for j in sup))
        b = next(iter(bt))
        self.assertEqual(b['image'].shape[-1] % 8, 0)
        self.assertEqual(b['support'].shape, (4, 3, 32, 192))   # VATr-pp: 192 px at 32 px height


@unittest.skipUnless((ROOT / 'runs/smoke_slots/best.pt').exists() and (ROOT / 'data/glyphs_v1.npz').exists(),
                     'needs the stage-1 smoke checkpoint and glyph bank')
class RealTokenizer(unittest.TestCase):
    def test_content_table_from_tokenizer(self):
        from partscribe.fonts import load_glyph_bank
        from partscribe.tokenizer import PartTokenizer
        glyphs, chars, fonts = load_glyph_bank(ROOT / 'data/glyphs_v1.npz')
        st = torch.load(ROOT / 'runs/smoke_slots/best.pt', map_location='cpu', weights_only=False)
        net = PartTokenizer()
        net.load_state_dict(st['model'])
        train = [f['index'] for f in fonts if f['split'] == 'train'][:2]
        t = build_content_table(net, glyphs, chars, train, device=DEV)
        self.assertEqual(t['codes'].shape, (2, len(chars), 6))
        self.assertTrue(t['valid'].any(-1).all())
        enc = ContentEncoder(t, 'parts', dim=32, depth=1, heads=4)
        self.assertEqual(enc(['Nguyễn'])['tokens'].shape[1], 6 * 6)


VN_DATA = os.environ.get('PARTSCRIBE_VN_DATA')


@unittest.skipUnless(VN_DATA and Path(VN_DATA, 'train.h5').exists(), 'set PARTSCRIBE_VN_DATA to the h5 folder')
class RealVN(unittest.TestCase):
    def test_holdout_applied(self):
        from partscribe.runtime import VATR, VN_TRAIN_SHA
        from partscribe.datasets import load_vn
        man = VATR / 'files/vietnamese/manifest.json'
        ds = load_vn(man, VN_DATA, 'train', verify_sha=VN_TRAIN_SHA)
        held = held_chars('vn')
        self.assertFalse(any(held & set(t) for t in ds.texts))
        full = load_vn(man, VN_DATA, 'train', apply_holdout=False)
        self.assertEqual(len(full) - len(ds), 1360)      # protocol v1: 1,360 train images removed
        self.assertTrue(all(w < 295 for w in ds.writers))
        self.assertEqual(ds.height, 32)
        self.assertEqual(len(ds.pools), 265)                     # VATr-pp train writers
        val = load_vn(man, VN_DATA, 'val')
        self.assertFalse(set(val.pools) & set(ds.pools))          # 30 disjoint dev writers
        self.assertEqual(len(val.pools), 30)
        bt = Batcher(ds, lambda t: True, writers_per_batch=64, per_writer=1, n_support=1)
        self.assertEqual({k: len(v) for k, v in bt.by_source.items()}, {'VNOnDB': 95, 'ViHTGen': 170})
        from collections import Counter
        it = iter(bt)
        c = Counter(ds.source_of(int(ds.writers[i])) for _ in range(20) for i in next(it)['index'])
        self.assertLess(abs(c['VNOnDB'] / sum(c.values()) - 0.5), 0.08)   # equal source mass

    def test_multiword_and_mark_weight(self):
        from partscribe.runtime import VATR
        from partscribe.datasets import load_vn
        ds = load_vn(VATR / 'files/vietnamese/manifest.json', VN_DATA, 'train')
        bt = Batcher(ds, lambda t: True, multiword=1.0, mark_weight=3.0, n_support=4)
        b = next(iter(bt))
        joined = [k for k, i in enumerate(b['index']) if isinstance(i, tuple)]
        self.assertGreater(len(joined), 10)
        for k in joined:
            i = b['index'][k]
            self.assertEqual(len({int(ds.writers[m]) for m in i}), 1)                 # one writer
            self.assertEqual(b['text'][k], ' '.join(ds.texts[m] for m in i))
            self.assertFalse(set(b['support_text'][k]) & {ds.texts[m] for m in i})   # anti-copy
            self.assertLessEqual(b['width'][k].item(), bt.max_width)
        self.assertEqual(b['pix_weight'].shape, b['image'].shape)
        self.assertTrue(((b['pix_weight'] == 1) | (b['pix_weight'] == 4)).all())


if __name__ == '__main__':
    unittest.main()


class TestLongTailSampler(unittest.TestCase):
    def test_factors_match_vatr_convention(self):
        from partscribe.datasets import factors
        self.assertEqual(factors('Ễ'), ('e', 'circumflex', 'tilde', 'upper'))
        self.assertEqual(factors('đ'), ('d', 'crossbar', 'none', 'lower'))
        self.assertEqual(factors('7'), ('7', 'none', 'none', 'literal'))

    def test_rare_words_weighted_up_and_capped(self):
        from partscribe.datasets import longtail_weights
        w = longtail_weights(['và'] * 500 + ['NGUYÊN'])
        self.assertGreater(w[-1], 5 * w[0])
        self.assertLessEqual(w.max(), 20.)

    def test_vatr_parity_if_available(self):
        import numpy as np
        vatr = ROOT.parent / 'VATr-pp'
        if not (vatr / 'util/vietnamese_longtail.py').exists():
            self.skipTest('VATr-pp not found')
        sys.path.insert(1, str(vatr))
        from util.vietnamese_longtail import ComponentCorpusSampler
        from partscribe.datasets import longtail_weights
        texts = ['và', 'Hà', 'NGUYỄN', 'trường', 'Đại', 'học', 'và', 'ẩm', '12', 'ĐẠI']
        np.testing.assert_allclose(longtail_weights(texts), ComponentCorpusSampler(texts).weights)


def _pack(n, writer0, h=8, seed=0):
    rng = np.random.default_rng(seed)
    widths = rng.integers(5, 12, n)
    starts = np.concatenate([[0], np.cumsum(widths)[:-1]])
    pix = rng.integers(0, 255, (h, int(widths.sum())), dtype=np.uint8)
    return PackedWords(pix, starts, widths, [f'W{k}' for k in range(n)], np.array([writer0 + k % 3 for k in range(n)]),
                       list(range(n)), 'x')


class SupplementaryData(unittest.TestCase):
    def test_concat_keeps_every_image(self):
        a, b = _pack(7, 0), _pack(5, PNL_FIRST_WRITER, seed=1)
        c = concat_words(a, b)
        self.assertEqual(len(c), 12)
        for k in range(12):
            src, j = (a, k) if k < 7 else (b, k - 7)
            np.testing.assert_array_equal(c.image(k), src.image(j))
            self.assertEqual(c.texts[k], src.texts[j])

    def test_source_weights(self):
        a, b = _pack(60, 0), _pack(60, PNL_FIRST_WRITER, seed=1)
        a.source_of = b.source_of = lambda w: 'PNL' if w >= PNL_FIRST_WRITER else 'VN'
        c = concat_words(a, b)
        bt = Batcher(c, lambda t: True, writers_per_batch=8, per_writer=2, n_support=2, support_width=48,
                     width_multiple=1, source_weights={'PNL': 0.1}, seed=0)
        self.assertAlmostEqual(dict(zip(bt.sources, bt.source_p))['PNL'], 0.1)
        it, pnl, tot = iter(bt), 0, 0
        for _ in range(100):
            for ix in next(it)['index']:
                tot += 1
                pnl += (ix[0] if isinstance(ix, tuple) else ix) >= 60
        self.assertLess(abs(pnl / tot - 0.1), 0.04)

    @unittest.skipUnless(Path(__file__).resolve().parents[1].joinpath('data/pnl32.npz').exists(), 'data/pnl32.npz')
    def test_pnl_pack_is_clean(self):
        npz = Path(__file__).resolve().parents[1] / 'data/pnl32.npz'
        held = set(held_chars('vn')) | {c.upper() for c in held_chars('vn')}
        tr, te = load_pnl(npz, 'train'), load_pnl(npz, 'test')
        self.assertEqual(tr.height, 32)
        self.assertTrue(set(map(int, tr.writers)).isdisjoint(map(int, te.writers)))
        for d in (tr, te):
            self.assertTrue(all(held.isdisjoint(t) for t in d.texts))


class EdgeCleanup(unittest.TestCase):
    def test_removes_uniform_line_keeps_stroke(self):
        ink = torch.zeros(1, 1, 32, 20)
        ink[0, 0, :, 18:20] = 0.3          # decoder boundary line: uniform, full height
        ink[0, 0, 10:13, 16:20] = 1.0      # a stroke reaching the edge
        out = clean_right_edge(ink, torch.tensor([20]))
        self.assertLess(out[0, 0, 20:, 18:].max().item(), 1e-6)
        self.assertGreater(out[0, 0, 10:13, 16:20].min().item(), 0.69)
