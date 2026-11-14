import json
from pathlib import Path
import unittest

import torch
from torch import nn

from research.vietcompose.orthography import decompose_character,compose_character,component_errors,strip_tone,TONES
from research.vietcompose.model import ComponentQuery


class Glyph(nn.Module):
    def __init__(self,alphabet):
        super().__init__();self.linear=nn.Linear(16,32);self.register_buffer('symbols',torch.randn(len(alphabet)+1,16))
    def forward(self,q):return self.linear(self.symbols[q])


class CompositionTests(unittest.TestCase):
    def test_unicode_round_trip_and_roles(self):
        alphabet=json.loads(Path('files/vietnamese/charset.json').read_text())
        if isinstance(alphabet,dict):alphabet=alphabet.get('alphabet',alphabet.get('characters'))
        if isinstance(alphabet,list):alphabet=''.join(alphabet)
        for c in alphabet:self.assertEqual(compose_character(decompose_character(c)),c)
        for c,shape,tone in [('Ắ','breve','acute'),('Ậ','circumflex','dot_below'),('Ỡ','horn','tilde'),('Đ','crossbar','none')]:
            f=decompose_character(c);self.assertEqual((f.shape,f.tone),(shape,tone))
        self.assertEqual(decompose_character('ệ'),decompose_character('e\u0323\u0302'))
        self.assertEqual(strip_tone('DƯỜNG ĐẶNG'),'DƯƠNG ĐĂNG')
        with self.assertRaises(ValueError):decompose_character('Ç')

    def test_tone_error_not_base_error(self):
        r=component_errors('Ắ','Ằ');self.assertEqual(r['tone_errors'],1);self.assertEqual(r['base_errors'],0);self.assertEqual(r['shape_errors'],0)
        r=component_errors('Ắ','Á');self.assertEqual(r['shape_errors'],1);self.assertEqual(r['tone_errors'],0)
        r=component_errors('a','A');self.assertEqual(r['case_errors'],1);self.assertEqual(r['base_errors'],0)
        self.assertEqual(component_errors('a','')['character_errors'],1)

    def test_zero_residual_preserves_glyph_and_padding(self):
        alphabet='aAáÁăĂắẮeEêÊếẾ';glyph=Glyph(alphabet);adapter=ComponentQuery(glyph,alphabet,hidden=16).eval()
        q=torch.tensor([[1,2,3,0]]);memory=torch.randn(5,1,32)
        torch.testing.assert_close(adapter.compose(q,memory),glyph(q),atol=0,rtol=0)
        with torch.no_grad():adapter.up.weight.normal_(std=.02)
        result=adapter.compose(q,memory)
        torch.testing.assert_close(result[:,-1],glyph(q)[:,-1],atol=0,rtol=0)

    def test_shared_tone_embedding_and_live_gradient(self):
        alphabet='aAáÁeEêÊếẾ';adapter=ComponentQuery(Glyph(alphabet),alphabet,hidden=16).eval()
        with torch.no_grad():adapter.up.weight.normal_(std=.02)
        qa=alphabet.index('á')+1;qe=alphabet.index('ế')+1
        self.assertEqual(int(adapter.tone_ids[qa]),int(adapter.tone_ids[qe]))
        output=adapter.compose(torch.tensor([[qa,qe]]),torch.randn(3,1,32));output.square().mean().backward()
        self.assertGreater(float(adapter.tone.weight.grad[TONES.index('acute')].abs().sum()),0)
        self.assertTrue(torch.isfinite(adapter.up.weight.grad).all())

class ComponentCTCTests(unittest.TestCase):
    def test_none_is_nonblank_and_gradient_is_finite(self):
        from research.vietcompose.losses import ComponentCTC
        ctc=ComponentCTC('aáAÁ')
        self.assertNotEqual(int(ctc.map_tone[1]),0)
        self.assertEqual(int(ctc.map_tone[0]),0)
        logits=torch.randn(9,1,5,requires_grad=True)
        loss,roles=ctc(logits.log_softmax(-1),['aá'],torch.tensor([32]))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(set(roles),{'base','shape','tone','case'})

class LayeredRendererTests(unittest.TestCase):
    def test_body_invariant_to_marks_and_none_has_no_ink(self):
        from research.vietcompose.renderer import LayeredInkComposer
        renderer=LayeredInkComposer(style_dim=16,hidden=16).eval();style=torch.randn(1,16)
        plain=renderer(['A'],style);marked=renderer(['Ắ'],style)
        torch.testing.assert_close(plain['base_ink'],marked['base_ink'],atol=0,rtol=0)
        self.assertEqual(float(plain['modifier_ink'].sum()),0)
        self.assertEqual(float(plain['tone_ink'].sum()),0)
        self.assertGreater(float(marked['modifier_ink'].sum()),0)

    def test_dot_below_placement_and_gradient(self):
        from research.vietcompose.renderer import LayeredInkComposer
        renderer=LayeredInkComposer(style_dim=16,hidden=16).eval();style=torch.randn(1,16)
        result=renderer(['Ạ'],style);ink=result['tone_ink'];y=torch.arange(32).view(1,1,32,1)
        centroid=(ink*y).sum()/ink.sum();self.assertGreater(float(centroid),22)
        result['image'].square().mean().backward()
        self.assertTrue(torch.isfinite(renderer.tone_head[-1].weight.grad).all())
        self.assertGreater(float(renderer.tone_head[-1].weight.grad.abs().sum()),0)

    def test_padding_white_and_no_whole_character_lookup(self):
        from research.vietcompose.renderer import LayeredInkComposer
        renderer=LayeredInkComposer(style_dim=16,hidden=16).eval()
        result=renderer(['Ễ','DƯƠNG'],torch.randn(2,16))
        self.assertEqual(tuple(result['image'].shape),(2,1,32,80))
        torch.testing.assert_close(result['image'][0,:,:,16:],torch.ones(1,32,64))
