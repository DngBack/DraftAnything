"""VATr++ Vietnamese adaptation: original four-phase GAN schedule and optional OCR warmup."""
import argparse
import hashlib
import json
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from data.vietnamese import PackedH5, CorpusSampler, collate, load_manifest
from models.model import Generator
from models.OCR_network import CRNN, strLabelConverter
from models.BigGAN_networks import Discriminator, WDiscriminator


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_args(manifest, device, query_position=True):
    return SimpleNamespace(alphabet=manifest['alphabet'], special_alphabet='', device=device,
                           glyph_path=manifest['glyph_path'], query_input='unifont',
                           vocab_size=len(manifest['alphabet']), resolution=16, add_noise=False,
                           imagenet_weights=False, correct_style_layout=True, query_position=query_position)


def ocr_loss(network, converter, images, texts, widths):
    targets, lengths, _ = converter.encode(texts)
    predictions = network(images)
    steps = widths.cpu().long() // 4 + 1
    required = torch.tensor([len(t) + sum(a == b for a, b in zip(t, t[1:])) for t in texts])
    if torch.any(steps < required) or torch.any(steps > predictions.shape[0]):
        raise ValueError('Invalid CTC geometry; audit must match OCR architecture')
    return F.ctc_loss(predictions, targets.to(images.device), steps, lengths, blank=0,
                      reduction='mean', zero_infinity=False)


def decoded(predictions, widths, alphabet):
    paths = predictions.argmax(-1).cpu()
    result = []
    for b, width in enumerate(widths):
        previous, chars = -1, []
        for token in paths[:int(width)//4+1, b].tolist():
            if token and token != previous:
                chars.append(alphabet[token-1])
            previous = token
        result.append(''.join(chars))
    return result


def edit_distance(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        next_row = [i]
        for j, y in enumerate(b, 1):
            next_row.append(min(row[j]+1, next_row[-1]+1, row[j-1]+(x != y)))
        row = next_row
    return row[-1]


def unaccent(text):
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))


@torch.no_grad()
def evaluate_ocr(network, dataset, alphabet, batch_size, device, max_batches=0):
    was_training = network.training
    network.eval()
    errors = chars = plain_errors = plain_chars = word_errors = count = 0
    examples = []
    for k, batch in enumerate(DataLoader(dataset, batch_size=batch_size, collate_fn=collate)):
        if max_batches and k >= max_batches:
            break
        preds = decoded(network(batch['img'].to(device)), batch['widths'], alphabet)
        for truth, pred in zip(batch['label'], preds):
            errors += edit_distance(truth, pred); chars += len(truth)
            plain_errors += edit_distance(unaccent(truth), unaccent(pred)); plain_chars += len(unaccent(truth))
            word_errors += truth != pred; count += 1
            if len(examples) < 24:
                examples.append({'reference': truth, 'prediction': pred})
    network.train(was_training)
    return dict(samples=count, cer=errors/max(chars, 1), cer_without_diacritics=plain_errors/max(plain_chars, 1),
                transcript_error_rate=word_errors/max(count, 1), examples=examples,
                evaluation='internal OCR validation, not independent generator quality')


def infinite(loader):
    while True:
        yield from loader


def trainable(network, enabled):
    network.train(enabled)
    for p in network.parameters():
        p.requires_grad_(enabled)


def state_digest(network):
    """Include parameters and buffers so frozen BatchNorm state is checked too."""
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def step_optimizer(loss, optimizer, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite loss: {loss.item()}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
    optimizer.step()
    return float(grad_norm.detach())


def partial_load(module, path, prefix):
    if Path(path).suffix == '.safetensors':
        from safetensors.torch import load_file
        raw = load_file(str(path), device='cpu')
    else:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        raw = checkpoint.get('model', checkpoint)
    raw = {k.removeprefix('model.'): v for k, v in raw.items()}
    if prefix == 'netG.' and 'G' in raw:
        raw = raw['G']
    elif any(k.startswith(prefix) for k in raw):
        raw = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
    own = module.state_dict()
    matched = {k: v for k, v in raw.items() if k in own and hasattr(v, 'shape') and v.shape == own[k].shape}
    if not matched:
        raise ValueError(f'No matching parameters in {path} for {prefix}')
    missing, unexpected = module.load_state_dict(matched, strict=False)
    if prefix == 'netG.' and missing:
        raise ValueError(f'Generator checkpoint incomplete: {list(missing)}')
    return {'loaded': len(matched), 'missing': list(missing), 'unexpected': list(unexpected)}


def save_checkpoint(path, networks, optimizers, step, args, manifest, fingerprint):
    state = dict(version=1, stage=args.stage, step=step, config=vars(args), alphabet=manifest['alphabet'],
                 manifest_sha256=fingerprint, model={k: v.state_dict() for k, v in networks.items()},
                 optimizers={k: v.state_dict() for k, v in optimizers.items()},
                 rng={'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                      'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
    temporary = path.with_suffix('.partial')
    torch.save(state, temporary)
    temporary.replace(path)


def override_resume_lrs(optimizers, args):
    """Explicit fine-tuning override after restoring Adam moments and step counters."""
    mult = getattr(args, 'glyph_encoder_lr_mult', 1.)
    g_rates = [args.encoder_lr, args.lr] + ([args.lr*mult] if mult != 1. else [])
    rates = {'G': g_rates, 'D': [args.d_lr],
             'W': [args.writer_lr if args.writer_lr is not None else args.lr], 'R': [args.lr]}
    for key, optimizer in optimizers.items():
        for group, rate in zip(optimizer.param_groups, rates[key], strict=True):
            group['lr'] = rate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--glyph-ablation', type=Path, help='Controlled glyph pickle override; data manifest stays unchanged')
    p.add_argument('--glyph-canvas', type=int, choices=[128], help='Nearest-repeat glyphs onto common capacity grid')
    p.add_argument('--reset-glyph-projection', action='store_true', help='Fresh seeded projection after strict shared checkpoint load')
    p.add_argument('--glyph-encoder', choices=['linear', 'cnn'], default='linear', help='Content glyph encoder (plan 261117_plan_03)')
    p.add_argument('--glyph-init', type=Path, help='Aligned glyph encoder from research/glyph_resolution/align.py')
    p.add_argument('--glyph-encoder-lr-mult', type=float, default=1., help='LR multiplier for the glyph encoder; 1 keeps one shared G group')
    p.add_argument('--uppercase-benchmark', action='store_true', help='Uppercase proper-name panel on all development writers (diagnostic only)')
    p.add_argument('--tf32', action='store_true', help='Allow TF32 matmul/conv on Ampere+ GPUs (speed; changes float32 numerics)')
    p.add_argument('--trace-training-inputs', action='store_true', help='Hash training supports and record batch indices/text for matched pilots')
    p.add_argument('--manifest', type=Path, default=Path('files/vietnamese/manifest.json'))
    p.add_argument('--stage', choices=['ocr', 'gan'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20000, help='Total steps including resumed steps')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-examples', type=int, default=15)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--gpu-memory-gb', type=float, default=0., help='Cap this process CUDA allocator; 0 means no explicit cap')
    p.add_argument('--benchmark-every', type=int, default=1000, help='Fixed held-out writer panel interval for GAN; 0 disables')
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--d-lr', type=float, default=1e-5)
    p.add_argument('--writer-lr', type=float, help='Separate writer LR; defaults to --lr')
    p.add_argument('--override-resume-lr', action='store_true', help='Apply CLI rates after loading optimizer state')
    p.add_argument('--keep-checkpoints', type=int, default=0, help='Retain this many numbered checkpoints in addition to last/best')
    p.add_argument('--expanded-benchmark', action='store_true', help='Evaluate 20 prompts on all development writers')
    p.add_argument('--best-window', type=int, default=3, help='Select best proxy checkpoint using this many evaluations')
    p.add_argument('--hat-content-chars', default='Ễ', help='Rare characters for the auxiliary single-image HAT task')
    p.add_argument('--hat-content-weight', type=float, default=0., help='Opt-in HAT self-reconstruction loss; not writer supervision')
    p.add_argument('--hat-content-every', type=int, default=4)
    p.add_argument('--hat-content-batch-size', type=int, default=2)
    p.add_argument('--paired-feature-weight', type=float, default=0.)
    p.add_argument('--paired-pixel-weight', type=float, default=0.)
    p.add_argument('--paired-every', type=int, default=4)
    p.add_argument('--visual-benchmark', action='store_true', help='Fixed source-D visual proxy on 60 genuine held-out writer targets')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--eval-batches', type=int, default=0, help='0 evaluates all development samples')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--seed', type=int, default=742)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--ocr-checkpoint', type=Path)
    p.add_argument('--generator-checkpoint', type=Path)
    p.add_argument('--encoder-checkpoint', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--finetune-from', type=Path, help='Load full native weights, reset Adam/RNG/step, explicitly permit an audited child manifest')
    p.add_argument('--freeze-writer', action='store_true', help='Skip writer losses and optimizer; keep two G and two D updates')
    p.add_argument('--include-hat-real', action='store_true', help='Use audited gan_train images; requires frozen writer')
    p.add_argument('--hat-support', choices=['real-writer', 'self-repeat'], default='real-writer')
    p.add_argument('--hat-real-fraction', type=float, default=.35)
    p.add_argument('--verify-frozen-state', action='store_true', help='Hash states before/after training and audit first-step updates')
    p.add_argument('--allow-untrained-ocr', action='store_true', help='Smoke tests only; unsuitable for quality training')
    p.add_argument('--legacy-generator-layout', action='store_true', default=True, help='Keep original layout and no positional queries for English-checkpoint ablation')
    p.add_argument('--adapt-generator-layout', dest='legacy_generator_layout', action='store_false', help='Experimental: correct style tensor ordering and add query position encoding; changes original architecture behavior')
    p.add_argument('--freeze-ocr', action='store_true', help='Ablation: freeze OCR during GAN rather than continue real-image OCR training')
    p.add_argument('--text-sampler', choices=['legacy', 'component'], default='legacy', help='Opt-in case/component interaction weighting within rare transcript draws')
    p.add_argument('--rare-fraction', type=float, default=.3)
    p.add_argument('--hat-fraction', type=float, default=.35, help='Fraction of generated transcript draws from HAT; remaining from real writers')
    p.add_argument('--no-hat', action='store_true', help='Exclude HAT from OCR and generated transcripts for a writer-identity ablation')
    p.add_argument('--ocr-weight', type=float, default=1.)
    p.add_argument('--writer-weight', type=float, default=1.)
    p.add_argument('--gradient-balance', action='store_true', default=True)
    p.add_argument('--no-gradient-balance', dest='gradient_balance', action='store_false')
    p.add_argument('--crop-discriminator', action='store_true', default=True)
    p.add_argument('--no-crop-discriminator', dest='crop_discriminator', action='store_false')
    args = p.parse_args()
    if (args.reset_glyph_projection or args.glyph_init) and not (args.finetune_from and args.glyph_ablation and args.glyph_canvas):
        p.error('reset-glyph-projection/glyph-init require finetune-from, glyph-ablation and glyph-canvas')
    if args.glyph_ablation and not (args.stage == 'gan' and args.glyph_canvas and (args.reset_glyph_projection or args.glyph_init or args.resume)):
        p.error('glyph-ablation requires GAN with common canvas and reset, glyph-init or resume')
    if args.glyph_encoder != 'linear' and not args.glyph_ablation:
        p.error('glyph-encoder cnn requires glyph-ablation (a new encoder cannot load native weights)')
    if args.glyph_encoder_lr_mult <= 0:
        p.error('glyph-encoder-lr-mult must be positive')
    if args.uppercase_benchmark and not (args.stage == 'gan' and args.benchmark_every):
        p.error('uppercase-benchmark requires GAN with a nonzero benchmark interval')
    if args.glyph_canvas and not args.glyph_ablation:
        p.error('glyph-canvas requires glyph-ablation')
    if args.finetune_from and (args.resume or args.generator_checkpoint or args.encoder_checkpoint or args.ocr_checkpoint):
        p.error('finetune-from cannot be combined with other checkpoint inputs')
    if args.finetune_from and args.stage != 'gan':
        p.error('finetune-from requires GAN stage')
    if args.include_hat_real and (args.stage != 'gan' or not args.freeze_writer or args.no_hat):
        p.error('HAT real images require GAN, frozen writer and HAT enabled')
    if not 0 < args.hat_real_fraction < 1:
        p.error('hat-real-fraction must lie in (0, 1)')
    if args.freeze_writer:
        args.writer_weight = 0.
    if args.keep_checkpoints < 0 or args.best_window < 1:
        p.error('keep-checkpoints must be nonnegative and best-window positive')
    if args.hat_content_weight < 0 or args.hat_content_every < 1 or args.hat_content_batch_size < 1:
        p.error('Invalid HAT content auxiliary configuration')
    if args.hat_content_weight and (args.stage != 'gan' or args.no_hat):
        p.error('HAT content auxiliary requires GAN with HAT enabled')
    if min(args.paired_feature_weight, args.paired_pixel_weight) < 0 or args.paired_every < 1:
        p.error('Invalid paired visual supervision configuration')
    if (args.visual_benchmark or args.paired_feature_weight or args.paired_pixel_weight) and args.stage != 'gan':
        p.error('Visual supervision/evaluation requires GAN')
    if args.visual_benchmark and not args.benchmark_every:
        p.error('visual-benchmark requires a nonzero benchmark interval')
    if args.no_hat:
        args.hat_fraction = 0.
    if min(args.save_every, args.eval_every, args.log_every) < 1:
        p.error('save-every, eval-every and log-every must be positive')
    if args.steps < 1 or args.batch_size < 1 or args.num_examples < 1:
        p.error('steps, batch-size and num-examples must be positive')
    if not 0 <= args.hat_fraction <= 1 or not 0 <= args.rare_fraction <= 1:
        p.error('sampling fractions must lie in [0, 1]')
    if args.stage == 'gan' and not (args.ocr_checkpoint or args.resume or args.finetune_from or args.allow_untrained_ocr):
        p.error('GAN requires --ocr-checkpoint from Vietnamese OCR warmup')
    if args.generator_checkpoint and args.encoder_checkpoint:
        p.error('Choose generator-checkpoint or encoder-checkpoint')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'last.pth').exists() and not args.resume:
        raise FileExistsError('Output already has a checkpoint; choose a new directory or use --resume')
    torch.set_num_threads(4)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.gpu_memory_gb and args.device.startswith('cuda'):
        allocation_device = torch.device(args.device)
        allocation_index = allocation_device.index if allocation_device.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(allocation_index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.gpu_memory_gb * 1024**3 / total, 1.), device=allocation_index)
    seed_all(args.seed)
    manifest = load_manifest(args.manifest)
    fingerprint = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    cfg = model_args(manifest, args.device, not args.legacy_generator_layout)
    if args.glyph_ablation:
        cfg.glyph_path = str(args.glyph_ablation.resolve())
        cfg.glyph_canvas = args.glyph_canvas
        cfg.glyph_encoder = args.glyph_encoder
        args.glyph_sha256 = hashlib.sha256(args.glyph_ablation.read_bytes()).hexdigest()
    args.glyph_init_sha256 = hashlib.sha256(args.glyph_init.read_bytes()).hexdigest() if args.glyph_init else None
    cfg.correct_style_layout = not args.legacy_generator_layout
    converter = strLabelConverter(manifest['alphabet'])
    ocr = CRNN(cfg).to(args.device)
    networks = {'R': ocr}
    initialization = {}
    if args.ocr_checkpoint:
        checkpoint = torch.load(args.ocr_checkpoint, map_location='cpu', weights_only=False)
        if checkpoint.get('alphabet') != manifest['alphabet'] or checkpoint.get('manifest_sha256') != fingerprint:
            raise ValueError('OCR checkpoint charset/split does not match manifest')
        if args.no_hat and not checkpoint.get('config', {}).get('no_hat', False):
            raise ValueError('--no-hat requires an OCR checkpoint also trained with --no-hat')
        ocr.load_state_dict(checkpoint['model']['R'])
        initialization['ocr'] = {'path': str(args.ocr_checkpoint), 'step': checkpoint['step']}
    dataset = PackedH5(manifest, include_hat=args.stage == 'ocr' and not args.no_hat, style=args.stage == 'gan', num_examples=args.num_examples,
                       include_hat_real=args.include_hat_real, hat_support=args.hat_support, hat_real_fraction=args.hat_real_fraction)
    validation = PackedH5(manifest, split='val')
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=dataset.sampler(), num_workers=args.workers,
                        collate_fn=collate, drop_last=True, pin_memory=args.device.startswith('cuda'))
    if len(loader) == 0:
        raise ValueError('Batch size exceeds dataset size')
    if args.stage == 'ocr':
        optimizers = {'R': torch.optim.Adam(ocr.parameters(), lr=args.lr)}
    else:
        generator = Generator(cfg).to(args.device)
        discriminator = Discriminator(resolution=16, crop_size=[64, 128] if args.crop_discriminator else None).to(args.device)
        writer = WDiscriminator(resolution=16, n_classes=len(manifest['alphabet']), output_dim=len(dataset.writer_map)).to(args.device)
        networks.update(G=generator, D=discriminator, W=writer)
        if args.generator_checkpoint:
            initialization['generator'] = partial_load(generator, args.generator_checkpoint, 'netG.')
        if args.encoder_checkpoint:
            state = torch.load(args.encoder_checkpoint, map_location='cpu', weights_only=False)
            state = state.get('model', state)
            if state['conv1.weight'].shape[1] == 3:
                state['conv1.weight'] = state['conv1.weight'].mean(1, keepdim=True)
            state = {k: v for k, v in state.items() if not k.startswith('fc.')}
            generator.Feat_Encoder.load_state_dict(state, strict=True)
            initialization['encoder'] = str(args.encoder_checkpoint)
        encoder_params = list(generator.Feat_Encoder.parameters())
        separate_glyph = args.glyph_encoder_lr_mult != 1.
        glyph_params = [v for k, v in generator.named_parameters() if k.startswith('query_embed.')]
        other_params = [v for k, v in generator.named_parameters() if not k.startswith('Feat_Encoder.')
                        and not (separate_glyph and k.startswith('query_embed.'))]
        g_groups = [{'params': encoder_params, 'lr': args.encoder_lr}, {'params': other_params, 'lr': args.lr}]
        if separate_glyph:
            g_groups.append({'params': glyph_params, 'lr': args.lr*args.glyph_encoder_lr_mult})
        optimizers = {'G': torch.optim.Adam(g_groups, betas=(0., .999)),
                      'D': torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0., .999)),
                      'W': torch.optim.Adam(writer.parameters(), lr=args.writer_lr if args.writer_lr is not None else args.lr, betas=(0., .999))}
        if args.freeze_writer:
            del optimizers['W']
            trainable(writer, False)
        if not args.freeze_ocr:
            optimizers['R'] = torch.optim.Adam(ocr.parameters(), lr=args.lr)
        trainable(ocr, False)
        corpus_class = CorpusSampler
        if args.text_sampler == 'component':
            from util.vietnamese_longtail import ComponentCorpusSampler
            corpus_class = ComponentCorpusSampler
        real_corpus = corpus_class([dataset.labels[i] for i in manifest['indices']['train']], rare_fraction=args.rare_fraction)
        hat_corpus = corpus_class([dataset.labels[i] for i in manifest['indices']['hat_ocr']], rare_fraction=args.rare_fraction)
        if args.text_sampler == 'component':
            (args.output/'text_sampling_exposure.json').write_text(json.dumps({
                'real_conditional_probability': real_corpus.exposure(),
                'hat_conditional_probability': hat_corpus.exposure(),
                'hat_fraction': args.hat_fraction, 'component_mix_within_rare_draw': .5,
                'note': 'Analytic probability a transcript draw contains the character; not image supervision or a quality claim.'
            }, ensure_ascii=False, indent=2))
    start = 0
    if args.finetune_from:
        state = torch.load(args.finetune_from, map_location='cpu', weights_only=False)
        if state['stage'] != 'gan' or state['alphabet'] != manifest['alphabet']:
            raise ValueError('Fine-tune checkpoint stage/charset mismatch')
        accepted = (fingerprint, manifest.get('parent_manifest_sha256'))
        if state['manifest_sha256'] not in accepted:
            raise ValueError('Fine-tune requires same manifest or explicitly audited parent')
        for key in ('num_examples', 'legacy_generator_layout'):
            if state['config'][key] != getattr(args, key):
                raise ValueError('Fine-tune architecture mismatch: '+key)
        for key, net in networks.items():
            if key == 'G' and (args.reset_glyph_projection or args.glyph_init):
                from research.glyph_resolution.initialization import load_shared_generator
                aligned = torch.load(args.glyph_init, map_location='cpu', weights_only=False) if args.glyph_init else None
                if aligned is not None:
                    record = aligned['record']
                    if record['encoder'] != args.glyph_encoder or record['glyph_sha256'] != args.glyph_sha256 or record['canvas'] != args.glyph_canvas:
                        raise ValueError('glyph-init encoder/glyph/canvas does not match this run')
                    if record['source_step'] != state['step']:
                        raise ValueError('glyph-init was aligned to a different source checkpoint')
                initialization['glyph_projection'] = load_shared_generator(
                    net, state['model'][key], args.seed, glyph_init=aligned['state_dict'] if aligned else None)
                if aligned is not None:
                    initialization['glyph_projection']['alignment'] = {k: aligned['record'][k] for k in
                        ('metrics', 'passed', 'gate_exception', 'design_rank', 'encoder_sha256', 'loss')}
            else:
                net.load_state_dict(state['model'][key], strict=True)
        with args.finetune_from.open('rb') as source_handle:
            parent_sha256 = hashlib.file_digest(source_handle, 'sha256').hexdigest()
        initialization['finetune'] = {'path': str(args.finetune_from.resolve()), 'sha256': parent_sha256,
                                      'parent_step': state['step'], 'parent_manifest_sha256': state['manifest_sha256'],
                                      'policy': 'Strict shared G/D/W/R; glyph encoder reset or aligned init when requested; fresh Adam, CLI seed and local step zero'}
        del state
        # Weight-loading and model construction must not define the pilot RNG.
        seed_all(args.seed)
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        initialization.update(state['config'].get('initialization', {}))
        initialization['resume'] = {'path': str(args.resume.resolve()), 'step': state['step']}
        if state['stage'] != args.stage or state['manifest_sha256'] != fingerprint:
            raise ValueError('Resume stage/data manifest mismatch')
        for key in ('num_examples', 'legacy_generator_layout', 'gradient_balance', 'crop_discriminator', 'no_hat', 'freeze_ocr'):
            if state['config'][key] != getattr(args, key):
                raise ValueError(f'Resume configuration mismatch: {key}')
        for key, default in [('freeze_writer', False), ('include_hat_real', False), ('hat_support', 'real-writer'), ('hat_real_fraction', .35)]:
            if state['config'].get(key, default) != getattr(args, key):
                raise ValueError('Resume configuration mismatch: '+key)
        for key in ('glyph_canvas', 'glyph_sha256'):
            if state['config'].get(key) != getattr(args, key, None):
                raise ValueError('Resume glyph mismatch: '+key)
        for key, default in [('glyph_encoder', 'linear'), ('glyph_encoder_lr_mult', 1.)]:
            if state['config'].get(key, default) != getattr(args, key):
                raise ValueError('Resume glyph mismatch: '+key)
        for k, net in networks.items():
            net.load_state_dict(state['model'][k])
        for k, opt in optimizers.items():
            opt.load_state_dict(state['optimizers'][k])
        if args.override_resume_lr:
            override_resume_lrs(optimizers, args)
        start = state['step']
        random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if torch.cuda.is_available() and state['rng']['cuda']:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
    # If no persistent benchmark OCR exists, use the loaded supervisor, never
    # the randomly initialized network from before resume.
    initial_ocr_state = {k: v.detach().cpu().clone() for k, v in ocr.state_dict().items()} if args.stage == 'gan' else None
    if args.glyph_ablation:
        from research.glyph_resolution.initialization import projection_stats
        initialization['glyph_statistics'] = projection_stats(generator)
        initialization['network_parameters'] = {k: {'total': sum(p.numel() for p in net.parameters()),
                                                       'trainable_initial': sum(p.numel() for p in net.parameters() if p.requires_grad)}
                                                for k, net in networks.items()}
    args.initialization = initialization
    (args.output/'config.json').write_text(json.dumps({**vars(args), 'manifest_sha256': fingerprint}, default=str, ensure_ascii=False, indent=2), encoding='utf-8')
    rare_content = None
    if args.hat_content_weight:
        from data.vietnamese_rare_content import RareHatContent, rare_reconstruction_loss
        rare_content = RareHatContent(manifest, args.hat_content_chars, args.seed)
        (args.output/'hat_content_split.json').write_text(json.dumps({'chars': args.hat_content_chars, 'train_indices': rare_content.train_indices, 'eval_indices': rare_content.eval_indices, 'warning': 'Single-image self-reconstruction. Eval held out from this auxiliary task only, not from prior OCR/corpus training. HAT IDs never supervise the writer head.'}, ensure_ascii=False, indent=2))
    benchmark = None
    expanded = None
    uppercase = None
    visual = None
    if args.stage == 'gan' and args.benchmark_every:
        from util.vietnamese_benchmark import FixedBenchmark
        benchmark = FixedBenchmark(manifest, args.num_examples, args.output, args.device)
        benchmark_ocr = CRNN(cfg).to(args.device)
        reference_path = args.output/'benchmark_ocr.pth'
        if reference_path.exists():
            initial_ocr_state = torch.load(reference_path, map_location='cpu', weights_only=True)
        else:
            torch.save(initial_ocr_state, reference_path)
        benchmark_ocr.load_state_dict(initial_ocr_state)
        benchmark_ocr.eval()
        for parameter in benchmark_ocr.parameters():
            parameter.requires_grad_(False)
        benchmark.render(generator, benchmark_ocr, converter, start)
        if args.expanded_benchmark:
            expanded = FixedBenchmark(manifest, args.num_examples, args.output/'expanded', args.device, expanded=True)
            expanded.render(generator, benchmark_ocr, converter, start)
        if args.uppercase_benchmark:
            from util.vietnamese_uppercase import UppercaseBenchmark
            uppercase = UppercaseBenchmark(manifest, args.num_examples, args.output/'uppercase', args.device)
            uppercase.render(generator, benchmark_ocr, converter, start)
        if rare_content is not None:
            from data.vietnamese_rare_content import evaluate_rare_reconstruction
            evaluate_rare_reconstruction(rare_content, generator, benchmark_ocr, converter, args.output/'rare_content_eval', start, args.num_examples, args.device)
        if args.visual_benchmark:
            import copy
            from util.vietnamese_visual import VisualBenchmark
            fixed_discriminator = copy.deepcopy(discriminator)
            visual_reference = args.output/'visual_discriminator.pth'
            if visual_reference.exists():
                fixed_discriminator.load_state_dict(torch.load(visual_reference, map_location='cpu', weights_only=True))
            else:
                torch.save({k:v.detach().cpu() for k,v in fixed_discriminator.state_dict().items()}, visual_reference)
            visual = VisualBenchmark(manifest, args.num_examples, args.output/'visual', args.device, fixed_discriminator)
            visual_initial = visual.evaluate(generator, benchmark_ocr, converter, start)
            visual_initial_cer = visual_initial['guidance_ocr_cer']
            visual_selection_path = args.output/'best_visual_selection.json'
            best_visual_score = json.loads(visual_selection_path.read_text())['feature_distance'] if visual_selection_path.exists() else float('inf')
    score_dir = args.output/'expanded' if args.expanded_benchmark else args.output
    history = []
    for path in sorted(score_dir.glob('benchmark_*.json')):
        if path.stem.removeprefix('benchmark_').isdigit():
            record = json.loads(path.read_text())
            if record['step'] <= start:
                history.append(record['guidance_ocr_cer'])
    selection_path = args.output/'best_selection.json'
    best_score = json.loads(selection_path.read_text())['rolling_cer'] if selection_path.exists() else float('inf')
    focus_path = args.output/'best_focus_selection.json'
    best_focus = json.loads(focus_path.read_text())['focus_cer'] if focus_path.exists() else float('inf')
    batches = infinite(loader)
    prior_evaluations = [json.loads(path.read_text())['cer'] for path in args.output.glob('validation_*.json')]
    best_cer = min(prior_evaluations, default=float('inf'))
    begun = time.monotonic()
    verification = None
    if args.verify_frozen_state and args.stage == 'gan':
        verification = {'before': {k: state_digest(v) for k, v in networks.items()},
                        'optimizer_keys': sorted(optimizers), 'first_step_updates': {'G': 0, 'D': 0, 'W': 0, 'R': 0}}
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for step in range(start+1, args.steps+1):
        if args.glyph_ablation:
            from research.glyph_resolution.initialization import seed_stream
            seed_all(seed_stream(args.seed, step, 'batch'))
        data_started = time.monotonic()
        batch = next(batches)
        images = batch['img'].to(args.device)
        data_seconds = time.monotonic()-data_started
        if args.stage == 'ocr':
            loss = ocr_loss(ocr, converter, images, batch['label'], batch['widths'])
            step_optimizer(loss, optimizers['R'], ocr.parameters())
            metrics = {'ocr_ctc': float(loss.detach())}
        else:
            style = batch['style'].to(args.device)
            writer_ids = batch['writer'].to(args.device)
            # Match train.py: G+OCR -> D+OCR -> G+writer -> D+writer.
            # Regenerate text/fakes per phase, as VATr++ does in self.forward().
            metrics = {}
            generator.train()
            phase_texts = {}
            for phase in ('G_OCR', 'D_OCR', 'G_W', 'D_W'):
                if args.glyph_ablation:
                    seed_all(seed_stream(args.seed, step, phase))
                texts = [(hat_corpus if random.random() < args.hat_fraction else real_corpus).sample(1)[0] for _ in batch['label']]
                phase_texts[phase] = texts
                query, lengths, _ = converter.encode(texts)
                query = query.to(args.device)
                is_g = phase.startswith('G')
                trainable(discriminator, not is_g)
                trainable(writer, phase == 'D_W' and not args.freeze_writer)
                trainable(ocr, phase == 'D_OCR' and not args.freeze_ocr)
                if is_g:
                    fake = generator.generate(generator.compute_style(style), query)
                else:
                    with torch.no_grad():
                        fake = generator.generate(generator.compute_style(style), query)
                mask = torch.arange(fake.shape[-1], device=args.device)[None, None, None, :] < (lengths.to(args.device)*16)[:, None, None, None]
                fake = torch.where(mask, fake, torch.ones_like(fake))
                if is_g:
                    adversarial = -discriminator(fake).mean()
                    auxiliary = (ocr_loss(ocr, converter, fake, texts, lengths*16) if phase == 'G_OCR'
                                 else None if args.freeze_writer else writer(fake, writer_ids))
                    weight = torch.tensor(args.ocr_weight if phase == 'G_OCR' else args.writer_weight, device=args.device)
                    if args.gradient_balance and auxiliary is not None:
                        g_adv = torch.autograd.grad(adversarial, fake, retain_graph=True)[0]
                        g_aux = torch.autograd.grad(auxiliary, fake, retain_graph=True)[0]
                        weight = weight * (.7*g_adv.std()/(g_aux.std()+1e-7)).detach().clamp(1e-4, 1000.)
                        metrics[phase+'_image_gradient_adv_std'] = float(g_adv.detach().std())
                        metrics[phase+'_image_gradient_aux_std'] = float(g_aux.detach().std())
                    loss = adversarial if auxiliary is None else adversarial + weight*auxiliary
                    grad_norm = step_optimizer(loss, optimizers['G'], generator.parameters())
                    metrics[phase+'_adversarial'] = float(adversarial.detach())
                    metrics[phase+'_effective_weight'] = float(weight.detach())
                    metrics[phase+'_grad_norm_before_clip'] = grad_norm
                    metrics[phase] = float(loss.detach())
                    if auxiliary is not None:
                        metrics['ocr_fake' if phase == 'G_OCR' else 'writer_fake'] = float(auxiliary.detach())
                    if verification is not None and step == start+1:
                        verification['first_step_updates']['G'] += 1
                        if phase == 'G_OCR':
                            verification['ocr_loss_has_gradient'] = bool(auxiliary.requires_grad)
                else:
                    discriminator.update_parameters(step)
                    real_scores, fake_scores = discriminator(images), discriminator(fake)
                    loss_d = F.relu(1-real_scores).mean() + F.relu(1+fake_scores).mean()
                    metrics[phase+'_real_score'] = float(real_scores.detach().mean())
                    metrics[phase+'_fake_score'] = float(fake_scores.detach().mean())
                    step_optimizer(loss_d, optimizers['D'], discriminator.parameters())
                    if verification is not None and step == start+1:
                        verification['first_step_updates']['D'] += 1
                    metrics[phase] = float(loss_d.detach())
                    if phase == 'D_OCR' and not args.freeze_ocr:
                        real_ocr = ocr_loss(ocr, converter, images, batch['label'], batch['widths'])
                        step_optimizer(real_ocr, optimizers['R'], ocr.parameters())
                        metrics['ocr_real'] = float(real_ocr.detach())
                        if verification is not None and step == start+1:
                            verification['first_step_updates']['R'] += 1
                    if phase == 'D_W' and not args.freeze_writer:
                        loss_w = writer(images, writer_ids)
                        step_optimizer(loss_w, optimizers['W'], writer.parameters())
                        metrics['writer_real'] = float(loss_w.detach())
                        if verification is not None and step == start+1:
                            verification['first_step_updates']['W'] += 1
            if verification is not None and step == start+1:
                verification['after_first_step'] = {k: state_digest(v) for k, v in networks.items()}
                verification['writer_gradient_absent'] = all(p.grad is None for p in writer.parameters())
                verification['ocr_gradient_absent'] = all(p.grad is None for p in ocr.parameters())
                if args.freeze_writer:
                    assert verification['before']['W'] == verification['after_first_step']['W']
                    assert verification['writer_gradient_absent']
                if args.freeze_ocr:
                    assert verification['before']['R'] == verification['after_first_step']['R']
                assert all(verification['before'][k] != verification['after_first_step'][k] for k in ('G', 'D'))
                (args.output/'update_verification.json').write_text(json.dumps(verification, indent=2))
        if args.stage == 'gan' and (args.paired_feature_weight or args.paired_pixel_weight) and step % args.paired_every == 0:
            from util.vietnamese_visual import feature_matching_loss, paired_pixel_loss
            trainable(discriminator, False); trainable(writer, False); trainable(ocr, False)
            paired_query, paired_lengths, _ = converter.encode(batch['label'])
            paired_fake = generator.generate(generator.compute_style(style), paired_query.to(args.device))
            paired_widths = batch['widths'].to(args.device)
            paired_mask = torch.arange(paired_fake.shape[-1], device=args.device)[None, None, None, :] < paired_widths[:, None, None, None]
            paired_fake = torch.where(paired_mask, paired_fake, torch.ones_like(paired_fake))
            feature_loss = feature_matching_loss(discriminator, paired_fake, images, paired_widths) if args.paired_feature_weight else paired_fake.sum()*0
            pixel_loss = paired_pixel_loss(paired_fake, images, paired_widths) if args.paired_pixel_weight else paired_fake.sum()*0
            paired_norm = step_optimizer(args.paired_feature_weight*feature_loss + args.paired_pixel_weight*pixel_loss, optimizers['G'], generator.parameters())
            metrics['paired_feature_loss'] = float(feature_loss.detach())
            metrics['paired_pixel_loss'] = float(pixel_loss.detach())
            metrics['paired_grad_norm_before_clip'] = paired_norm
        if rare_content is not None and step % args.hat_content_every == 0:
            trainable(discriminator, False); trainable(writer, False); trainable(ocr, False)
            auxiliary_batch = rare_content.batch(args.hat_content_batch_size, args.num_examples, args.device)
            auxiliary_query, _, _ = converter.encode(auxiliary_batch['texts'])
            auxiliary_fake = generator.generate(generator.compute_style(auxiliary_batch['style']), auxiliary_query.to(args.device))
            auxiliary_loss = rare_reconstruction_loss(auxiliary_fake, auxiliary_batch['target'], auxiliary_batch['widths'])
            auxiliary_norm = step_optimizer(args.hat_content_weight*auxiliary_loss, optimizers['G'], generator.parameters())
            if step % args.log_every == 0:
                metrics['hat_content_reconstruction'] = float(auxiliary_loss.detach())
                metrics['hat_content_grad_norm_before_clip'] = auxiliary_norm
                metrics['hat_content_reference_mode'] = 'target image as single self-reference; not writer/style transfer'
        if args.trace_training_inputs and args.stage == 'gan':
            trace = {'step': step, 'indices': batch['indices'], 'texts': texts, 'phase_texts': phase_texts,
                     'support_sha256': hashlib.sha256(style.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            with (args.output/'training_inputs.jsonl').open('a') as f:
                f.write(json.dumps(trace, ensure_ascii=False)+'\n')
        metrics.update(step=step, seconds=time.monotonic()-begun, data_seconds=data_seconds)
        if args.stage == 'gan':
            metrics['hat_real_in_batch'] = sum(int(dataset.writers[i]) >= 295 for i in batch['indices'])
            metrics['writer_frozen'] = args.freeze_writer
        if step == start+1 or step % args.log_every == 0:
            if args.stage == 'gan':
                metrics['effective_learning_rates'] = {k: [g['lr'] for g in opt.param_groups] for k, opt in optimizers.items()}
            print(json.dumps(metrics), flush=True)
            with (args.output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(metrics)+'\n')
        if step % args.eval_every == 0 or step == args.steps:
            if args.stage == 'ocr':
                result = evaluate_ocr(ocr, validation, manifest['alphabet'], args.batch_size, args.device, args.eval_batches)
                (args.output/f'validation_{step:06d}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps({k: v for k, v in result.items() if k != 'examples'}), flush=True)
                if result['cer'] < best_cer:
                    best_cer = result['cer']
                    save_checkpoint(args.output/'best.pth', networks, optimizers, step, args, manifest, fingerprint)
            else:
                generator.eval()
                with torch.no_grad():
                    render = generator.generate(generator.compute_style(style[:1]), query[:1, :int(lengths[0])])
                array = ((render[0, 0].clamp(-1, 1).cpu().numpy()+1)*127.5).astype('uint8')
                Image.fromarray(array).save(args.output/f'preview_{step:06d}.png')
                (args.output/f'preview_{step:06d}.json').write_text(json.dumps({'text': texts[0], 'support_target_index': batch['indices'][0]}, ensure_ascii=False), encoding='utf-8')
        if benchmark is not None and (step % args.benchmark_every == 0 or step == args.steps):
            result = benchmark.render(generator, benchmark_ocr, converter, step)
            if expanded is not None:
                result = expanded.render(generator, benchmark_ocr, converter, step)
                focus_prompts = [p for p in result['per_prompt'] if any(c in p for c in args.hat_content_chars)]
                if focus_prompts:
                    focus_cer = sum(result['per_prompt'][p]['cer'] for p in focus_prompts)/len(focus_prompts)
                    if focus_cer < best_focus:
                        save_checkpoint(args.output/'best_focus.pth', networks, optimizers, step, args, manifest, fingerprint)
                        best_focus = focus_cer
                        focus_path.write_text(json.dumps({'step': step, 'focus_cer': focus_cer, 'prompts': focus_prompts, 'overall_cer': result['guidance_ocr_cer'], 'warning': 'Supervisor OCR proxy on genuine development writer references, not independent OCR.'}, ensure_ascii=False, indent=2))
            if uppercase is not None:
                uppercase.render(generator, benchmark_ocr, converter, step)
            if rare_content is not None:
                evaluate_rare_reconstruction(rare_content, generator, benchmark_ocr, converter, args.output/'rare_content_eval', step, args.num_examples, args.device)
            if visual is not None:
                visual_result = visual.evaluate(generator, benchmark_ocr, converter, step)
                if visual_result['mean_feature_distance'] < best_visual_score and visual_result['guidance_ocr_cer'] <= visual_initial_cer+.02:
                    save_checkpoint(args.output/'best_visual.pth', networks, optimizers, step, args, manifest, fingerprint)
                    best_visual_score = visual_result['mean_feature_distance']
                    visual_selection_path.write_text(json.dumps({'step': step, 'feature_distance': best_visual_score, 'pixel_l1': visual_result['mean_pixel_l1'], 'cer': visual_result['guidance_ocr_cer'], 'initial_cer': visual_initial_cer, 'warning': 'Fixed discriminator proxy; OCR degradation gate is +2 percentage points. Human sample review required before promotion.'}, indent=2))
            history.append(result['guidance_ocr_cer'])
            if len(history) >= args.best_window:
                score = sum(history[-args.best_window:])/args.best_window
                if score < best_score:
                    save_checkpoint(args.output/'best.pth', networks, optimizers, step, args, manifest, fingerprint)
                    best_score = score
                    selection_path.write_text(json.dumps({'step': step, 'rolling_cer': score, 'current_cer': result['guidance_ocr_cer'], 'window': args.best_window, 'metric': 'expanded supervisor OCR proxy' if expanded else 'small supervisor OCR proxy', 'warning': 'Rolling window ranks training progress; best.pth contains only the current model, not a weight average.'}, indent=2))
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.output/'last.pth', networks, optimizers, step, args, manifest, fingerprint)
            if args.keep_checkpoints:
                import os
                numbered = args.output/f'checkpoint_{step:06d}.pth'
                if numbered.exists():
                    numbered.unlink()
                os.link(args.output/'last.pth', numbered)
                for old in sorted(args.output.glob('checkpoint_*.pth'))[:-args.keep_checkpoints]:
                    old.unlink()
    if verification is not None:
        verification['after'] = {k: state_digest(v) for k, v in networks.items()}
        for key, frozen in [('W', args.freeze_writer), ('R', args.freeze_ocr)]:
            if frozen:
                assert verification['before'][key] == verification['after'][key], key+' changed while frozen'
        verification['completed_steps'] = args.steps-start
        (args.output/'update_verification.json').write_text(json.dumps(verification, indent=2))
    profile = {'stage': args.stage, 'completed_step': max(start, args.steps), 'seconds_this_invocation': time.monotonic()-begun,
               'torch': torch.__version__, 'device': args.device}
    if args.device.startswith('cuda'):
        profile.update(gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (args.output/'profile.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
