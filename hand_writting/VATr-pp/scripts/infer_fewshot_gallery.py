"""Reproducible Vietnamese easy-to-hard gallery from an external style folder."""
import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import random
import sys
import time
import unicodedata as ud

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.vietnamese import load_manifest, to_tensor
from models.model import Generator
from models.OCR_network import strLabelConverter
from train_vietnamese import model_args, seed_all


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prompts():
    rows = []
    seen = set()

    def add(group, text, kind='word'):
        text = ud.normalize('NFC', text)
        if text not in seen:
            rows.append(dict(id=f'{len(rows)+1:04d}', group=group, text=text, kind=kind))
            seen.add(text)

    for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZđĐ0123456789':
        add('01_co_ban', c, 'single_character')
    for word in ('ba be bi bo bu me cha con em anh chi nam lan hoa mai hai ba bon nha ban hoc di ve '
                 'bạn mẹ bố chị nhà bàn học đi về đẹp vui yêu thương bình an xin chào cảm ơn hôm nay '
                 'ngày mới trời xanh nắng vàng gió nhẹ mùa xuân gia đình quê hương Việt Nam').split():
        add('01_co_ban', word)
    for base in 'aăâeêioôơuưy':
        for tone in ['', '\u0301', '\u0300', '\u0309', '\u0303', '\u0323']:
            c = ud.normalize('NFC', ud.normalize('NFD', base) + tone)
            add('02_nguyen_am_va_thanh', c, 'single_character')
            add('02_nguyen_am_va_thanh', c.upper(), 'single_character')
    for word in ('ma má mà mả mã mạ ba bá bà bả bã bạ la lá là lả lã lạ '
                 'ban bán bàn bản bãn bạn can cán càn cản cãn cạn '
                 'tâm tấm tầm tẩm tẫm tậm tăm tắm tằm tẳm tẵm tặm '
                 'me mé mè mẻ mẽ mẹ mê mế mề mể mễ mệ '
                 'mo mó mò mỏ mõ mọ mô mố mồ mổ mỗ mộ mơ mớ mờ mở mỡ mợ '
                 'mu mú mù mủ mũ mụ mư mứ mừ mử mữ mự '
                 'mi mí mì mỉ mĩ mị my mý mỳ mỷ mỹ mỵ').split():
        add('03_doi_chieu_dau', word, 'controlled_tone_probe')
    for word in ('Nguyễn Trần Lê Phạm Hoàng Huỳnh Phan Vũ Võ Đặng Bùi Đỗ Hồ Ngô Dương Lý '
                 'Nguyễn Thị Lan Anh Vĩnh Tường Nam Triệu Mười Trăm Hai '
                 'Bách Dũng Huyền Thảo Hiếu Nghĩa Quyên Yến Tuyết Quỳnh Kiều Diễm '
                 'Nhã Mỹ Hữu Hưởng Cường Tưởng Thưởng Quyền Quyết Huệ Thủy Phượng').split():
        add('04_ten_rieng', word)
        add('05_viet_hoa', word.upper())
    for word in ('tiếng Việt người trường đường thưởng tưởng trưởng thường thương '
                 'nghiêng nghiễng ngưỡng nguyễn huyễn chuyển chuyện quyển quyền quyến quyện '
                 'khuyến khuỷu khuấy khuất khoảnh khuôn khuông khuếch ngoằn ngoèo ngoẵng '
                 'ngoảnh ngoặt ngoắc nguệch ngoạc khuỵu quỵt quỹ quỵ '
                 'bỗng rỗng nhẫn nhẵng sẵn vững vĩnh bẫng bẵng bẫy mẫm '
                 'lưỡng cưỡng dưỡng tưởng tượng hưởng thưởng trưởng ngưởng '
                 'ướt ước ược ương ường ưởng ưỡng ượng ươi ười ưởi ưỡi ượi '
                 'yêu yếu yểu yễu yệu yên yến yền yển yễn yện '
                 'uyên uyến uyền uyển uyễn uyện uyết uyệt '
                 'oanh oánh oành oảnh oãnh oạnh oăng oắng oằng oẳng oẵng oặng').split():
        add('06_tu_kho', word, 'word_or_controlled_syllable')
        add('07_hoa_kho', word.upper(), 'word_or_controlled_syllable')
    # Explicitly synthetic composition probes, not a Vietnamese dictionary.
    for base in 'ăâêôơư':
        for tone in ['', '\u0301', '\u0300', '\u0309', '\u0303', '\u0323']:
            c = ud.normalize('NFC', ud.normalize('NFD', base) + tone)
            for prefix, suffix in [('ng', 'n'), ('th', 'ng')]:
                word = prefix + c + suffix
                add('08_to_hop_stress', word, 'synthetic_composition_probe')
                add('08_to_hop_stress', word.upper(), 'synthetic_composition_probe')
    for line in [
        'Nguyễn Thị Lan Anh', 'Vĩnh Tường Nam Triệu Mười Trăm Hai',
        'NGUYỄN THỊ LAN ANH', 'NGUYỄN TRẦN LÊ PHẠM HOÀNG HUỲNH',
        'Tiếng Việt rất đẹp', 'Xin chào Việt Nam', 'Hôm nay trời nắng đẹp',
        'Gia đình là nơi để trở về', 'Người thầy đứng trước cổng trường',
        'Những người bạn cùng trưởng thành', 'Quyển truyện của Nguyễn Thị Mỹ',
        'Tưởng thưởng những nỗ lực bền bỉ', 'Nghiêng mình trước những ước mơ',
        'Chữ thường chữ hoa và những dấu thanh', 'ma má mà mả mã mạ',
        'a ă â e ê i o ô ơ u ư y', 'A Ă Â E Ê I O Ô Ơ U Ư Y',
        'Ắ Ằ Ẳ Ẵ Ặ Ấ Ầ Ẩ Ẫ Ậ', 'Ế Ề Ể Ễ Ệ Ớ Ờ Ở Ỡ Ợ',
        'Ứ Ừ Ử Ữ Ự Ỳ Ỵ Ỷ Ỹ', 'Nguyễn NGUYỄN Ễ ễ',
        'NGƯỠNG NGHIỄNG HUYỄN QUỸ KHUỴU',
        'Năm 2026 ngày 16 tháng 11', '0123456789 100 1000 250000',
    ]:
        add('09_cau_va_cum_tu', line, 'wordwise_composed_line')
    return rows


def white_rgb(path):
    rgba = Image.open(path).convert('RGBA')
    white = Image.new('RGBA', rgba.size, 'white')
    return Image.alpha_composite(white, rgba).convert('RGB')


def prepare(path, clean):
    rgb = white_rgb(path)
    if clean:
        a = np.asarray(rgb).astype(np.float32)
        # Blue ink is dark in red/green but light in blue; gray paper/black rules
        # have little blue chroma. Keep a grayscale fallback for other inputs.
        blue = np.maximum(0, a[:, :, 2] - np.minimum(a[:, :, 0], a[:, :, 1]))
        if (blue > 25).sum() > 20:
            ink = np.clip((blue - 12) / max(30, np.percentile(blue[blue > 25], 95) - 12), 0, 1)
            im = Image.fromarray(np.uint8(255 * (1 - ink)))
        else:
            a = np.asarray(rgb.convert('L')).astype(np.float32)
            lo, hi = np.percentile(a, [3, 85])
            im = Image.fromarray(np.uint8(np.clip((a-lo)/max(1, hi-lo), 0, 1)*255))
        ink = np.asarray(im) < 200
        ys, xs = np.where(ink)
        if len(xs):
            im = im.crop((max(0, xs.min()-3), max(0, ys.min()-3), min(im.width, xs.max()+4), min(im.height, ys.max()+4)))
    else:
        im = rgb.convert('L')
    im = im.resize((max(1, round(im.width*32/im.height)), 32), Image.Resampling.BICUBIC)
    if im.width > 192:
        im = im.resize((192, 32), Image.Resampling.BICUBIC)
    padded = Image.new('L', (192, 32), 255)
    padded.paste(im, (0, 0))
    return padded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--style-folder', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(0.08)
    seed_all(742)
    start = time.time()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    manifest = load_manifest(Path(checkpoint['config']['manifest']), verify_sources=False)
    assert checkpoint['alphabet'] == manifest['alphabet']
    cfg = model_args(manifest, args.device, not checkpoint['config']['legacy_generator_layout'])
    cfg.correct_style_layout = not checkpoint['config']['legacy_generator_layout']
    model = Generator(cfg).to(args.device).eval()
    model.load_state_dict(checkpoint['model']['G'], strict=True)
    converter = strLabelConverter(checkpoint['alphabet'])
    cases = prompts()
    for row in cases:
        converter.encode(row['text'].split())
    paths = sorted(args.style_folder.glob('*.png'), key=lambda x: (int(x.stem) if x.stem.isdigit() else 9999, x.name))
    if not paths:
        raise ValueError('No PNG references')
    variants = [('clean_742', True, 742), ('clean_743', True, 743), ('raw_742', False, 742)]
    metadata = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
                    checkpoint_step=checkpoint['step'], source_folder=str(args.style_folder.resolve()),
                    source_files=[dict(path=str(x.resolve()), sha256=digest(x)) for x in paths],
                    num_examples=checkpoint['config']['num_examples'], prompts=len(cases),
                    method='Native generator, eval FP32, no training; same support for every prompt within variant.',
                    caveats=['Grayscale output, not ink-color transfer.',
                             'Different support seeds are sensitivity checks, not independent writer samples.',
                             'Multiword lines concatenate word images with 16px gaps; not continuous sentence generation.',
                             'Controlled/synthetic syllable probes are not all dictionary words.',
                             'Seen-reference words are not held-out generalization tests.'], variants=[])
    records = []
    font = ImageFont.truetype('/opt/conda/fonts/DejaVuSans.ttf', 18)
    for variant, clean, seed in variants:
        seed_all(seed)
        folder = out/variant
        folder.mkdir()
        rng = random.Random(seed)
        chosen = list(paths)
        n = checkpoint['config']['num_examples']
        if len(chosen) >= n:
            chosen = rng.sample(chosen, n)
        else:
            chosen += rng.choices(paths, k=n-len(chosen))
        images = [prepare(path, clean) for path in chosen]
        sheet = Image.new('RGB', (600, n*80+38), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 4), variant + ' | all 12 inputs + 3 repeated', fill='black', font=font)
        for i, (path, im) in enumerate(zip(chosen, images)):
            draw.text((8, 38+i*80), path.name, fill='black', font=font)
            sheet.paste(im.resize((384, 64)), (150, 38+i*80))
        sheet.save(out/f'references_{variant}.png')
        style = torch.stack([to_tensor(im)[0] for im in images]).unsqueeze(0).to(args.device)
        cache = {}
        with torch.inference_mode():
            memory = model.compute_style(style)
            for idx, row in enumerate(cases):
                results = []
                for word in row['text'].split():
                    if word not in cache:
                        q, _, _ = converter.encode([word])
                        fake = model.generate(memory, q.to(args.device))
                        a = ((fake[0, 0].clamp(-1, 1).cpu().numpy()+1)*127.5).astype('uint8')
                        cache[word] = Image.fromarray(a)
                    results.append(cache[word])
                im = Image.new('L', (sum(x.width for x in results)+16*(len(results)-1), 32), 255)
                x = 0
                for word_im in results:
                    im.paste(word_im, (x, 0))
                    x += word_im.width + 16
                rel = f'{variant}/{row["id"]}.png'
                im.save(out/rel)
                records.append(dict(row, variant=variant, image=rel, width=im.width, height=im.height))
                if idx % 100 == 0:
                    print(f'{variant}: {idx}/{len(cases)}', flush=True)
        metadata['variants'].append(dict(name=variant, clean=clean, seed=seed, references=[str(x.resolve()) for x in chosen], unique_generated_words=len(cache)))
    groups = defaultdict(list)
    for row in records:
        if row['variant'] == 'clean_742':
            groups[row['group']].append(row)
    sheets = []
    for group, rows in groups.items():
        for page, offset in enumerate(range(0, len(rows), 32), 1):
            subset = rows[offset:offset+32]
            width = 1500 if group == '09_cau_va_cum_tu' else 1050
            panel = Image.new('RGB', (width, 58+len(subset)*86), 'white')
            draw = ImageDraw.Draw(panel)
            draw.text((16, 12), f'{group} | clean_742 | page {page} | step {checkpoint["step"]}', fill='black', font=font)
            for i, row in enumerate(subset):
                y = 58+i*86
                draw.text((12, y+8), row['id']+'  '+row['text'], fill='black', font=font)
                image = Image.open(out/row['image']).resize((row['width']*2, 64), Image.Resampling.NEAREST)
                # Long lines fit the remaining width, preserving aspect ratio.
                image.thumbnail((width-365, 64))
                panel.paste(image, (350, y))
                draw.line((0, y+80, width, y+80), fill='#dddddd')
            name = f'{group}_page_{page:02d}.png'
            panel.save(out/name)
            sheets.append(name)
    metadata.update(images=len(records), elapsed_seconds=round(time.time()-start, 2),
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                    contact_sheets=sheets)
    (out/'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    (out/'samples.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    (out/'prompts.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    options=''.join(f'<option>{html.escape(x)}</option>' for x in groups)
    data=json.dumps(records, ensure_ascii=False).replace('</', '<\\/')
    (out/'index.html').write_text('''<!doctype html><html lang="vi"><meta charset="utf-8"><title>VATr++ few-shot 176500</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#17212f}header{position:sticky;top:0;background:#fff;padding:16px;z-index:2}select,input{padding:10px;margin:5px}article{background:white;border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}article img{height:64px;max-width:100%;object-fit:contain;object-position:left;image-rendering:pixelated}small{color:#596579}a{color:#155fc0}</style>
<header><h1>Few-shot · checkpoint 176.500</h1><p>12 ảnh gốc → grayscale. Không fine-tune. clean = tách nét bút xanh khỏi nền; raw = grayscale gốc.</p>
<p>Câu nhiều từ ghép từ từng ảnh từ. Mục stress chứa tổ hợp tổng hợp, không phải tất cả đều là từ có nghĩa. Chưa có điểm chất lượng độc lập.</p>
<select id="variant"><option>clean_742</option><option>clean_743</option><option>raw_742</option><option value="all">Tất cả để so sánh</option></select>
<select id="group"><option value="all">Tất cả mức khó</option>'''+options+'''</select><input id="query" placeholder="Tìm chữ: NGUYỄN, Ễ, ..."><span id="count"></span>
<p><a href="references_clean_742.png">Support clean 742</a> · <a href="references_clean_743.png">Support clean 743</a> · <a href="references_raw_742.png">Support raw</a> · <a href="run.json">Thông số và checksum</a></p></header><main id="gallery"></main>
<script>const rows='''+data+''';const v=document.querySelector('#variant'),g=document.querySelector('#group'),q=document.querySelector('#query'),box=document.querySelector('#gallery');function render(){box.replaceChildren();const query=q.value.normalize('NFC');const filtered=rows.filter(x=>(v.value==='all'||x.variant===v.value)&&(g.value==='all'||x.group===g.value)&&x.text.includes(query));document.querySelector('#count').textContent=filtered.length+' ảnh';for(const row of filtered){const a=document.createElement('article'),title=document.createElement('strong'),meta=document.createElement('small'),link=document.createElement('a'),im=document.createElement('img');title.textContent=row.text+' ';meta.textContent=row.group+' · '+row.variant+' · '+row.kind;im.src=row.image;im.loading='lazy';im.alt=row.text;link.href=row.image;link.append(im);a.append(title,meta,document.createElement('br'),link);box.append(a)}}for(const e of [v,g,q])e.addEventListener('input',render);render();</script></html>''')
    (out/'README.md').write_text(f'''# Few-shot inference 16/11/2026

Checkpoint: component 176.500, SHA-256 `{metadata['checkpoint_sha256']}`.

{len(cases)} prompt × 3 support variants = {len(records)} PNG. All 12 inputs are retained plus 3 repeats to fill 15 slots. Seeds alter repeated references, not a learned stochastic style distribution. Single characters are generated as single-character queries. Multiword lines concatenate word outputs with 16px gaps. Synthetic probes are labelled explicitly.

Open [index.html](index.html) to filter by category, prompt and variant. Primary previews use clean_742. Compare raw_742 before attributing changes to the model. Background cleaning is deterministic blue-chroma extraction, not retraining or image synthesis. Native model emits grayscale at 32px height; previews enlarge it for inspection.

No human/OCR quality scores are claimed. Checkpoint selection is based on the previous fixed benchmark, not a guarantee of optimality for this external writer. Matching reference words are demonstrations, not held-out tests.

See run.json for all inputs/checksums, support composition, memory and runtime; prompts.json and samples.json contain the exact workload and outputs.
''')
    print(json.dumps(dict(output=str(out), prompts=len(cases), images=len(records), seconds=metadata['elapsed_seconds']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
