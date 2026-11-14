"""Fixed paired real/generation/support panels; no aesthetic score implied."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from research.longtail.prepare import read_records
from research.longtail.data import read_image


def font(size=15):
    try: return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True); p.add_argument('--first', type=Path, required=True)
    p.add_argument('--second', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = {r['sample_id']: r for r in read_records(args.records)}
    a = read_records(args.first); b = {r['sample_id']: r for r in read_records(args.second)}
    selected = ([r for r in a if set('ễỡẵ').intersection(r['reference'])][:8]
                +[r for r in a if not set('ễỡẵ').intersection(r['reference'])][:4])
    panel = Image.new('RGB', (1000, 55+len(selected)*75), 'white'); draw = ImageDraw.Draw(panel); label_font = font()
    for x, label in [(0, 'Text'), (160, 'Real'), (380, 'Glyph control'), (600, 'Factorized'), (820, 'Support example')]:
        draw.text((x, 5), label, font=label_font, fill='black')
    for i, r in enumerate(selected):
        y = 45+i*75; draw.text((2, y), r['reference'], font=label_font, fill='black')
        with Image.open(r['image_path']) as first, Image.open(b[r['sample_id']]['image_path']) as second:
            images = [read_image(records[r['sample_id']]), first.copy(), second.copy(), read_image(records[r['support_ids'][0]])]
        for x, image in zip([160, 380, 600, 820], images):
            # Display width is capped for compactness; evaluate original images.
            image = image.convert('L').resize((min(170, max(32, image.width*2)), 48))
            panel.paste(image, (x, y))
    args.output.parent.mkdir(parents=True, exist_ok=True); panel.save(args.output)


if __name__ == '__main__': main()
