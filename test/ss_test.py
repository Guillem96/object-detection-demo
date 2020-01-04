import torch
from PIL import Image, ImageDraw

import odet.utils.ss as ss
import odet.utils.bb as bb_utils

def _scale_up_rects(rects, image_size):
    h, w = image_size
    x, y, x2, y2 = rects.split(1, dim=1)
    return torch.cat([x * (w - 1), y * (h - 1),
                      x2 * (w - 1), y2 * (h - 1)], dim=1).long()


if __name__ == "__main__":
    im = Image.open('images/celebi.png')
    im = im.resize((300, 250))
    regions = ss.ss(im, normalize_bbs=False)

    draw = ImageDraw.Draw(im)
    for r in regions.tolist():
        draw.rectangle(r)

    im.show()
