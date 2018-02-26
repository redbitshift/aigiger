"""This util generates ground truth for the classifier."""

import itertools
import math
import os
import os.path
import sys

import gflags
from PIL import Image

gflags.DEFINE_string('in_dir', '', 'Input image directory')
gflags.DEFINE_string('out_dir', '', 'Output data directory')
gflags.DEFINE_integer('size', 128, 'Size of a single ground truth image')

F = gflags.FLAGS

gflags.register_validator('in_dir', os.path.isdir, 'Invalid input path')
gflags.register_validator('out_dir', os.path.isdir, 'Invalid output path')


def transform_image(img, flip, angle, scale):
    img = img.copy()
    w, h = img.size
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if angle != 0:
        img = img.rotate(angle)
        # This uncovered black stripes. Let's crop them away.
        dy = abs(w / 2 * math.sin(angle / 180 * math.pi))
        dx = abs(h / 2 * math.sin(angle / 180 * math.pi))
        img = img.crop((dx, dy, w - dx, h - dy))

        rw, rh = img.size
        scale *= min(w / rw, h / rh)  # Cropping made image smaller; compensate.

    if scale != 1:
        new_w = int(w * scale + .5)
        new_h = int(h * scale + .5)
        img = img.resize((new_w, new_h), Image.ANTIALIAS)

    return img


def prepare_scales(img):
    w, h = img.size

    # Prepare list of transformations.
    scale = max(F.size / w, F.size / h)
    if scale > 1:
        print('w Input image is too small: %d x %d, skipping' % (w, h))
        return []

    scales = []
    max_scale = scale * 1.33
    while scale < max_scale:
        scales.append(scale)
        scale *= 1.15

    return scales


def make_examples(img_path):
    image = Image.open(img_path)

    # Prepare list of transformations.
    scales = prepare_scales(image)
    if not scales:
        return []

    angles = [-5, 0, 5]
    flips = [True, False]

    examples = []
    for flip, angle, scale in itertools.product(flips, angles, scales):
        img = transform_image(image, flip, angle, scale)
        w, h = img.size
        for x in range(0, w - F.size + 1, F.size):
            for y in range(0, h - F.size + 1, F.size):
                example = img.crop((x, y, x + F.size, y + F.size))
                examples.append(example)
    return examples


def main(argv):
    examples = []
    for f in os.listdir(F.in_dir):
        examples += make_examples(os.path.join(F.in_dir, f))
    print('i Generated %d examples' % len(examples))

    for i, x in enumerate(examples):
        x.save(os.path.join(F.out_dir, '%04d.jpeg' % i))

    return 0


if __name__ == '__main__':
    F(sys.argv)
    sys.exit(main(sys.argv))
