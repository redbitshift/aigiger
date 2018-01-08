"""This util generates images to use in generator."""

import csv
import io
import os
import os.path
import random
import sys
from concurrent.futures import as_completed, ThreadPoolExecutor
from urllib.request import urlopen

import gflags
from PIL import Image

gflags.DEFINE_string('in_csv', '', 'OpenImages CSV')
gflags.DEFINE_string('out_dir', '', 'Output data directory')
gflags.DEFINE_integer('size', 128, 'Size of a single ground truth image')
gflags.DEFINE_integer('count', 5000, 'Number of images to generate')

F = gflags.FLAGS

gflags.register_validator('in_csv', os.path.isfile, 'Invalid input path')
gflags.register_validator('out_dir', os.path.isdir, 'Invalid output path')


def make_image(url, num):
    with urlopen(url) as f:
        data = f.read()
    buf = io.BytesIO(data)
    image = Image.open(buf)
    if image.mode != 'RGB':
        raise Exception('Invalid mode %s' % image.mode)

    w, h = image.size
    scale = max(F.size / w, F.size / h)
    new_w = int(w * scale + .5)
    new_h = int(h * scale + .5)
    image = image.resize((new_w, new_h), Image.ANTIALIAS)

    x0 = (new_w - F.size) // 2
    y0 = (new_h - F.size) // 2
    x1 = x0 + F.size
    y1 = y0 + F.size
    image = image.crop((x0, y0, x1, y1))

    image.save(os.path.join(F.out_dir, '%04d.jpeg' % num))


def main(argv):
    reader = csv.reader(open(F.in_csv))
    urls = [row[2] for row in reader]
    sample = random.sample(urls, F.count)

    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(make_image, url, num): url
                   for num, url in enumerate(sample)}
        for num, future in enumerate(as_completed(futures)):
            url = futures[future]
            try:
                future.result()
                if num % 10 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
            except Exception as e:
                print('e Exception for %s: %s' % (url, e))
    print


if __name__ == '__main__':
    F(sys.argv)
    sys.exit(main(sys.argv))
