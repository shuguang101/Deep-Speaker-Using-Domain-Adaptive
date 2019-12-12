# -*- coding:utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw


class RandColor(object):

    def __init__(self):
        self.r_set = set(range(1, 255))
        self.g_set = set(range(1, 255))
        self.b_set = set(range(1, 255))

    def get_random_rgb(self):
        r, g, b = 1, 1, 1
        if len(self.r_set) > 0 and len(self.g_set) > 0 and len(self.b_set) > 0:
            r = random.choice(list(self.r_set))
            g = random.choice(list(self.g_set))
            b = random.choice(list(self.b_set))

            self.r_set = self.r_set - {r}
            self.g_set = self.g_set - {g}
            self.b_set = self.b_set - {b}

        return r, g, b


class RandIcon(object):

    def __init__(self, length, alpha):
        self.length = length
        self.alpha = alpha
        self.rand_color = RandColor()

    def get_random_icon(self):
        color = self.rand_color.get_random_rgb()
        rgb_img = self.get_square(self.length, color, self.alpha)
        return rgb_img

    def get_square(self, length, color, alpha=1.0):
        pil_image = Image.new('RGBA', (length, length), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        draw.rectangle((0, 0, length - 1, length - 1), fill=color, outline=(0, 0, 0, 120))

        img_blender = Image.new('RGBA', (length, length), (255, 255, 255, 0))
        pil_image = Image.blend(img_blender, pil_image, alpha)

        pil_image = pil_image.convert('RGB')
        rgb_image = np.asanyarray(pil_image)

        return rgb_image


def plot_eer(frr_arr, far_arr, eer, th_root, text='', save_path=None):
    plt.figure()
    plt.plot(frr_arr, far_arr, '.-', linewidth=1)
    plt.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001))
    plt.annotate('th=%s, eer=%s' % (round(th_root, 3), round(eer, 3)), (eer, eer))
    plt.annotate('%s' % text, (0.45, 0.9))
    plt.grid()
    plt.xlabel('frr')
    plt.ylabel('far')
    plt.title('roc')

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
