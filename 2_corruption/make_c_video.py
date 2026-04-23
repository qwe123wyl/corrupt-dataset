#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_c_video.py
────────────────────────────────────────────────────────────────
读取 noise_assignment.py 的输出 CSV，按 assigned_noise 字段
对视频帧进行污染，输出到 save_dir/。

污染严重程度由 --severity 指定（1-5）。

使用方式：
  python make_c_video.py \
      --noise-assignment-csv /path/to/dataset_with_noise.csv \
      --video-frame-dir /path/to/image_mulframe_test \
      --output-dir /path/to/image_mulframe_test-C \
      --severity 3 \
      --num-workers 8

注意：
  视频帧目录结构：
    image_mulframe_test/
      frame_0/xxx.jpg
      frame_1/xxx.jpg
      ...
      frame_9/xxx.jpg

  输出目录结构：
    image_mulframe_test-C/
      gaussian_noise/
        severity_3/
          frame_0/xxx.jpg
          frame_1/xxx.jpg
          ...
          frame_9/xxx.jpg
      shot_noise/
        ...
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image as PILImage
import torchvision.transforms as trn
from io import BytesIO
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════
# 25 种噪声 → 视频污染函数映射（仅包含视频/VA 联合部分）
# ══════════════════════════════════════════════════════════════

# 视频噪声名称（与 CSV 中的 assigned_noise 列值一致）
VIDEO_NOISES = [
    "V_gaussian_noise", "V_shot_noise", "V_impulse_noise",
    "V_defocus_blur", "V_glass_blur", "V_motion_blur", "V_zoom_blur",
    "V_rain", "V_snow", "V_frost", "V_fog",
    "V_brightness", "V_contrast", "V_elastic_transform",
    "V_pixelate", "V_jpeg_compression",
]
VA_NOISES_VIDEO = ["VA_gaussian", "VA_rain"]   # 联合噪声中也涉及视频

NEED_VIDEO_CORRUPTION = set(VIDEO_NOISES + VA_NOISES_VIDEO + ["Missing_video"])


# ══════════════════════════════════════════════════════════════
# 噪声函数库（来自 ICLR-READ，原样保留）
# ══════════════════════════════════════════════════════════════

import skimage as sk
from skimage.filters import gaussian
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates
from PIL import Image as PILImage


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def plasma_fractal(mapsize=256, wibbledecay=3):
    assert mapsize & (mapsize - 1) == 0
    maparray = np.zeros((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.0

    # 四个角的坐标（处理边界换行）
    def corners(si, sj):
        return (
            maparray[si,                      sj % mapsize],
            maparray[si,                      (sj + stepsize) % mapsize],
            maparray[(si + stepsize) % mapsize, sj % mapsize],
            maparray[(si + stepsize) % mapsize, (sj + stepsize) % mapsize],
        )

    while stepsize >= 2:
        half = stepsize // 2

        # fillsquares：填充每个网格中心
        for si in range(0, mapsize, stepsize):
            for sj in range(0, mapsize, stepsize):
                c0, c1, c2, c3 = corners(si, sj)
                maparray[(si + half) % mapsize, (sj + half) % mapsize] = (
                    (c0 + c1 + c2 + c3) / 4 + wibble * np.random.uniform(-1, 1)
                )

        # filldiamonds：填充每个网格的四个边中点
        for si in range(0, mapsize, stepsize):
            for sj in range(0, mapsize, stepsize):
                ci, cj = (si + half) % mapsize, (sj + half) % mapsize
                # 上边中点
                maparray[si, cj] = (maparray[si, sj] + maparray[si, (sj + stepsize) % mapsize] +
                                    maparray[(si - half) % mapsize, cj] + maparray[ci, cj]) / 4 + wibble * np.random.uniform(-1, 1)
                # 下边中点
                maparray[(si + stepsize) % mapsize, cj] = (maparray[si, sj] + maparray[si, (sj + stepsize) % mapsize] +
                                                            maparray[(si + half * 3) % mapsize, cj] + maparray[ci, cj]) / 4 + wibble * np.random.uniform(-1, 1)
                # 左边中点
                maparray[ci, sj] = (maparray[si, sj] + maparray[(si + stepsize) % mapsize, sj] +
                                    maparray[ci, (cj - half) % mapsize] + maparray[ci, cj]) / 4 + wibble * np.random.uniform(-1, 1)
                # 右边中点
                maparray[ci, (sj + stepsize) % mapsize] = (maparray[si, (sj + stepsize) % mapsize] +
                                                            maparray[(si + stepsize) % mapsize, (sj + stepsize) % mapsize] +
                                                            maparray[ci, (cj + half) % mapsize] + maparray[ci, cj]) / 4 + wibble * np.random.uniform(-1, 1)

        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    vmax = maparray.max()
    return maparray / vmax if vmax > 0 else maparray


# ── 15 种视频污染函数 ────────────────────────────────────────

def _gaussian_noise(x, severity):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def _shot_noise(x, severity):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def _impulse_noise(x, severity):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def _defocus_blur(x, severity):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])
    channels = [cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)]
    return np.clip(np.array(channels).transpose((1, 2, 0)), 0, 1) * 255


def _glass_blur(x, severity):
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)
    for _ in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]
    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255


def _motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    radius, sigma = c[0], c[1]

    x = np.array(x)

    angle = np.random.uniform(-45, 45)
    kernel_size = radius * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        kernel[radius, i] = np.exp(-0.5 * ((i - radius) / sigma) ** 2)
    kernel = kernel / kernel.sum()
    M = cv2.getRotationMatrix2D((radius, radius), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / kernel.sum()

    x = cv2.filter2D(x, -1, kernel)

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)

    return np.clip(x, 0, 255).astype(np.uint8)


def _zoom_blur(x, severity):
    c = [np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02), np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]
    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zf in c:
        out += clipped_zoom(x, zf)
    return np.clip((x + out) / (len(c) + 1), 0, 1) * 255


def _fog(x, severity):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


FROST_FILES_RESOLVED = None   # 由 main 填充，需在模块顶层声明 global


def _frost(x, severity):
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    idx = np.random.randint(6)
    # 优先使用 resolved 路径，否则 fallback 到本地相对路径
    if FROST_FILES_RESOLVED:
        frost_file = FROST_FILES_RESOLVED[idx]
    else:
        frost_file = ['./frost1.png', './frost2.png', './frost3.png',
                      './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
    frost = cv2.imread(frost_file)
    xs = np.random.randint(0, frost.shape[0] - 224)
    ys = np.random.randint(0, frost.shape[1] - 224)
    frost = frost[xs:xs + 224, ys:ys + 224][..., [2, 1, 0]]
    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def _missing_video(x, severity):
    """生成全黑帧（224x224x3），用于 Missing_video 噪声类型"""
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = np.clip(snow_layer.squeeze(), 0, 1).astype(np.float32)
    radius, sigma = int(c[4]), c[5]
    kernel_size = radius * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        kernel[radius, i] = np.exp(-0.5 * ((i - radius) / sigma) ** 2)
    kernel = kernel / kernel.sum()
    M = cv2.getRotationMatrix2D((radius, radius), np.random.uniform(-135, -45), 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / kernel.sum()
    snow_layer = cv2.filter2D(snow_layer, -1, kernel)
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def _brightness(x, severity):
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    return np.clip(sk.color.hsv2rgb(x), 0, 1) * 255


def _contrast(x, severity):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def _elastic_transform(x, severity):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1), (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02), (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
    image = np.array(x, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
    x_idx, y_idx, z_idx = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y_idx + dy, (-1, 1)), np.reshape(x_idx + dx, (-1, 1)), np.reshape(z_idx, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def _pixelate(x, severity):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    if isinstance(x, np.ndarray):
        # numpy array → 用 OpenCV 重缩放
        h, w = x.shape[:2]
        small_h, small_w = int(h * c), int(w * c)
        interp = cv2.INTER_AREA if c < 1 else cv2.INTER_CUBIC
        x_small = cv2.resize(x, (small_w, small_h), interpolation=interp)
        return cv2.resize(x_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        # PIL Image → 原有逻辑
        x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
        x = x.resize((224, 224), PILImage.BOX)
        return x


def _jpeg_compression(x, severity):
    c = [25, 18, 15, 10, 7][severity - 1]
    if isinstance(x, np.ndarray):
        # numpy array → 用 OpenCV JPEG 编码
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), c]
        _, buf = cv2.imencode('.jpg', x, encode_param)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        # PIL Image → 原有逻辑
        buf = BytesIO()
        x.save(buf, 'JPEG', quality=c)
        buf.seek(0)
        return PILImage.open(buf)


def _rain(img, severity):
    """Adds rain. Adapted from MiOIR."""
    img = np.array(img)
    img = img.copy()
    w = 3
    length = np.random.randint(20, 40)
    angle = np.random.randint(-30, 30)
    value = np.random.randint(50, 100)
    noise = np.random.uniform(0, 256, img.shape[0:2])
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])
    noise = cv2.filter2D(noise, -1, k)
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))
    k = cv2.GaussianBlur(k, (w, w), 0)
    blurred = cv2.filter2D(noise, -1, k)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    rain = np.expand_dims(blurred, 2)
    rain = np.repeat(rain, 3, 2)
    img = img.astype('float32') + rain
    np.clip(img, 0, 255, out=img)
    return img.round().astype(np.uint8)


# VA 噪声 → 视频输出子目录名（独立目录，与纯视频噪声 V_gaussian/V_rain 分离）
VA_NOISES_VIDEO_CORRUPTION_DIR = {
    "VA_gaussian": "va_gaussian",
    "VA_rain":     "va_rain",
}

# 噪声名称 → 函数映射
NAME_TO_FUNC = {
    "V_gaussian_noise":    _gaussian_noise,
    "V_shot_noise":       _shot_noise,
    "V_impulse_noise":    _impulse_noise,
    "V_defocus_blur":     _defocus_blur,
    "V_glass_blur":       _glass_blur,
    "V_motion_blur":      _motion_blur,
    "V_zoom_blur":        _zoom_blur,
    "V_rain":             _rain,
    "V_snow":             _snow,
    "V_frost":            _frost,
    "V_fog":              _fog,
    "V_brightness":        _brightness,
    "V_contrast":         _contrast,
    "V_elastic_transform": _elastic_transform,
    "V_pixelate":         _pixelate,
    "V_jpeg_compression": _jpeg_compression,
    "VA_gaussian":        _gaussian_noise,
    "VA_rain":            _rain,
    "Missing_video":       _missing_video,
}


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

def pil_loader(path):
    with open(path, 'rb') as f:
        return PILImage.open(f).convert('RGB')


class CorruptVideoDataset(data.Dataset):
    """只处理需要视频污染的样本"""

    def __init__(self, video_frame_dir: str,
                 noise_assignment_csv: str,
                 output_dir: str,
                 severity: int,
                 frame_count: int = 10,
                 transform=None):
        super().__init__()

        self.frame_dir = video_frame_dir
        self.severity = severity
        self.frame_count = frame_count
        self.transform = transform or trn.Compose([
            trn.Resize(256), trn.CenterCrop(224)
        ])
        self.output_dir = output_dir

        # ── 读取噪声分配 CSV ───────────────────────────────
        df = pd.read_csv(noise_assignment_csv)
        for col in ["sample_id", "assigned_noise"]:
            if col not in df.columns:
                for alias in ["video_id", "VideoId"]:
                    if alias in df.columns and col == "sample_id":
                        df = df.rename(columns={alias: col})
                        break
        df["sample_id"] = df["sample_id"].astype(str)
        df["assigned_noise"] = df["assigned_noise"].astype(str)

        # 只保留需要视频污染的样本
        df_filtered = df[df["assigned_noise"].isin(NEED_VIDEO_CORRUPTION)].copy()
        self.df = df_filtered.reset_index(drop=True)
        print(f"[Dataset] 共 {len(self.df)} 个样本需要视频污染 "
              f"（占总 {len(df)} 个中的 {len(self.df)/len(df)*100:.1f}%）")

    def __len__(self):
        return len(self.df) * self.frame_count

    def __getitem__(self, idx):
        sample_idx = idx // self.frame_count
        frame_idx = idx % self.frame_count

        row = self.df.iloc[sample_idx]
        vid = row["sample_id"]
        noise = row["assigned_noise"]

        # 读取帧图像
        frame_path = os.path.join(
            self.frame_dir, f"frame_{frame_idx}", f"{vid}.jpg")
        try:
            img = pil_loader(frame_path)
        except Exception:
            # 尝试 .png
            frame_path = os.path.join(
                self.frame_dir, f"frame_{frame_idx}", f"{vid}.png")
            img = pil_loader(frame_path)

        # 预处理 + 变换 + 噪声
        img_tensor = self.transform(img)
        corrupt_func = NAME_TO_FUNC.get(noise, None)

        if corrupt_func is not None:
            # 对 PIL Image 执行噪声（convert 回 numpy）
            img_np = np.array(img_tensor)
            img_corrupted = corrupt_func(img_np, self.severity)
            # 如果函数返回的是 numpy（HWC），转回 PIL
            if isinstance(img_corrupted, np.ndarray):
                # 处理 shape：可能是 (H,W,C) 也可能是 (C,H,W)
                if img_corrupted.ndim == 3 and img_corrupted.shape[0] == 3:
                    img_corrupted = img_corrupted.transpose(1, 2, 0)
                img_out = PILImage.fromarray(
                    np.uint8(np.clip(img_corrupted, 0, 255)))
            else:
                img_out = img_corrupted
        else:
            img_out = img_tensor

        # 保存
        if noise.startswith("V_"):
            corruption_dir = noise.replace("V_", "").lower()
        elif noise.startswith("VA_"):
            corruption_dir = VA_NOISES_VIDEO_CORRUPTION_DIR.get(noise, noise.lower().replace("_", "_"))
        elif noise == "Missing_video":
            corruption_dir = "missing_video"
        else:
            corruption_dir = noise.lower()

        save_dir = os.path.join(
            self.output_dir, corruption_dir,
            f"severity_{self.severity}", f"frame_{frame_idx}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{vid}.jpg")

        img_out.save(save_path, quality=85, optimize=True)

        return 0


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据噪声分配结果生成污染视频帧")
    parser.add_argument("--noise-assignment-csv", type=str, required=True,
                        help="noise_assignment.py 输出的 CSV")
    parser.add_argument("--video-frame-dir", type=str, required=True,
                        help="原始视频帧目录（含 frame_0~frame_9）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="污染视频帧输出目录")
    parser.add_argument("--severity", type=int, default=3,
                        choices=[1, 2, 3, 4, 5],
                        help="污染严重程度（1-5）")
    parser.add_argument("--frame-count", type=int, default=10,
                        help="每条视频的帧数（默认 10）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="DataLoader batch_size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader num_workers")

    args = parser.parse_args()

    # frost 图片使用相对于本脚本目录的路径（兼容 git clone 后直接运行）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frost_file_list = ['frost1.png', 'frost2.png', 'frost3.png',
                       'frost4.jpg', 'frost5.jpg', 'frost6.jpg']
    frost_base = os.path.join(script_dir, "make_corruptions")
    if os.path.isdir(frost_base):
        resolved = [os.path.join(frost_base, f) for f in frost_file_list]
        if all(os.path.exists(f) for f in resolved):
            FROST_FILES_RESOLVED = resolved
            print(f"[INFO] frost 图片使用: {frost_base}/")
        else:
            FROST_FILES_RESOLVED = None
            print(f"[警告] make_corruptions/ 目录不完整，frost 图片缺失")
    else:
        FROST_FILES_RESOLVED = None
        print(f"[警告] make_corruptions/ 目录不存在，frost 图片缺失")

    dataset = CorruptVideoDataset(
        video_frame_dir=args.video_frame_dir,
        noise_assignment_csv=args.noise_assignment_csv,
        output_dir=args.output_dir,
        severity=args.severity,
        frame_count=args.frame_count,
    )

    loader = data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    print(f"\n开始生成污染视频帧（severity={args.severity}）...")
    total = len(dataset)
    done = 0
    import time
    t0 = time.time()

    for batch in loader:
        done += args.batch_size
        if done % 500 <= args.batch_size:
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0
            print(f"  进度: {done}/{total} ({done/total*100:.1f}%)  "
                  f"速度: {speed:.0f} img/s  ETA: {eta:.0f}s")

    print(f"\n[完成] 污染视频帧已保存到: {args.output_dir}")
