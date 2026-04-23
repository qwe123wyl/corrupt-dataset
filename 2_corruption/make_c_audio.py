#!/usr/bin/env python3
"""
make_c_audio.py
────────────────────────────────────────────────────────────────
读取 noise_assignment.py 的输出 CSV，按 assigned_noise 字段
对音频进行污染，输出到 save_dir/。

污染方式：
  - 音频噪声（A_xxx）：叠加高斯噪声或混入环境音
  - 视听联合噪声（VA_xxx）：混入对应的环境音
  - 缺失模态（Missing_audio）：生成静音音频

使用方式：
  python make_c_audio.py \
      --noise-assignment-csv /path/to/dataset_with_noise.csv \
      --audio-dir /path/to/audio_test \
      --weather-path /path/to/weather_audios \
      --output-dir /path/to/audio_test-C \
      --severity 3 \
      --num-workers 4

依赖：
  pip install soundfile pydub
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data as data
from pydub import AudioSegment

# ══════════════════════════════════════════════════════════════
# 25 种噪声中涉及音频的部分
# ══════════════════════════════════════════════════════════════

# 需要音频污染的噪声集合
NEED_AUDIO_CORRUPTION = {
    # 纯音频噪声
    "A_gaussian_noise", "A_traffic", "A_crowd",
    "A_rain", "A_thunder", "A_wind",
    # 视听联合噪声（也需要音频部分）
    "VA_gaussian", "VA_rain",
}

# 缺失音频（生成静音）
MISSING_AUDIO = {"Missing_audio"}

# 噪声 → 天气音频文件名（不含后缀），需与 weather/ 目录中的文件名一致
# 注意：VA_gaussian 和 VA_rain 不在这里！它们各自有独立目录（va_gaussian/、va_rain/）
NOISE_TO_WEATHER = {
    "A_traffic":   "traffic",
    "A_crowd":     "crowd",
    "A_rain":      "rain",
    "A_thunder":   "thunder",
    "A_wind":      "wind",
}

# 高斯噪声标准差（severity 1-5）
GAUSSIAN_NOISE_STD = [0.08, 0.12, 0.18, 0.26, 0.38]


# ══════════════════════════════════════════════════════════════
# 音频污染函数
# ══════════════════════════════════════════════════════════════

def apply_audio_corruption(audio_file: str, noise: str,
                           severity: int,
                           weather_dir: str,
                           output_file: str):
    """
    对单个音频文件应用污染。

    Args:
        audio_file:   原始音频路径（.wav）
        noise:        噪声类型（如 "A_gaussian_noise", "VA_rain" 等）
        severity:     污染强度 1-5
        weather_dir:  环境音目录（含 traffic.wav, rain.wav 等）
        output_file:  输出路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if noise == "A_gaussian_noise" or noise == "VA_gaussian":
        _corrupt_gaussian(audio_file, severity, output_file)
    elif noise == "VA_rain":
        weather_file = os.path.join(weather_dir, "rain.wav")
        _corrupt_mix(audio_file, weather_file, severity, output_file, is_rain=True)
    elif noise in NOISE_TO_WEATHER:
        weather_file = os.path.join(
            weather_dir, NOISE_TO_WEATHER[noise] + ".wav")
        _corrupt_mix(audio_file, weather_file, severity, output_file,
                     is_rain=(noise == "A_rain"))
    elif noise == "Missing_audio":
        _corrupt_missing(audio_file, output_file)
    else:
        raise ValueError(f"不支持的音频噪声类型: {noise}")


def _corrupt_gaussian(audio_file: str, severity: int, output_file: str):
    """叠加高斯白噪声"""
    audio, sr = sf.read(audio_file)
    noise_std = GAUSSIAN_NOISE_STD[severity - 1]
    noise = np.random.normal(0, noise_std, len(audio))
    audio_noisy = audio + noise
    sf.write(output_file, audio_noisy, sr)


def _corrupt_mix(audio_file: str, weather_file: str,
                 severity: int, output_file: str,
                 is_rain: bool = False):
    """将环境音混入原始音频"""
    audio = AudioSegment.from_file(audio_file)
    weather = AudioSegment.from_file(weather_file)

    # 长度对齐
    if len(audio) <= len(weather):
        weather = weather[:len(audio)]
    else:
        repeats = len(audio) // len(weather) + 1
        weather = weather * repeats
        weather = weather[:len(audio)]

    # 增益（强度越高，环境音越大）
    scale = [1, 2, 4, 6, 8][severity - 1]
    weather = weather.apply_gain(scale)

    # 混合
    mixed = audio.overlay(weather)
    mixed.export(output_file, format="wav")


def _corrupt_missing(audio_file: str, output_file: str):
    """生成静音（与原音频等长）"""
    audio, sr = sf.read(audio_file)
    silent = np.zeros_like(audio)
    sf.write(output_file, silent, sr)


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class CorruptAudioDataset(data.Dataset):
    """只处理需要音频污染的样本"""

    def __init__(self, audio_dir: str,
                 noise_assignment_csv: str,
                 output_dir: str,
                 weather_dir: str,
                 severity: int,
                 audio_suffix: str = ".wav"):
        super().__init__()

        self.audio_dir = audio_dir
        self.severity = severity
        self.output_dir = output_dir
        self.weather_dir = weather_dir
        self.audio_suffix = audio_suffix

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

        # 只保留需要音频污染的样本
        mask = df["assigned_noise"].isin(NEED_AUDIO_CORRUPTION | MISSING_AUDIO)
        self.df = df[mask].copy().reset_index(drop=True)
        print(f"[Dataset] 共 {len(self.df)} 个样本需要音频污染 "
              f"（占总 {len(df)} 个中的 {len(self.df)/len(df)*100:.1f}%）")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = row["sample_id"]
        noise = row["assigned_noise"]

        audio_file = os.path.join(self.audio_dir, vid + self.audio_suffix)

        # 输出路径
        if noise == "VA_gaussian":
            subdir = "va_gaussian"
        elif noise == "VA_rain":
            subdir = "va_rain"
        elif noise in NOISE_TO_WEATHER:
            subdir = NOISE_TO_WEATHER[noise]
        elif noise == "Missing_audio":
            subdir = "missing_audio"
        else:
            subdir = noise.replace("A_", "").lower()

        output_file = os.path.join(
            self.output_dir, subdir,
            f"severity_{self.severity}", vid + self.audio_suffix)

        # 如果文件已存在则跳过（节省时间）
        if os.path.exists(output_file):
            return 0

        try:
            apply_audio_corruption(
                audio_file=audio_file,
                noise=noise,
                severity=self.severity,
                weather_dir=self.weather_dir,
                output_file=output_file,
            )
        except Exception as e:
            print(f"[错误] 处理 {vid} 时出错: {e}")
            import traceback
            traceback.print_exc()

        return 0


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据噪声分配结果生成污染音频")
    parser.add_argument("--noise-assignment-csv", type=str, required=True,
                        help="noise_assignment.py 输出的 CSV")
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="原始音频目录（clean）")
    parser.add_argument("--weather-path", type=str, required=True,
                        help="环境音目录（含 traffic.wav, rain.wav 等）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="污染音频输出目录")
    parser.add_argument("--severity", type=int, default=3,
                        choices=[1, 2, 3, 4, 5],
                        help="污染严重程度（1-5）")
    parser.add_argument("--audio-suffix", type=str, default=".wav",
                        help="音频文件后缀")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="DataLoader batch_size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader num_workers")

    args = parser.parse_args()

    dataset = CorruptAudioDataset(
        audio_dir=args.audio_dir,
        noise_assignment_csv=args.noise_assignment_csv,
        output_dir=args.output_dir,
        weather_dir=args.weather_path,
        severity=args.severity,
        audio_suffix=args.audio_suffix,
    )

    loader = data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    print(f"\n开始生成污染音频（severity={args.severity}）...")
    total = len(dataset)
    t0 = time.time()

    for i, _ in enumerate(loader):
        done = min((i + 1) * args.batch_size, total)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        print(f"  进度: {done}/{total} ({done/total*100:.1f}%)  "
              f"速度: {speed:.0f} audio/s  ETA: {eta:.0f}s")

    print(f"\n[完成] 污染音频已保存到: {args.output_dir}")
