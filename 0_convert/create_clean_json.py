#!/usr/bin/env python3
"""
create_clean_json.py
────────────────────────────────────────────────────────────────
为干净数据（无噪声）生成索引 JSON 文件。

支持数据集：VGGSound、Kinetics50

使用方式：
  # VGGSound
  python create_clean_json.py \
      --dataset vggsound \
      --video-id-file /path/to/vggsound_test.csv \
      --audio-dir /path/to/audio_test \
      --video-frame-dir /path/to/image_mulframe_test \
      --output-dir ./json_output

  # Kinetics50
  python create_clean_json.py \
      --dataset kinetics50 \
      --video-id-file /path/to/kinetics50_val.csv \
      --audio-dir /path/to/audio_val256_k=50 \
      --video-frame-dir /path/to/image_mulframe_val256_k=50 \
      --output-dir ./json_output
"""

import os
import json
import argparse
import pandas as pd


# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

    # 数据集 → 帧数量
FRAME_COUNTS = {
    "vggsound": 10,
    "kinetics50": 10,
}

# 数据集 → 音频文件名后缀（VGGSound 有 _N 后缀，Kinetics50 没有）
AUDIO_SUFFIX = {
    "vggsound": "",       # 保持空字符串，CSV 中 video_id 本身就包含完整文件名
    "kinetics50": ".wav",
}

# 数据集 → 视频帧文件名后缀（.jpg）
VIDEO_SUFFIX = {
    "vggsound": ".jpg",
    "kinetics50": ".jpg",
}


def build_clean_json(dataset: str,
                     video_id_file: str,
                     audio_dir: str,
                     video_frame_dir: str,
                     output_dir: str,
                     severity: int = 0):
    """
    读取样本列表，生成 clean JSON。

    Args:
        dataset:         "vggsound" 或 "kinetics50"
        video_id_file:   CSV 文件路径，需包含 video_id 列（或用 --video-id-col 指定列名）
        audio_dir:       原始音频目录
        video_frame_dir: 原始视频帧目录（含 frame_0 ~ frame_9 子目录）
        output_dir:      输出 JSON 的目录（会自动创建）
        severity:        0 表示 clean，1-5 表示 corrupt severity 等级
    """
    if dataset not in FRAME_COUNTS:
        raise ValueError(f"未知数据集: {dataset}，可选: {list(FRAME_COUNTS.keys())}")

    frame_count = FRAME_COUNTS[dataset]
    audio_suffix = AUDIO_SUFFIX[dataset]

    # ── 1. 读取样本列表 ───────────────────────────────────────
    df = pd.read_csv(video_id_file)
    # 自动识别 video_id 列名（支持多种常见列名）
    vid_col = None
    for col in ["video_id", "videoId", "videoid", "id"]:
        if col in df.columns:
            vid_col = col
            break
    if vid_col is None:
        # 用第一列
        vid_col = df.columns[0]
        print(f"[警告] 未找到 video_id 列，使用第一列: '{vid_col}'")

    video_ids = df[vid_col].astype(str).tolist()
    print(f"[1/3] 读取到 {len(video_ids)} 个样本 (dataset={dataset})")

    # ── 2. 构造 JSON ───────────────────────────────────────────
    data_list = []
    missing_audio = 0
    missing_video = 0

    for vid in video_ids:
        audio_path = os.path.join(audio_dir, vid + audio_suffix)
        video_path = os.path.join(video_frame_dir, "frame_0")

        entry = {
            "video_id": vid,
            "wav": audio_path,
            "video_path": video_path,
            "frame_count": frame_count,
            "corruption": "clean",
            "severity": severity,
        }
        data_list.append(entry)

        # 检查文件是否存在
        if not os.path.exists(audio_path):
            missing_audio += 1
            if missing_audio <= 3:
                print(f"[警告] 音频文件不存在: {audio_path}")

    if missing_audio > 0:
        print(f"[警告] 共 {missing_audio}/{len(video_ids)} 个音频文件不存在")

    # ── 3. 保存 ───────────────────────────────────────────────
    subdir = "clean" if severity == 0 else f"corrupt/severity_{severity}"
    out_subdir = os.path.join(output_dir, subdir)
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"severity_{severity}.json")

    result = {"dataset": dataset, "data": data_list}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    print(f"[3/3] 已保存: {out_path}（共 {len(data_list)} 条记录）")
    return out_path


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 clean 数据的索引 JSON")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["vggsound", "kinetics50"],
                        help="数据集名称")
    parser.add_argument("--video-id-file", type=str, required=True,
                        help="CSV 文件路径，需包含 video_id 列")
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="原始音频目录")
    parser.add_argument("--video-frame-dir", type=str, required=True,
                        help="原始视频帧目录（含 frame_0 ~ frame_9 子目录）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出 JSON 的目录")
    parser.add_argument("--severity", type=int, default=0,
                        help="severity 等级（0=clean）")

    args = parser.parse_args()
    build_clean_json(
        dataset=args.dataset,
        video_id_file=args.video_id_file,
        audio_dir=args.audio_dir,
        video_frame_dir=args.video_frame_dir,
        output_dir=args.output_dir,
        severity=args.severity,
    )
