#!/usr/bin/env python3
"""
create_corrupted_json.py
────────────────────────────────────────────────────────────────
读取 noise_assignment.py 的输出 CSV（包含 sample_id, class_name, assigned_noise），
根据 assigned_noise 字段生成污染数据的索引 JSON。

使用方式：
  python create_corrupted_json.py \
      --dataset vggsound \
      --noise-assignment-csv /path/to/dataset_with_noise.csv \
      --clean-json /path/to/clean/severity_0.json \
      --audio-c-dir /path/to/audio_test-C \
      --video-c-dir /path/to/image_mulframe_test-C \
      --output-dir ./json_output

JSON 输出格式（与 clean JSON 兼容）：
  {
    "dataset": "vggsound",
    "data": [
      {
        "video_id": "xxx",
        "wav": "/path/to/corrupted_audio/xxx.wav",
        "video_path": "/path/to/corrupted_frames/frame_0",
        "frame_count": 10,
        "corruption": "V_gaussian_noise",
        "severity": 3,
        "labels": [...]
      },
      ...
    ]
  }
"""

import os
import json
import argparse
import pandas as pd


# ══════════════════════════════════════════════════════════════
# 25 种噪声 → 路径映射规则
# ══════════════════════════════════════════════════════════════

# 视频噪声（需要遍历 frame_0 ~ frame_9）
VIDEO_NOISES = {
    "V_gaussian_noise":    "gaussian_noise",
    "V_shot_noise":        "shot_noise",
    "V_impulse_noise":     "impulse_noise",
    "V_defocus_blur":      "defocus_blur",
    "V_glass_blur":        "glass_blur",
    "V_motion_blur":       "motion_blur",
    "V_zoom_blur":         "zoom_blur",
    "V_snow":              "snow",
    "V_frost":             "frost",
    "V_fog":               "fog",
    "V_brightness":        "brightness",
    "V_contrast":          "contrast",
    "V_elastic_transform": "elastic_transform",
    "V_pixelate":          "pixelate",
    "V_jpeg_compression":  "jpeg_compression",
}

# 音频噪声（混合进音频）
AUDIO_NOISES = {
    "A_gaussian_noise": "gaussian_noise",
    "A_traffic":         "traffic",
    "A_crowd":           "crowd",
    "A_rain":            "rain",
    "A_thunder":         "thunder",
    "A_wind":            "wind",
}

# 视听联合噪声（同时需要视频和音频）
VA_NOISES = {
    "VA_gaussian": "gaussian",
    "VA_rain":     "rain",
}

# 缺失模态噪声（路径指向 clean 数据，但标记为缺失）
MISSING_NOISES = {
    "Missing_audio": "missing_audio",
    "Missing_video": "missing_video",
}


def classify_noise(noise_name: str) -> str:
    """返回噪声类型：video / audio / va / missing"""
    if noise_name in VIDEO_NOISES:
        return "video"
    elif noise_name in AUDIO_NOISES:
        return "audio"
    elif noise_name in VA_NOISES:
        return "va"
    elif noise_name in MISSING_NOISES:
        return "missing"
    else:
        raise ValueError(f"未知噪声类型: {noise_name}")


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════

def build_corrupted_json(dataset: str,
                         noise_assignment_csv: str,
                         clean_json: str,
                         audio_c_dir: str,
                         video_c_dir: str,
                         output_dir: str,
                         severity: int = 3,
                         by_noise_type: bool = False):
    """
    根据噪声分配结果生成污染数据 JSON。

    Args:
        dataset:             "vggsound" 或 "kinetics50"
        noise_assignment_csv: noise_assignment.py 输出的 CSV（含 sample_id, class_name, assigned_noise）
        clean_json:          干净数据的 JSON（用于获取 labels 等元信息）
        audio_c_dir:         污染音频根目录
        video_c_dir:         污染视频帧根目录
        output_dir:          输出 JSON 的目录
        severity:            污染严重程度（1-5），默认 3
        by_noise_type:       True=按噪声类型分别生成 JSON，False=合并成一个 JSON
    """
    # ── 1. 读取噪声分配结果 ───────────────────────────────────
    df = pd.read_csv(noise_assignment_csv)
    # 自动识别列名
    for col in ["sample_id", "class_name", "assigned_noise", "video_id"]:
        if col not in df.columns:
            for alias in ["video_id", "VideoId", "id"]:
                if alias in df.columns:
                    df = df.rename(columns={alias: col})
                    break

    # sample_id 可能是字符串或整数
    df["sample_id"] = df["sample_id"].astype(str)
    print(f"[1/4] 读取噪声分配结果: {len(df)} 条记录")
    print(f"      CSV 列: {df.columns.tolist()}")
    print(f"      噪声类型分布:\n{df['assigned_noise'].value_counts().to_string()}")

    # ── 2. 读取 clean JSON 获取 labels ───────────────────────
    with open(clean_json, "r", encoding="utf-8") as f:
        clean_data = json.load(f)

    # 建立 video_id → labels 映射
    vid_to_labels = {}
    for item in clean_data.get("data", []):
        vid_to_labels[str(item["video_id"])] = item.get("labels")

    vid_to_wav_clean = {}
    vid_to_video_path_clean = {}
    for item in clean_data.get("data", []):
        vid_to_wav_clean[str(item["video_id"])] = item.get("wav", "")
        vid_to_video_path_clean[str(item["video_id"])] = item.get("video_path", "")

    # ── 3. 构建 JSON 条目 ────────────────────────────────────
    entries = []
    unknown_noise_count = 0

    for _, row in df.iterrows():
        vid = str(row["sample_id"])
        noise = str(row["assigned_noise"])
        labels = vid_to_labels.get(vid)
        wav_clean = vid_to_wav_clean.get(vid, "")
        video_path_clean = vid_to_video_path_clean.get(vid, "")

        noise_type = classify_noise(noise)
        corruption_dir_name = None
        wav_path = ""
        video_path = ""

        if noise_type == "video":
            corruption_dir_name = VIDEO_NOISES[noise]
            # 视频帧：video_c_dir/corruption/severity_N/frame_X/
            video_path = os.path.join(
                video_c_dir, corruption_dir_name,
                f"severity_{severity}", "frame_0"
            )
            # 音频用 clean
            wav_path = wav_clean

        elif noise_type == "audio":
            corruption_dir_name = AUDIO_NOISES[noise]
            # 音频：audio_c_dir/corruption/severity_N/vid.wav
            wav_path = os.path.join(
                audio_c_dir, corruption_dir_name,
                f"severity_{severity}", vid + ".wav"
            )
            # 视频用 clean
            video_path = video_path_clean

        elif noise_type == "va":
            corruption_dir_name = VA_NOISES[noise]
            # 视听联合噪声
            if "gaussian" in noise.lower():
                video_corruption = "gaussian_noise"
            elif "rain" in noise.lower():
                video_corruption = "rain"
            else:
                video_corruption = noise

            video_path = os.path.join(
                video_c_dir, video_corruption,
                f"severity_{severity}", "frame_0"
            )
            wav_path = os.path.join(
                audio_c_dir, corruption_dir_name,
                f"severity_{severity}", vid + ".wav"
            )

        elif noise_type == "missing":
            if noise == "Missing_audio":
                wav_path = ""          # 音频缺失，路径为空
                video_path = video_path_clean
            else:  # Missing_video
                wav_path = wav_clean
                video_path = ""        # 视频缺失，路径为空

        # 检查是否已知噪声
        if noise not in VIDEO_NOISES and \
           noise not in AUDIO_NOISES and \
           noise not in VA_NOISES and \
           noise not in MISSING_NOISES:
            unknown_noise_count += 1
            if unknown_noise_count <= 3:
                print(f"[警告] 未知噪声类型: '{noise}'，跳过")
            continue

        entry = {
            "video_id": vid,
            "wav": wav_path,
            "video_path": video_path,
            "frame_count": 10,
            "corruption": noise,
            "corruption_dir": corruption_dir_name or noise,
            "severity": severity,
            "class_name": str(row.get("class_name", "")),
        }
        if labels is not None:
            entry["labels"] = labels

        entries.append(entry)

    if unknown_noise_count > 0:
        print(f"[警告] 共 {unknown_noise_count} 条记录包含未知噪声类型")

    print(f"[2/4] 构建了 {len(entries)} 条 JSON 条目")

    # ── 4. 按需求分别输出 ─────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    if by_noise_type:
        # 按噪声类型分别生成 JSON（每个噪声一个文件）
        noise_to_entries = {}
        for e in entries:
            c = e["corruption"]
            noise_to_entries.setdefault(c, []).append(e)

        for noise_name, sub_entries in sorted(noise_to_entries.items()):
            subdir = os.path.join(output_dir, "by_noise_type", noise_name)
            os.makedirs(subdir, exist_ok=True)
            out_path = os.path.join(subdir, f"severity_{severity}.json")
            result = {"dataset": dataset, "data": sub_entries, "corruption": noise_name}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=1, ensure_ascii=False)
            print(f"  已保存 [{noise_name}]: {out_path} ({len(sub_entries)} 条)")
    else:
        # 合并成一个 JSON（所有噪声混合，随机顺序）
        import numpy as np
        rng = np.random.default_rng(42)
        indices = np.arange(len(entries))
        rng.shuffle(indices)
        shuffled = [entries[i] for i in indices]

        out_path = os.path.join(output_dir, f"corrupt/severity_{severity}", f"severity_{severity}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result = {"dataset": dataset, "data": shuffled, "corruption": "mixed"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        print(f"[4/4] 已保存（混合噪声）: {out_path} ({len(shuffled)} 条)")

    print("\n[完成] JSON 生成完毕")
    return entries


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据噪声分配结果生成污染数据索引 JSON")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["vggsound", "kinetics50"],
                        help="数据集名称")
    parser.add_argument("--noise-assignment-csv", type=str, required=True,
                        help="noise_assignment.py 输出的 CSV")
    parser.add_argument("--clean-json", type=str, required=True,
                        help="干净数据的 JSON（用于获取 labels）")
    parser.add_argument("--audio-c-dir", type=str, required=True,
                        help="污染音频根目录")
    parser.add_argument("--video-c-dir", type=str, required=True,
                        help="污染视频帧根目录")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出 JSON 的目录")
    parser.add_argument("--severity", type=int, default=3,
                        choices=[1, 2, 3, 4, 5],
                        help="污染严重程度（1-5），默认 3")
    parser.add_argument("--by-noise-type", action="store_true",
                        help="按噪声类型分别生成 JSON（每个噪声一个文件）")

    args = parser.parse_args()
    build_corrupted_json(
        dataset=args.dataset,
        noise_assignment_csv=args.noise_assignment_csv,
        clean_json=args.clean_json,
        audio_c_dir=args.audio_c_dir,
        video_c_dir=args.video_c_dir,
        output_dir=args.output_dir,
        severity=args.severity,
        by_noise_type=args.by_noise_type,
    )
