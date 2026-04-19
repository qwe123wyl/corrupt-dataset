"""
corrupt-dataset 运行主控脚本
用法：python run.py [选项]

功能：
  Step 0 - 将 refer.json 转换为 sample_list.csv
  Step 1 - 根据噪声兼容性表分配噪声
  Step 2 - 生成污染视频帧
  Step 3 - 生成污染音频
  Step 4 - 生成 clean 数据索引 JSON（训练用）
  Step 5 - 生成 corrupt 数据索引 JSON（训练用）

示例：
  # 只跑 KS50：
  python run.py --dataset ks50 --step 0
  python run.py --dataset ks50 --step 1
  python run.py --dataset ks50 --step 2 --severity 3 --workers 4
  python run.py --dataset ks50 --step 3 --severity 3 --workers 4
  python run.py --dataset ks50 --step 4 --severity 3
  python run.py --dataset ks50 --step 5 --severity 3

  # 或者一次性跑完：
  python run.py --dataset ks50 --from-step 0 --to-step 5 --severity 3
"""

import os
import re
import sys
import subprocess
import argparse
import config   # 统一路径管理


def step0(dataset_key):
    """Step 0: refer.json → sample_list.csv"""
    cfg = config.DATASETS[dataset_key]
    os.makedirs(config.SAMPLE_LISTS_DIR, exist_ok=True)

    if os.path.exists(cfg["sample_list"]):
        print(f"[Step 0] {cfg['name']} 的样本列表已存在，跳过。")
        return

    if not os.path.exists(cfg["refer_json"]):
        print(f"[错误] refer.json 不存在：{cfg['refer_json']}")
        sys.exit(1)

    print(f"[Step 0] 将 refer.json 转换为 CSV：{cfg['name']}")
    cmd = [
        sys.executable,
        os.path.join(config.BASE_DIR, "0_convert", "convert_refer_to_csv.py"),
        "--json-path",   cfg["refer_json"],
        "--output-csv",  cfg["sample_list"],
    ]
    if "class_csv" in cfg and cfg["class_csv"]:
        cmd += ["--class-csv", cfg["class_csv"]]
    result = subprocess.run(cmd, check=True)
    print(f"[Step 0] 完成：{cfg['sample_list']}")


def step1(dataset_key):
    """Step 1: 分配噪声"""
    cfg = config.DATASETS[dataset_key]
    print(f"[Step 1] 为 {cfg['name']} 分配噪声...")

    script = os.path.join(config.BASE_DIR, "1_noise_assignment", "noise_assignment.py")
    result = subprocess.run([
        sys.executable, script,
        "--xlsx-path",  config.NOISE_COMPAT,
        "--input-csv",  cfg["sample_list"],
        "--output-csv", cfg["noise_csv"],
        "--sheet-name", cfg["sheet_name"],
    ], check=True)
    print(f"[Step 1] 完成：{cfg['noise_csv']}")


def step2(dataset_key, severity, workers):
    """Step 2: 生成污染视频帧"""
    cfg = config.DATASETS[dataset_key]
    print(f"[Step 2] 为 {cfg['name']} 生成污染视频帧...")

    script = os.path.join(config.BASE_DIR, "2_corruption", "make_c_video.py")
    os.makedirs(cfg["video_out"], exist_ok=True)
    result = subprocess.run([
        sys.executable, script,
        "--noise-assignment-csv", cfg["noise_csv"],
        "--video-frame-dir",      cfg["video_dir"],
        "--output-dir",           cfg["video_out"],
        "--severity",             str(severity),
        "--num-workers",          str(workers),
    ], check=True)
    print(f"[Step 2] 完成：{cfg['video_out']}")


def step3(dataset_key, severity, workers):
    """Step 3: 生成污染音频"""
    cfg = config.DATASETS[dataset_key]
    print(f"[Step 3] 为 {cfg['name']} 生成污染音频...")

    script = os.path.join(config.BASE_DIR, "2_corruption", "make_c_audio.py")
    os.makedirs(cfg["audio_out"], exist_ok=True)
    result = subprocess.run([
        sys.executable, script,
        "--noise-assignment-csv", cfg["noise_csv"],
        "--audio-dir",            cfg["audio_dir"],
        "--weather-path",         config.EXTERNAL_DATA["weather"],
        "--output-dir",           cfg["audio_out"],
        "--severity",             str(severity),
        "--num-workers",          str(workers),
    ], check=True)
    print(f"[Step 3] 完成：{cfg['audio_out']}")


# ── 数据集名称映射（create_clean/corrupt_json.py 用）────────────
_DATASET_ALIAS = {
    "ks50":       "kinetics50",
    "ks50_train": "kinetics50",
    "vgg":         "vggsound",
}


def step4(dataset_key, severity):
    """Step 4: 生成 clean 数据索引 JSON（训练用，含数字 labels）"""
    cfg = config.DATASETS[dataset_key]
    dataset_alias = _DATASET_ALIAS[dataset_key]
    print(f"[Step 4] 为 {cfg['name']} 生成 clean JSON...")

    import pandas as pd
    import json

    # 读取 class name → index 映射
    class_csv = cfg["class_csv"]
    if not os.path.exists(class_csv):
        print(f"[警告] class label 文件不存在，跳过 labels 注入：{class_csv}")
        # fallback：直接调用 create_clean_json.py
        script = os.path.join(config.BASE_DIR, "0_convert", "create_clean_json.py")
        subprocess.run([
            sys.executable, script,
            "--dataset",         dataset_alias,
            "--video-id-file",   cfg["sample_list"],
            "--audio-dir",       cfg["audio_dir"],
            "--video-frame-dir", cfg["video_dir"],
            "--output-dir",      config.OUTPUT_DIR,
            "--severity",        str(severity),
        ], check=True)
        return

    class_df = pd.read_csv(class_csv)
    # 支持多种列名
    idx_col = next((c for c in class_df.columns if c.lower() in ("index", "id", "idx")), class_df.columns[0])
    name_col = next((c for c in class_df.columns if c.lower() in ("mid", "name", "label", "class_name")), None)
    if name_col is None:
        for c in class_df.columns:
            if c != idx_col:
                name_col = c
                break
    class_df = class_df.rename(columns={idx_col: "idx", name_col: "name"})
    class_df["name"] = class_df["name"].astype(str)
    name_to_idx = dict(zip(class_df["name"], class_df["idx"]))

    # 读取 sample list
    sample_df = pd.read_csv(cfg["sample_list"])
    vid_col = next((c for c in sample_df.columns if c.lower() in ("sample_id", "video_id", "videoid")), None)
    label_col = next((c for c in sample_df.columns if c.lower() in ("class_name", "labels", "label")), None)
    if vid_col is None:
        vid_col = sample_df.columns[0]
    if label_col is None:
        label_col = sample_df.columns[1]

    sample_df[vid_col] = sample_df[vid_col].astype(str)
    sample_df[label_col] = sample_df[label_col].astype(str)

    audio_suffix = "" if dataset_alias == "vggsound" else ".wav"

    entries = []
    for _, row in sample_df.iterrows():
        vid = str(row[vid_col])
        class_name = str(row[label_col])
        label_idx = name_to_idx.get(class_name)
        if label_idx is None:
            print(f"[警告] 类别 '{class_name}' 不在映射表中，跳过")
            continue
        entries.append({
            "video_id":     vid,
            "wav":          os.path.join(cfg["audio_dir"], vid + audio_suffix),
            "video_path":   os.path.join(cfg["video_dir"], "frame_0"),
            "frame_count":  10,
            "corruption":   "clean",
            "severity":     severity,
            "labels":       label_idx,
            "class_name":   class_name,
        })

    out_dir = os.path.join(config.OUTPUT_DIR, "clean", f"severity_{severity}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"severity_{severity}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_alias, "data": entries}, f, indent=1, ensure_ascii=False)

    print(f"[Step 4] 完成：{out_path}（{len(entries)} 条记录）")


def step5(dataset_key, severity):
    """Step 5: 生成 corrupt 数据索引 JSON（训练用）"""
    cfg = config.DATASETS[dataset_key]
    dataset_alias = _DATASET_ALIAS[dataset_key]
    print(f"[Step 5] 为 {cfg['name']} 生成 corrupt JSON...")

    # clean_json 输出路径来自 create_clean_json.py，固定为 _output/clean/severity_N/severity_N.json
    clean_json = os.path.join(
        config.OUTPUT_DIR, "clean", f"severity_{severity}", f"severity_{severity}.json")

    if not os.path.exists(clean_json):
        print(f"[错误] clean JSON 不存在，请先运行 Step 4：{clean_json}")
        sys.exit(1)

    script = os.path.join(config.BASE_DIR, "0_convert", "create_corrupted_json.py")
    result = subprocess.run([
        sys.executable, script,
        "--dataset",              dataset_alias,
        "--noise-assignment-csv", cfg["noise_csv"],
        "--clean-json",           clean_json,
        "--audio-c-dir",          cfg["audio_out"],
        "--video-c-dir",          cfg["video_out"],
        "--output-dir",           config.OUTPUT_DIR,
        "--severity",             str(severity),
    ], check=True)
    print(f"[Step 5] 完成：{os.path.join(config.OUTPUT_DIR, 'corrupt', f'severity_{severity}', f'severity_{severity}.json')}")


def main():
    parser = argparse.ArgumentParser(description="corrupt-dataset 污染数据生成流水线")
    parser.add_argument("--dataset", type=str, default="ks50",
                        choices=list(config.DATASETS.keys()),
                        help="选择数据集")
    parser.add_argument("--step", type=int, choices=[0, 1, 2, 3, 4, 5],
                        help="只运行指定 step")
    parser.add_argument("--from-step", type=int, default=0,
                        help="从哪个 step 开始（包含）")
    parser.add_argument("--to-step", type=int, default=5,
                        help="到哪个 step 结束（包含）")
    parser.add_argument("--severity", type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help="污染严重程度（1-5）")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行 worker 数")
    args = parser.parse_args()

    print("=" * 60)
    print(f"数据集：{config.DATASETS[args.dataset]['name']}")
    print(f"严重程度：{args.severity}")
    print(f"Workers：{args.workers}")
    print("=" * 60)

    steps = range(args.from_step, args.to_step + 1)
    if args.step is not None:
        steps = [args.step]

    for s in steps:
        if s == 0:
            step0(args.dataset)
        elif s == 1:
            step1(args.dataset)
        elif s == 2:
            step2(args.dataset, args.severity, args.workers)
        elif s == 3:
            step3(args.dataset, args.severity, args.workers)
        elif s == 4:
            step4(args.dataset, args.severity)
        elif s == 5:
            step5(args.dataset, args.severity)

    print("=" * 60)
    print("全部完成！")


if __name__ == "__main__":
    main()
