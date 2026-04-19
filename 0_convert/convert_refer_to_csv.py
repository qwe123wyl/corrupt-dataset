"""
将 refer.json 转换为 sample_id,class_name 的 CSV
支持 VGGSound 的 vgg_X 数字 ID 自动转换为文字类别名。
"""
import json
import csv
import argparse
import os

# VGGSound 的 vgg_X → 文字名称映射（从 class_labels_indices_vgg.csv 加载）
_MID_TO_DISPLAY = {}


def _load_vgg_map(csv_path: str):
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _MID_TO_DISPLAY[row["mid"]] = row["display_name"]


def _convert_label(label: str) -> str:
    """将 vgg_X 格式的标签转换为文字名称，无映射时原样返回。"""
    return _MID_TO_DISPLAY.get(label, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, required=True,
                        help="refer.json 路径")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="输出 CSV 路径")
    parser.add_argument("--class-csv", type=str, default=None,
                        help="类别映射 CSV（如 class_labels_indices_vgg.csv），"
                             "存在时用于将 vgg_X 转换为文字类别名")
    args = parser.parse_args()

    if args.class_csv:
        _load_vgg_map(args.class_csv)

    with open(args.json_path, "r") as f:
        data = json.load(f)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "class_name"])
        for item in data["data"]:
            writer.writerow([item["video_id"], _convert_label(item["labels"])])

    print(f"转换完成，共 {len(data['data'])} 条记录 -> {args.output_csv}")
