"""
将 refer.json 转换为 sample_id,class_name 的 CSV
"""
import json
import csv
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, required=True,
                        help="refer.json 路径")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="输出 CSV 路径")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "class_name"])
        for item in data["data"]:
            writer.writerow([item["video_id"], item["labels"]])

    print(f"转换完成，共 {len(data['data'])} 条记录 -> {args.output_csv}")
