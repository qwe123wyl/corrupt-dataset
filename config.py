"""
corrupt-dataset 全局配置文件
所有路径统一在此管理，避免硬编码分散在各脚本中。

用户需要：
  1. 复制 USER_CONFIG.example.py → USER_CONFIG.py
  2. 修改 USER_CONFIG.py 中的路径为本地实际路径
  3. 无需修改本文件或其他任何代码

如果 USER_CONFIG.py 不存在，尝试使用示例路径（有概率报错，提示用户配置）。
"""
import os

# ── 项目根目录 ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 用户自定义路径（从 USER_CONFIG.py 加载）────────────────────
_user_cfg_path = os.path.join(BASE_DIR, "USER_CONFIG.py")
if os.path.exists(_user_cfg_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_config", _user_cfg_path)
    _uc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_uc)
    EXTERNAL_DATA = _uc.EXTERNAL_DATA
    print(f"[config] 已加载用户配置：{_user_cfg_path}")
else:
    # 找不到 USER_CONFIG.py 时，打印警告并使用示例路径（仅供本机开发调试）
    print("[警告] USER_CONFIG.py 不存在，外部数据路径使用默认值。")
    print("[警告] 请复制 USER_CONFIG.example.py 并修改为本地路径。")
    print()
    EXTERNAL_DATA = {
        "ks50_video":         "/data/wyl/dataset/Kinetics50/image_mulframe_val256_k=50",
        "ks50_audio":         "/data/wyl/dataset/Kinetics50/audio_val256_k=50",
        "ks50_train_video":    "/data/wyl/dataset/Kinetics50/image_mulframe_train256_k=50",
        "ks50_train_audio":    "/data/wyl/dataset/Kinetics50/audio_train256_k=50",
        "vgg_video":           "/data/wyl/dataset/VGGSound/image_mulframe_test",
        "vgg_audio":           "/data/wyl/dataset/VGGSound/audio_test",
        # 环境音文件（5 个 .wav），必须与 make_corruptions/ 目录结构配合使用
        "weather":             "/data/wyl/dataset/NoisyAudios",
    }

# ── 项目内部路径（固定，相对于 BASE_DIR）──────────────────────
SAMPLE_LISTS_DIR = os.path.join(BASE_DIR, "_sample_lists")
OUTPUT_DIR       = os.path.join(BASE_DIR, "_output")
DATA_DIR         = os.path.join(BASE_DIR, "data")
NOISE_COMPAT     = os.path.join(BASE_DIR, "1_noise_assignment",
                                 "noise_compatibility_KS50_VGGSound.xlsx")

# refer.json（已随项目一起分发）
REFER_JSON = {
    "ks50_test":  os.path.join(DATA_DIR, "refer", "ks50_test_refer.json"),
    "ks50_train": os.path.join(DATA_DIR, "refer", "ks50_train_refer.json"),
    "vgg_test":   os.path.join(DATA_DIR, "refer", "vgg_test_refer.json"),
}

# class label → index 映射
CLASS_LABELS = {
    "ks50":  os.path.join(DATA_DIR, "class_labels", "class_labels_indices_ks50.csv"),
    "vgg":   os.path.join(DATA_DIR, "class_labels", "class_labels_indices_vgg.csv"),
}

# ── 各数据集配置 ──────────────────────────────────────────────
DATASETS = {
    "ks50": {
        "name":         "Kinetics-Sound 50（测试集）",
        "refer_json":   REFER_JSON["ks50_test"],
        "sample_list":  os.path.join(SAMPLE_LISTS_DIR, "ks50_sample_list.csv"),
        "noise_csv":    os.path.join(OUTPUT_DIR, "ks50_with_noise.csv"),
        "video_dir":    EXTERNAL_DATA["ks50_video"],
        "audio_dir":    EXTERNAL_DATA["ks50_audio"],
        "video_out":    os.path.join(OUTPUT_DIR, "ks50_video_frames_C"),
        "audio_out":    os.path.join(OUTPUT_DIR, "ks50_audio_C"),
        "clean_json":   os.path.join(OUTPUT_DIR, "ks50_clean.json"),
        "corrupt_json": os.path.join(OUTPUT_DIR, "ks50_corrupt.json"),
        "sheet_name":   "KS-50 (50 classes)",
        "class_csv":    CLASS_LABELS["ks50"],
    },
    "ks50_train": {
        "name":         "Kinetics-Sound 50（训练集）",
        "refer_json":   REFER_JSON["ks50_train"],
        "sample_list":  os.path.join(SAMPLE_LISTS_DIR, "ks50_train_sample_list.csv"),
        "noise_csv":    os.path.join(OUTPUT_DIR, "ks50_train_with_noise.csv"),
        "video_dir":    EXTERNAL_DATA["ks50_train_video"],
        "audio_dir":    EXTERNAL_DATA["ks50_train_audio"],
        "video_out":    os.path.join(OUTPUT_DIR, "ks50_train_video_frames_C"),
        "audio_out":    os.path.join(OUTPUT_DIR, "ks50_train_audio_C"),
        "clean_json":   os.path.join(OUTPUT_DIR, "ks50_train_clean.json"),
        "corrupt_json": os.path.join(OUTPUT_DIR, "ks50_train_corrupt.json"),
        "sheet_name":   "KS-50 (50 classes)",
        "class_csv":    CLASS_LABELS["ks50"],
    },
    "vgg": {
        "name":         "VGGSound（测试集）",
        "refer_json":   REFER_JSON["vgg_test"],
        "sample_list":  os.path.join(SAMPLE_LISTS_DIR, "vgg_sample_list.csv"),
        "noise_csv":    os.path.join(OUTPUT_DIR, "vgg_with_noise.csv"),
        "video_dir":    EXTERNAL_DATA["vgg_video"],
        "audio_dir":    EXTERNAL_DATA["vgg_audio"],
        "video_out":    os.path.join(OUTPUT_DIR, "vgg_video_frames_C"),
        "audio_out":    os.path.join(OUTPUT_DIR, "vgg_audio_C"),
        "clean_json":   os.path.join(OUTPUT_DIR, "vgg_clean.json"),
        "corrupt_json": os.path.join(OUTPUT_DIR, "vgg_corrupt.json"),
        "sheet_name":   "VGGSound (309 classes)",
        "class_csv":    CLASS_LABELS["vgg"],
    },
}
