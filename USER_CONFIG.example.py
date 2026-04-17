# USER_CONFIG.example.py
# ─────────────────────────────────────────────────────────────
# 复制本文件为 USER_CONFIG.py，然后修改路径为本地实际路径。
# config.py 会自动加载本文件。
# ─────────────────────────────────────────────────────────────
#
# 需要修改的路径（必填）：
#   1. Kinetics50 数据集目录（视频帧 + 音频）
#   2. VGGSound 数据集目录（视频帧 + 音频）
#   3. 环境音目录（5 个 .wav 文件）
#
# 路径要求：
#   - Kinetics50 需要：image_mulframe_val256_k=50/   （frame_0~9 子目录）
#                       audio_val256_k=50/             （.wav 文件）
#   - VGGSound 需要：  image_mulframe_test/            （frame_0~9 子目录）
#                      audio_test/                     （.wav 文件）
#   - 环境音目录需要：  traffic.wav, crowd.wav, rain.wav,
#                      thunder.wav, wind.wav

EXTERNAL_DATA = {
    # ── Kinetics-Sound 50 ───────────────────────────────────
    "ks50_video":   "/data/wyl/TTA/corrupt-dataset/dataset/Kinetics50/image_mulframe_val256_k=50",
    "ks50_audio":   "/data/wyl/TTA/corrupt-dataset/dataset/Kinetics50/audio_val256_k=50",
    # （KS50 训练集）
    "ks50_train_video": "/data/wyl/TTA/corrupt-dataset/dataset/Kinetics50/image_mulframe_train256_k=50",
    "ks50_train_audio": "/data/wyl/TTA/corrupt-dataset/dataset/Kinetics50/audio_train256_k=50",

    # ── VGGSound ─────────────────────────────────────────────
    "vgg_video":    "/data/wyl/TTA/corrupt-dataset/dataset/VGGSound/image_mulframe_test",
    "vgg_audio":    "/data/wyl/TTA/corrupt-dataset/dataset/VGGSound/audio_test",

    # ── 环境音（叠加到音频噪声）───────────────────────────────
    # 需要以下 5 个 .wav 文件：
    #   traffic.wav, crowd.wav, rain.wav, thunder.wav, wind.wav
    "weather":      "/data/wyl/dataset/NoisyAudios",
}
