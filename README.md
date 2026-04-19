# corrupt-dataset

音视频视听联合鲁棒性数据集生成流水线。给定 Kinetics-Sound 50 和 VGGSound 的原始视频帧 + 音频，遍历每个样本，根据类别兼容性表为其分配一种噪声，使用配额上限法保证 26 种噪声（× 5 级严重程度）均匀覆盖全量样本，最终输出污染后的视频帧、音频文件及 PyTorch 可直接加载的训练索引 JSON。

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置路径（复制示例配置文件并修改路径）
cp USER_CONFIG.example.py USER_CONFIG.py

# 一键运行完整流水线（severity=3, 4 workers）
python run.py --dataset ks50 --from-step 0 --to-step 5 --severity 3 --workers 4
```

---

## 数据集依赖

运行前需要准备以下外部数据集（自行下载，不在 git 内）：

| 数据集 | 目录结构要求 | 下载方式 |
|---|---|---|
| **Kinetics-Sound 50**（测试集） | `image_mulframe_val256_k=50/frame_0~9/*.jpg` + `audio_val256_k=50/*.wav` | Kinetics-400 官方 |
| **Kinetics-Sound 50**（训练集） | `image_mulframe_train256_k=50/frame_0~9/*.jpg` + `audio_train256_k=50/*.wav` | Kinetics-400 官方 |
| **VGGSound** | `image_mulframe_test/frame_0~9/*.jpg` + `audio_test/*.wav` | VGGSound 官方 |
| **环境音**（5 个 .wav） | `traffic.wav`, `crowd.wav`, `rain.wav`, `thunder.wav`, `wind.wav` | Audioset 或自采 |

在 `USER_CONFIG.py` 的 `EXTERNAL_DATA` 字典中填写各数据集的本地根目录路径。

---

## 流水线用法

### 支持的数据集

| `--dataset` | 说明 |
|---|---|
| `ks50` | Kinetics-Sound 50 测试集（50 类） |
| `ks50_train` | Kinetics-Sound 50 训练集（50 类） |
| `vgg` | VGGSound 测试集（309 类） |

### 严重程度

`--severity 1~5`，所有样本共用同一个 severity 等级，数值越大污染越强。不同噪声类型在各级别有不同的参数（噪声强度、模糊半径、叠加音量等）。

### 完整流水线

```bash
python run.py --dataset ks50 --from-step 0 --to-step 5 --severity 3 --workers 4
```

### 分步运行

```bash
python run.py --dataset ks50 --step 0
python run.py --dataset ks50 --step 1
python run.py --dataset ks50 --step 2 --severity 3 --workers 4
python run.py --dataset ks50 --step 3 --severity 3 --workers 4
python run.py --dataset ks50 --step 4 --severity 3
python run.py --dataset ks50 --step 5 --severity 3
```

---

## 流水线各步骤

| Step | 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| **Step 0** | `0_convert/convert_refer_to_csv.py` | `data/refer/*.json` | `_sample_lists/*.csv` | 将样本索引转换为 CSV |
| **Step 1** | `1_noise_assignment/noise_assignment.py` | `_sample_lists/*.csv` + Excel 兼容性表 | `_output/*_with_noise.csv` | 为每个样本分配一种兼容的噪声类型 |
| **Step 2** | `2_corruption/make_c_video.py` | `_output/*_with_noise.csv` + 视频帧 | `_output/ks50_video_frames_C/` | 生成污染视频帧（19 种：16 种 V_ + 2 种 VA_ + Missing_video） |
| **Step 3** | `2_corruption/make_c_audio.py` | `_output/*_with_noise.csv` + 音频 + 环境音 | `_output/ks50_audio_C/` | 生成污染音频（9 种：6 种 A_ + 2 种 VA_ + Missing_audio） |
| **Step 4** | `run.py`（内置） | `_sample_lists/*.csv` + `data/class_labels/*.csv` | `_output/clean/severity_N/severity_N.json` | 生成 clean 数据索引（含数字 label） |
| **Step 5** | `0_convert/create_corrupted_json.py` | Step 1-4 的输出 | `_output/corrupt/severity_N/severity_N.json` | 生成 corrupt 数据索引（替换文件路径为污染路径） |

---

## 26 种噪声类型

### 纯视频噪声（V_）

| 噪声 | 说明 |
|------|------|
| `V_gaussian_noise` | 高斯白噪声 |
| `V_shot_noise` | 散粒噪声（光子泊松噪声） |
| `V_impulse_noise` | 椒盐脉冲噪声 |
| `V_defocus_blur` | 散焦模糊 |
| `V_glass_blur` | 玻璃模糊 |
| `V_motion_blur` | 运动模糊 |
| `V_zoom_blur` | 缩放模糊 |
| `V_rain` | 下雨（叠加雨条纹噪声） |
| `V_snow` | 雪花遮挡 |
| `V_frost` | 霜冻遮挡（使用 `2_corruption/make_corruptions/frost*.png/.jpg` 素材） |
| `V_fog` | 雾气 |
| `V_brightness` | 亮度变化 |
| `V_contrast` | 对比度变化 |
| `V_elastic_transform` | 弹性形变 |
| `V_pixelate` | 像素化 |
| `V_jpeg_compression` | JPEG 压缩伪影 |

### 纯音频噪声（A_）

| 噪声 | 说明 |
|------|------|
| `A_gaussian_noise` | 高斯白噪声 |
| `A_traffic` | 交通噪声叠加 |
| `A_crowd` | 人群噪声叠加 |
| `A_rain` | 雨声叠加 |
| `A_thunder` | 雷声叠加 |
| `A_wind` | 风声叠加 |

### 视听联合噪声（VA_）

| 噪声 | 说明 |
|------|------|
| `VA_gaussian` | 视频 + 音频同时加高斯噪声 |
| `VA_rain` | 视频下雨 + 音频混入雨声 |

### 缺失模态（Missing_）

| 噪声 | 说明 |
|------|------|
| `Missing_audio` | 音频静音 |
| `Missing_video` | 视频全黑帧 |

---

## 输出 JSON 格式

`clean.json` 和 `corrupt.json` 结构相同，区别在于路径指向干净或污染后的文件：

```json
{
  "dataset": "kinetics50",
  "data": [
    {
      "video_id": "-LsQK3zOcwo",
      "wav": "/path/to/-LsQK3zOcwo.wav",
      "video_path": "/path/to/frame_0",
      "frame_count": 10,
      "corruption": "clean",
      "severity": 3,
      "labels": 0,
      "class_name": "pumping_fist"
    }
  ]
}
```

`corrupt.json` 中额外包含 `corruption_dir` 字段，标识噪声对应的子目录名。

---

## 目录结构

```
corrupt-dataset/
├── config.py                        # 全局配置（路径管理）
├── run.py                           # 流水线主控脚本
├── USER_CONFIG.example.py           # 用户配置示例
├── requirements.txt
├── .gitignore
│
├── 0_convert/                       # Step 0, 4, 5 的脚本
│   ├── convert_refer_to_csv.py
│   ├── create_clean_json.py
│   └── create_corrupted_json.py
│
├── 1_noise_assignment/
│   ├── noise_assignment.py
│   └── noise_compatibility_KS50_VGGSound.xlsx   # 噪声-类别兼容性表
│
├── 2_corruption/
│   ├── make_c_video.py              # Step 2
│   ├── make_c_audio.py              # Step 3
│   └── make_corruptions/            # frost 噪声素材
│       ├── frost1.png ~ frost3.png
│       └── frost4.jpg ~ frost6.jpg
│
├── data/                            # 预置元数据（随项目分发）
│   ├── refer/
│   │   ├── ks50_test_refer.json
│   │   ├── ks50_train_refer.json
│   │   └── vgg_test_refer.json
│   └── class_labels/
│       ├── class_labels_indices_ks50.csv
│       └── class_labels_indices_vgg.csv
│
├── _sample_lists/                   # Step 0 产出（可重新生成）
│   ├── ks50_sample_list.csv
│   ├── ks50_train_sample_list.csv
│   └── vgg_sample_list.csv
│
└── _output/                        # Step 1-5 产出
    ├── ks50_with_noise.csv
    ├── ks50_train_with_noise.csv
    ├── vgg_with_noise.csv
    ├── ks50_audio_C/                # 污染音频
    ├── ks50_train_audio_C/          # 污染音频
    ├── vgg_audio_C/                 # 污染音频
    ├── ks50_video_frames_C/         # 污染视频帧
    ├── ks50_train_video_frames_C/   # 污染视频帧
    ├── vgg_video_frames_C/          # 污染视频帧
    ├── clean/severity_N/severity_N.json
    └── corrupt/severity_N/severity_N.json
```
