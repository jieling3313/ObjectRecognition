# One-Shot 動作辨識系統實作

> 詳細技術資訊請參閱 [TECHNICAL_GUIDE.md](./TECHNICAL_GUIDE.md)

## 專案概述

- **主題**：在 ROS Noetic + Docker 環境中實作 One-Shot Action Recognition
- **方法**：Multi-Scale Spatial-Temporal Skeleton Matching + Earth Mover's Distance (EMD)
- **硬體**：Intel RealSense D435i 相機
- **應用**：辨識動作（例如：揮手、跌倒）僅需 1-5 個範例
- **環境**：Ubuntu 20.04 (Docker) + ROS Noetic + YOLOv8-Pose

---

## 開發時間軸

### 2025-11-19：初始實作
- ✅ 建立 `skeleton_extractor.py` - YOLOv8-Pose 骨架提取（17 COCO 關鍵點）
- ✅ 建立 `skeleton_model.py` - AGCN + EMD 模型實作
  - COCOGraph：3 個空間尺度的圖結構
  - SkeletonEmbedding：多尺度嵌入網路
  - EMDMatcher：最佳傳輸距離匹配
- ✅ 建立 `one_shot_action_node.py` - ROS 即時辨識節點
- ✅ 建立 `record_support_set.py` - 支持集錄製工具
- ✅ 建立 `action_recognition.launch` - 系統啟動檔
- ✅ 更新 `.devcontainer/Dockerfile` - 新增 `pot` 和 `scipy` 依賴
- 🔧 國際化調整：所有 Python 執行時字串改為英文，註解保留繁體中文

### 2025-11-20：NTU RGB+D Dataset 整合與 GPU 配置
- ✅ 建立 `test_ntu_rgbd_loader.py` - 測試 NTU RGB+D 資料載入
- ✅ 建立 `train_ntu_rgbd.py` - 預訓練腳本（56,880 骨架序列，60 動作類別）
- 🐛 修復骨架讀取錯誤：joint 資料格式從 `(x, y, z, ...)` 改為 `(x, y, confidence=1.0)`
- ✅ 配置 GPU 支援：
  - 安裝 NVIDIA Container Toolkit
  - 更新 `docker-compose.yml` 啟用 GPU runtime
  - 新增 `tqdm` 依賴
- 📊 訓練速度對比：GPU (RTX 3060) 比 CPU 快 **8-10 倍**

### 2025-11-22：RTX 5080 GPU 環境建置
- ✅ 完成 RTX 5080 Laptop GPU (sm_120) 支援配置
- ✅ 升級至 Python 3.10.16 + PyTorch 2.9.1+cu128 + CUDA 12.8
- 🐛 解決 NumPy 2.x 與 POT 相容性問題：升級至 POT 0.9.6+
- 🔧 建置最佳化：移除 `--no-cache`，使用 Docker 層快取加速
- 📝 建立 `RTX5080_GPU_SETUP_LOG.md` - 完整 GPU 配置記錄
- 📝 建立 `TECHNICAL_GUIDE.md` - 詳細技術文件
- 📊 GPU 驗證成功：PyTorch 2.9.1+cu128 完整支援 RTX 5080 (sm_120)

---

## 快速開始

### 環境建置

#### CPU 環境（基礎方案）
```bash
cd /home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/.devcontainer
docker compose build
docker compose up -d
```

#### GPU 環境（RTX 5080 / RTX 30/40 系列）
```bash
# 詳細步驟請參閱 RTX5080_GPU_SETUP_LOG.md
cd /home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/.devcontainer
./rebuild_with_gpu.sh  # 自動建置並啟動 GPU 容器
```

### 測試骨架提取
```bash
docker compose exec ros-dev bash
cd /root/catkin_ws/src/yolo_ros/scripts
python3.10 test_skeleton_extractor.py
```

### 訓練模型（使用 NTU RGB+D 預訓練）
```bash
# GPU 訓練（推薦）
python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 50 \
    --batch_size 32 \
    --device cuda

# CPU 訓練（備用）
python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 5 \
    --batch_size 8 \
    --device cpu
```

### 錄製支持集
```bash
rosrun yolo_ros record_support_set.py --action waving --num_samples 5
```

### 啟動即時辨識
```bash
roslaunch yolo_ros action_recognition.launch device:=cuda
```

---

## 核心檔案

### Python 模組
- `scripts/skeleton_extractor.py` - YOLOv8-Pose 骨架提取
- `scripts/skeleton_model.py` - AGCN + EMD 模型
- `scripts/one_shot_action_node.py` - ROS 即時辨識節點
- `scripts/record_support_set.py` - 支持集錄製工具
- `scripts/train_ntu_rgbd.py` - NTU RGB+D 預訓練腳本
- `scripts/test_ntu_rgbd_loader.py` - 資料載入測試

### 配置檔案
- `launch/action_recognition.launch` - 系統啟動檔
- `.devcontainer/Dockerfile` - Docker 環境配置
- `.devcontainer/docker-compose.yml` - GPU/CPU 容器配置
- `.devcontainer/rebuild_with_gpu.sh` - GPU 環境一鍵建置腳本

### 文件
- `TECHNICAL_GUIDE.md` - 完整技術文件（模型架構、訓練流程、程式碼範例）
- `RTX5080_GPU_SETUP_LOG.md` - RTX 5080 GPU 配置完整記錄
- `ONE_SHOT_ACTION_RECOGNITION.md` - 本文件（專案概覽）

---

## ROS Topics & Services

### 訂閱的 Topics
- `/camera/color/image_raw` (sensor_msgs/Image) - RGB 影像輸入

### 發布的 Topics
- `/action_recognition/result` (std_msgs/String) - 辨識結果（動作名稱）
- `/action_recognition/score` (std_msgs/Float32) - 信心分數
- `/action_recognition/annotated_image` (sensor_msgs/Image) - 標註影像

### 服務
- `/start_recording` - 開始錄製新動作
- `/stop_recording` - 停止並儲存錄製
- `/reload_support_set` - 重新載入支持動作

---

## 主要參數

### Launch 檔案參數
- `buffer_size`：骨架緩衝區大小（預設：64 幀）
- `recognition_interval`：辨識間隔（預設：每 30 幀）
- `confidence_threshold`：信心門檻（預設：0.5）
- `pose_model`：姿態模型（預設：yolov8m-pose.pt）
- `device`：運算裝置（預設：cpu，可選：cuda）

### 訓練參數
- `--data_path`：NTU RGB+D 資料集路徑（必填）
- `--epochs`：訓練週期（預設：50）
- `--batch_size`：批次大小（GPU：32，CPU：8）
- `--device`：裝置（cuda/cpu）
- `--learning_rate`：學習率（預設：0.001）

---

## 效能指標

### 訓練速度（NTU RGB+D Dataset）
- **GPU (RTX 5080)**: ~3-4 秒/epoch（batch_size=32）
- **GPU (RTX 3060)**: ~5-6 秒/epoch（batch_size=32）
- **CPU (AMD Ryzen 7)**: ~50-60 秒/epoch（batch_size=8）

### 推論速度
- **YOLOv8m-Pose**: ~30-50 ms/幀（GPU），~100-150 ms/幀（CPU）
- **EMD 匹配**: ~10-20 ms（取決於支持集大小）

---

## 依賴套件

### Python 套件
- `ultralytics` - YOLOv8-Pose
- `torch`, `torchvision` - PyTorch 深度學習框架
- `pot` (≥0.9.6) - Python Optimal Transport（EMD 計算）
- `scipy` - 科學計算
- `opencv-python` - 影像處理
- `numpy` - 數值計算
- `tqdm` - 進度條顯示

### ROS 套件
- `rospy` - Python ROS 客戶端
- `sensor_msgs` - 影像訊息
- `std_msgs` - 標準訊息
- `cv_bridge` - OpenCV/ROS 影像轉換

---

## 注意事項

1. **One-Shot Learning 特性**：本系統設計為少樣本學習，每個動作僅需 1-5 個範例即可辨識
2. **NTU RGB+D 預訓練**：可選使用 56,880 骨架序列進行預訓練以提升特徵品質
3. **GPU 記憶體**：完整訓練建議至少 4GB VRAM（batch_size=32）
4. **相機要求**：需支援 RGB 影像輸出的相機（例如：RealSense D435i）

---

## 常見問題

### Q1: 是否必須下載 NTU RGB+D Dataset？
**A**: 不一定。直接使用（無預訓練）即可進行 One-Shot 辨識，但使用預訓練權重可提升準確度。

### Q2: GPU 訓練無法啟用？
**A**: 檢查以下項目：
- NVIDIA Container Toolkit 是否已安裝？
- `docker-compose.yml` 是否包含 `runtime: nvidia`？
- PyTorch 是否正確安裝 CUDA 版本？

詳細排解請參閱 `RTX5080_GPU_SETUP_LOG.md`

### Q3: 訓練時出現 NumPy dtype 錯誤？
**A**: 升級 POT 至 0.9.6+ 版本：
```bash
python3.10 -m pip install --force-reinstall "pot>=0.9.6"
```

### Q4: 如何切換 CPU/GPU 訓練？
**A**: 使用 `--device` 參數：
- GPU: `--device cuda`
- CPU: `--device cpu`

---

## 參考資料

- **論文**: One-Shot Action Recognition via Multi-Scale Spatial-Temporal Skeleton Matching
- **技術文件**: [TECHNICAL_GUIDE.md](./TECHNICAL_GUIDE.md)
- **GPU 配置**: [RTX5080_GPU_SETUP_LOG.md](./RTX5080_GPU_SETUP_LOG.md)
- **YOLOv8-Pose**: https://github.com/ultralytics/ultralytics
- **POT Library**: https://pythonot.github.io/

---

**最後更新**: 2025-11-22
**版本**: 2.0 (簡化版)
