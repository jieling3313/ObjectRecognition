#!/bin/bash
# 重建容器以支援 GPU 訓練
# 此腳本會重建 Docker 容器，安裝 PyTorch Nightly（支援 RTX 5080 Blackwell 架構）

set -e

echo "=========================================="
echo "重建容器以支援 GPU 訓練"
echo "安裝 PyTorch Nightly 版本"
echo "=========================================="

# 1. 停止並移除舊容器
echo "步驟 1/4: 停止並移除舊容器..."
docker compose down
echo "  ✓ 舊容器已停止"

# 2. 重建映像（包含 PyTorch Nightly）
echo "步驟 2/4: 重建 Docker 映像..."
echo "  注意：這將安裝 PyTorch Nightly 版本，可能需要 10-15 分鐘"
docker compose build --no-cache
echo "  ✓ 映像重建完成"

# 3. 啟動新容器
echo "步驟 3/4: 啟動新容器..."
docker compose up -d
echo "  ✓ 容器已啟動"

# 4. 驗證 GPU 支援
echo "步驟 4/4: 驗證 GPU 支援..."
echo ""
docker compose exec ros-dev bash -c "python3 -c 'import torch; print(\"=\"*60); print(\"PyTorch 版本:\", torch.__version__); print(\"CUDA 版本:\", torch.version.cuda); print(\"CUDA 可用:\", torch.cuda.is_available()); print(\"GPU 數量:\", torch.cuda.device_count()); print(\"GPU 名稱:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"); print(\"=\"*60)'"

echo ""
echo "=========================================="
echo "✓ 容器重建完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 如果沒有警告，可以開始 GPU 訓練"
echo "  2. 進入容器：docker compose exec ros-dev bash"
echo "  3. 開始訓練：cd /root/catkin_ws/src/yolo_ros/scripts && python3 train_ntu_rgbd.py --device cuda"
echo ""
