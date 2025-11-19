# --- 最終通用版：從 Ubuntu 20.04 開始，使用官方來源並繞過 GPG 伺服器 ---
    FROM ubuntu:20.04

    # 設定環境變數
    ENV DEBIAN_FRONTEND=noninteractive
    ENV TZ=Asia/Taipei
    RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
    
    #安裝基礎工具與完整的 ROS Noetic
    #
    RUN apt-get update && apt-get install -y --no-install-recommends \
        # 基礎工具
        git cmake build-essential \
        # ROS 與憑證所需工具
        curl gnupg2 lsb-release ca-certificates && \
        # 新增 ROS 套件來源
        sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
        # 從 GitHub 下載金鑰檔案
        curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /tmp/ros.key && \
        apt-key add /tmp/ros.key && \
        # 再次更新套件列表並安裝 ROS
        apt-get update && \
        apt-get install -y --no-install-recommends ros-noetic-desktop-full && \
        # 清理 apt 快取與臨時檔案
        rm -rf /var/lib/apt/lists/* && \
        rm -f /tmp/ros.key
    
    # 設定 ROS 環境
    RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
    ENV ROS_HOSTNAME=localhost
    ENV ROS_MASTER_URI=http://localhost:11311
    # ros-noetic d435i專用版本但下方已經載好更完善版本（使用下方版本）
    # RUN apt-get update && apt-get install -y \
    #     ros-noetic-realsense2-camera \
    #     && rm -rf /var/lib/apt/lists/*
        
    # 安裝編譯 librealsense 所需的額外依賴
    RUN apt-get update && apt-get install -y --no-install-recommends \
        udev \
        ros-noetic-ddynamic-reconfigure \
        libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev \
        python3-pip python3-rosdep python3-catkin-tools \
        && rm -rf /var/lib/apt/lists/*
    
    # 編譯與安裝 librealsense SDK
    WORKDIR /tmp
    ENV REALSENSE_VERSION v2.50.0
    RUN git clone --depth 1 --branch ${REALSENSE_VERSION} https://github.com/IntelRealSense/librealsense.git && \
        cd librealsense && \
        # 從 librealsense 原始碼中複製 UDEV 規則，解決權限問題
        cp config/99-realsense-libusb.rules /etc/udev/rules.d/ && \

        ./scripts/setup_udev_rules.sh && \
        mkdir build && cd build && \
        cmake ../ -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release && \
        make -j$(nproc) && \
        make install && \
        rm -rf /tmp/librealsense
    
    # 初始化 rosdep
    RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && rosdep init && rosdep update"
    
    # 設定 Python 環境
    RUN pip3 install --no-cache-dir --upgrade pip
    
    # 安裝 ultralytics
    RUN pip3 install --no-cache-dir ultralytics
    
    # PyTorch 相關套件
    RUN pip3 install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # 建立 Catkin 工作空間並編譯 realsense-ros
    WORKDIR /root/catkin_ws/src
    ENV REALSENSE_ROS_VERSION 2.3.2
    RUN git clone --depth 1 --branch ${REALSENSE_ROS_VERSION} https://github.com/IntelRealSense/realsense-ros.git
    # 使用 rosdep 自動安裝所有原始碼套件的相依性
    RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
        cd /root/catkin_ws && \
        rosdep install --from-paths src --ignore-src -r -y"
    # 返回工作空間根目錄
    WORKDIR /root/catkin_ws
    RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"
    # 設定容器啟動時的預設指令
    CMD ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && exec /bin/bash"]