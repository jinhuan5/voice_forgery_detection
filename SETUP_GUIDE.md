# 🚀 项目环境搭建指南

## 第一步：安装Python

### Windows用户
1. 访问 [Python官网](https://www.python.org/downloads/)
2. 下载Python 3.8或更高版本
3. 安装时**务必勾选**"Add Python to PATH"
4. 验证安装：打开命令提示符，输入 `python --version`

### Mac用户
```bash
# 使用Homebrew安装（推荐）
brew install python3

# 或从官网下载安装包
```

### Linux用户
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# CentOS/RHEL
sudo yum install python3 python3-pip
```

## 第二步：创建虚拟环境（推荐）

虚拟环境就像给每个项目准备一个独立的工具箱，避免不同项目的库版本冲突。

```bash
# 创建虚拟环境
python -m venv voice_detection_env

# 激活虚拟环境
# Windows:
voice_detection_env\Scripts\activate

# Mac/Linux:
source voice_detection_env/bin/activate
```

## 第三步：安装项目依赖

```bash
# 确保在项目目录下
cd voice_forgery_detection

# 安装所有依赖库
pip install -r requirements.txt
```

## 第四步：验证安装

创建一个测试文件来验证所有库都安装成功：

```python
# test_installation.py
try:
    import librosa
    import sklearn
    import flask
    import numpy as np
    import matplotlib.pyplot as plt
    print("✅ 所有依赖库安装成功！")
    print(f"librosa版本: {librosa.__version__}")
    print(f"scikit-learn版本: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ 安装失败: {e}")
```

运行测试：
```bash
python test_installation.py
```

## 常见问题解决

### 问题1：pip不是内部命令
**解决方案**：重新安装Python，确保勾选"Add Python to PATH"

### 问题2：librosa安装失败
**解决方案**：
```bash
# 先安装音频处理依赖
pip install soundfile
pip install librosa
```

### 问题3：权限错误
**解决方案**：
```bash
# 使用用户安装模式
pip install --user -r requirements.txt
```

## 下一步
环境搭建完成后，我们就可以开始准备数据了！
