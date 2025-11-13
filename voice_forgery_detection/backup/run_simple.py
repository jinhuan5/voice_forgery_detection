#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的项目启动脚本（无emoji版本）
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"目标: {title}")
    print("=" * 60)

def print_step(step_num, description):
    """打印步骤"""
    print(f"\n步骤 {step_num}: {description}")
    print("-" * 40)

def check_environment():
    """检查环境"""
    print_step(1, "检查Python环境")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("Python版本过低，需要Python 3.7+")
        return False
    
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的库
    required_libs = ['numpy', 'librosa', 'sklearn', 'flask', 'matplotlib']
    missing_libs = []
    
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"已安装: {lib}")
        except ImportError:
            missing_libs.append(lib)
            print(f"未安装: {lib}")
    
    if missing_libs:
        print(f"\n缺少依赖库: {', '.join(missing_libs)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """创建必要的目录"""
    print_step(2, "创建项目目录结构")
    
    directories = [
        'data/real',
        'data/fake', 
        'data/processed/real',
        'data/processed/fake',
        'features',
        'models',
        'uploads',
        'webapp/templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def check_data():
    """检查数据"""
    print_step(3, "检查训练数据")
    
    real_dir = Path("data/real")
    fake_dir = Path("data/fake")
    
    real_files = list(real_dir.glob("*.wav")) + list(real_dir.glob("*.mp3"))
    fake_files = list(fake_dir.glob("*.wav")) + list(fake_dir.glob("*.mp3"))
    
    print(f"真实语音文件: {len(real_files)} 个")
    print(f"伪造语音文件: {len(fake_files)} 个")
    
    if len(real_files) == 0 or len(fake_files) == 0:
        print("数据不足，请准备训练数据")
        print("真实语音放在: data/real/")
        print("伪造语音放在: data/fake/")
        return False
    
    return True

def run_script(script_name, description):
    """运行脚本"""
    print(f"运行 {description}...")
    
    if not Path(script_name).exists():
        print(f"脚本不存在: {script_name}")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"{description}完成")
            return True
        else:
            print(f"{description}失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"{description}超时")
        return False
    except Exception as e:
        print(f"{description}出错: {e}")
        return False

def main():
    """主函数"""
    print_header("语音伪造检测项目启动")
    
    # 1. 检查环境
    if not check_environment():
        print("\n环境检查失败，请先安装依赖")
        return False
    
    # 2. 创建目录
    create_directories()
    
    # 3. 检查数据
    if not check_data():
        print("\n请先准备训练数据，然后重新运行")
        return False
    
    # 4. 数据预处理（包含特征提取）
    if not run_script("data_preparation.py", "数据预处理"):
        print("\n数据预处理失败")
        return False
    
    # 5. 模型训练
    if not run_script("train_simple.py", "模型训练"):
        print("\n模型训练失败")
        return False
    
    # 6. 启动Web应用
    print_step(6, "启动Web应用")
    print("启动Web应用...")
    print("访问地址: http://localhost:5000")
    print("健康检查: http://localhost:5000/health")
    print("\n按 Ctrl+C 停止应用")
    
    try:
        subprocess.run([sys.executable, "webapp/app.py"])
    except KeyboardInterrupt:
        print("\nWeb应用已停止")
    except Exception as e:
        print(f"Web应用启动失败: {e}")
        return False
    
    print("\n项目运行完成！")
    return True

if __name__ == "__main__":
    main()
