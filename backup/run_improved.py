#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键启动改进版语音伪造检测系统
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_step(step, description):
    """打印步骤信息"""
    print(f"\n{'='*60}")
    print(f"步骤 {step}: {description}")
    print(f"{'='*60}")

def run_script(script_name, description):
    """运行Python脚本"""
    print(f"\n运行 {description}...")
    print(f"脚本: {script_name}")
    
    try:
        # 运行脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(f"✅ {description} 成功完成")
            if result.stdout:
                print("输出信息:")
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} 失败")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行 {description} 时出错: {e}")
        return False

def check_requirements():
    """检查依赖包"""
    print_step(1, "检查依赖包")
    
    # 定义包名映射（有些包的导入名和安装名不同）
    package_mapping = {
        'numpy': 'numpy',
        'librosa': 'librosa', 
        'sklearn': 'scikit-learn',  # scikit-learn的导入名是sklearn
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'flask': 'flask',
        'joblib': 'joblib',
        'soundfile': 'soundfile'
    }
    
    missing_packages = []
    
    for import_name, package_name in package_mapping.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} 已安装")
        except ImportError:
            print(f"❌ {package_name} 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("所有依赖包检查完成")
    return True

def check_data_structure():
    """检查数据结构"""
    print_step(2, "检查数据结构")
    
    # 检查数据目录
    data_dirs = ['data/real', 'data/fake']
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            print(f"✅ {data_dir}: {len(files)} 个文件")
        else:
            print(f"❌ {data_dir} 不存在")
            return False
    
    # 检查特征文件
    feature_files = ['features/X_scaled.npy', 'features/y.npy', 'features/scaler.pkl']
    for feature_file in feature_files:
        if os.path.exists(feature_file):
            print(f"✅ {feature_file} 存在")
        else:
            print(f"❌ {feature_file} 不存在")
            print("请先运行数据预处理")
            return False
    
    print("数据结构检查完成")
    return True

def main():
    """主函数"""
    print("🚀 启动改进版语音伪造检测系统")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 检查依赖包
    if not check_requirements():
        print("\n❌ 依赖包检查失败，请安装缺少的包")
        return False
    
    # 2. 检查数据结构
    if not check_data_structure():
        print("\n❌ 数据结构检查失败，请检查数据文件")
        return False
    
    # 3. 训练改进模型
    print_step(3, "训练改进模型")
    if not run_script("train_improved.py", "改进模型训练"):
        print("\n❌ 改进模型训练失败")
        return False
    
    # 4. 启动改进版Web应用
    print_step(4, "启动改进版Web应用")
    print("启动改进版Web应用...")
    print("访问地址: http://localhost:5000")
    print("健康检查: http://localhost:5000/health")
    print("\n按 Ctrl+C 停止应用")
    
    try:
        # 切换到webapp目录
        os.chdir('webapp')
        
        # 启动改进版Web应用
        subprocess.run([sys.executable, 'app_improved.py'])
        
    except KeyboardInterrupt:
        print("\n\n应用已停止")
    except Exception as e:
        print(f"\n启动Web应用失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 改进版系统启动成功！")
    else:
        print("\n❌ 系统启动失败")
        sys.exit(1)
