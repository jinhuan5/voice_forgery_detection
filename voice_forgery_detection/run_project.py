#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动使用清理后数据的Web应用
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    """主函数"""
    print(" 启动语音伪造检测Web应用")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查模型文件
    model_files = [
        'models/detector.pkl',
        'models/scaler.pkl'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f" {model_file} 存在")
        else:
            print(f" {model_file} 不存在")
            print("请先运行: python train_model.py")
            return False
    
    # 切换到webapp目录
    if not os.path.exists('webapp'):
        print(" webapp目录不存在")
        return False
    
    print("\n启动Web应用...")
    print("访问地址: http://localhost:5000")
    print("健康检查: http://localhost:5000/health")
    print("\n按 Ctrl+C 停止应用")
    
    try:
        # 切换到webapp目录并启动应用
        os.chdir('webapp')
        subprocess.run([sys.executable, 'app.py'])
        
    except KeyboardInterrupt:
        print("\n\n应用已停止")
    except Exception as e:
        print(f"\n启动Web应用失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n ")
    else:
        print("\n 系统启动失败")
        sys.exit(1)
