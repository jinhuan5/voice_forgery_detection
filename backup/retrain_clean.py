#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键重新训练使用清理后的数据
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

def check_data_quality():
    """检查数据质量"""
    print_step(1, "检查数据质量")
    
    # 检查数据目录
    data_dirs = ['data/real', 'data/fake']
    total_files = 0
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            audio_files = [f for f in files if f.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg'))]
            print(f"✅ {data_dir}: {len(audio_files)} 个音频文件")
            total_files += len(audio_files)
        else:
            print(f"❌ {data_dir} 不存在")
            return False
    
    print(f"\n总音频文件数: {total_files}")
    
    if total_files < 10:
        print("⚠️ 警告: 音频文件数量较少，可能影响模型性能")
    
    return True

def main():
    """主函数"""
    print("🚀 使用清理后的数据重新训练语音伪造检测模型")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 检查数据质量
    if not check_data_quality():
        print("\n❌ 数据质量检查失败")
        return False
    
    # 2. 重新训练模型
    print_step(2, "使用清理后的数据重新训练模型")
    if not run_script("retrain_with_clean_data.py", "清理数据模型训练"):
        print("\n❌ 清理数据模型训练失败")
        return False
    
    # 3. 启动清理后的Web应用
    print_step(3, "启动清理后的Web应用")
    print("启动使用清理后数据的Web应用...")
    print("访问地址: http://localhost:5000")
    print("健康检查: http://localhost:5000/health")
    print("\n按 Ctrl+C 停止应用")
    
    try:
        # 切换到webapp目录
        os.chdir('webapp')
        
        # 启动清理后的Web应用
        subprocess.run([sys.executable, 'app_clean.py'])
        
    except KeyboardInterrupt:
        print("\n\n应用已停止")
    except Exception as e:
        print(f"\n启动Web应用失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 使用清理后的数据重新训练完成！")
        print("现在可以使用更准确的模型进行检测了")
    else:
        print("\n❌ 重新训练失败")
        sys.exit(1)
