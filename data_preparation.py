#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本
用于音频数据的预处理和格式标准化
"""

import os
import librosa
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AudioDataProcessor:
    """音频数据处理器"""
    
    def __init__(self, target_sr=16000, target_duration=5.0):
        """
        初始化音频处理器
        
        参数:
        - target_sr: 目标采样率 (Hz)
        - target_duration: 目标音频时长 (秒)
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_sr * target_duration)
    
    def load_audio(self, file_path):
        """
        加载音频文件
        
        参数:
        - file_path: 音频文件路径
        
        返回:
        - audio: 音频数据
        - sr: 采样率
        """
        try:
            # 使用librosa加载音频
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            return audio, sr
        except Exception as e:
            print(f" 加载音频失败 {file_path}: {e}")
            return None, None
    
    def preprocess_audio(self, audio):
        """
        预处理音频数据
        
        参数:
        - audio: 原始音频数据
        
        返回:
        - processed_audio: 处理后的音频
        """
        # 1. 去除静音段
        audio = self.remove_silence(audio)
        
        # 2. 标准化长度
        audio = self.normalize_length(audio)
        
        # 3. 音量标准化
        audio = self.normalize_volume(audio)
        
        return audio
    
    def remove_silence(self, audio, threshold=0.01):
        """
        去除静音段
        
        参数:
        - audio: 音频数据
        - threshold: 静音阈值
        
        返回:
        - trimmed_audio: 去除静音后的音频
        """
        # 找到非静音段的起始和结束位置
        non_silent = np.where(np.abs(audio) > threshold)[0]
        
        if len(non_silent) == 0:
            # 如果整个音频都是静音，返回零数组
            return np.zeros(self.target_length)
        
        start = non_silent[0]
        end = non_silent[-1]
        
        return audio[start:end+1]
    
    def normalize_length(self, audio):
        """
        标准化音频长度
        
        参数:
        - audio: 音频数据
        
        返回:
        - normalized_audio: 标准化长度的音频
        """
        current_length = len(audio)
        
        if current_length > self.target_length:
            # 如果太长，截取中间部分
            start = (current_length - self.target_length) // 2
            return audio[start:start + self.target_length]
        elif current_length < self.target_length:
            # 如果太短，用零填充
            padding = self.target_length - current_length
            return np.pad(audio, (0, padding), mode='constant')
        else:
            # 长度正好
            return audio
    
    def normalize_volume(self, audio):
        """
        音量标准化
        
        参数:
        - audio: 音频数据
        
        返回:
        - normalized_audio: 标准化音量的音频
        """
        # 计算RMS（均方根）
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            # 标准化到0.1的RMS值
            target_rms = 0.1
            audio = audio * (target_rms / rms)
        
        return audio
    
    def extract_features(self, audio):
        """
        提取音频特征
        
        参数:
        - audio: 音频数据
        
        返回:
        - features: 特征向量
        """
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.target_sr, 
            n_mfcc=13,  # 13个MFCC系数
            n_fft=2048,
            hop_length=512
        )
        
        # 计算统计特征
        features = []
        for i in range(mfcc.shape[0]):
            features.extend([
                np.mean(mfcc[i]),      # 均值
                np.std(mfcc[i]),       # 标准差
                np.min(mfcc[i]),       # 最小值
                np.max(mfcc[i])        # 最大值
            ])
        
        return np.array(features)
    
    def process_directory(self, input_dir, output_dir, label):
        """
        处理整个目录的音频文件
        
        参数:
        - input_dir: 输入目录
        - output_dir: 输出目录
        - label: 标签（'real' 或 'fake'）
        """
        print(f"处理 {label} 数据...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有音频文件
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
            audio_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
        if not audio_files:
            print(f"  在 {input_dir} 中没有找到音频文件")
            return
        
        print(f" 找到 {len(audio_files)} 个音频文件")
        
        # 处理每个音频文件
        processed_count = 0
        features_list = []
        
        for file_path in tqdm(audio_files, desc=f"处理{label}音频"):
            # 加载音频
            audio, sr = self.load_audio(file_path)
            if audio is None:
                continue
            
            # 预处理
            processed_audio = self.preprocess_audio(audio)
            
            # 提取特征
            features = self.extract_features(processed_audio)
            features_list.append(features)
            
            # 跳过音频保存，只提取特征
            # output_file = os.path.join(output_dir, os.path.basename(file_path))
            # import soundfile as sf
            # sf.write(output_file, processed_audio, self.target_sr)
            
            processed_count += 1
        
        # 保存特征
        features_array = np.array(features_list)
        features_file = os.path.join(output_dir, f"{label}_features.npy")
        np.save(features_file, features_array)
        
        print(f" 成功处理 {processed_count} 个文件")
        print(f" 特征保存到: {features_file}")
        
        return features_array

def main():
    """主函数"""
    print("语音数据预处理工具")
    print("=" * 50)
    
    # 创建处理器
    processor = AudioDataProcessor()
    
    # 处理真实语音数据
    real_dir = "data/real"
    if os.path.exists(real_dir):
        real_features = processor.process_directory(
            real_dir, 
            "data/processed/real", 
            "real"
        )
    else:
        print("  真实语音目录不存在，请先准备数据")
    
    # 处理伪造语音数据
    fake_dir = "data/fake"
    if os.path.exists(fake_dir):
        fake_features = processor.process_directory(
            fake_dir, 
            "data/processed/fake", 
            "fake"
        )
    else:
        print(" 伪造语音目录不存在，请先准备数据")
    
    # 合并特征和标签
    if real_features is not None and fake_features is not None:
        print("\n合并特征和标签...")
        
        # 创建标签
        real_labels = np.zeros(len(real_features))  # 真实语音标签为0
        fake_labels = np.ones(len(fake_features))   # 伪造语音标签为1
        
        # 合并特征和标签
        X = np.vstack([real_features, fake_features])
        y = np.hstack([real_labels, fake_labels])
        
        # 特征标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 保存数据
        np.save("features/X_scaled.npy", X_scaled)
        np.save("features/y.npy", y)
        
        # 保存标准化器
        import joblib
        joblib.dump(scaler, "features/scaler.pkl")
        
        print(f"数据集统计:")
        print(f"  总样本数: {len(X)}")
        print(f"  真实语音: {len(real_features)}")
        print(f"  伪造语音: {len(fake_features)}")
        print(f"  特征维度: {X.shape[1]}")
        print("数据预处理完成！")
    else:
        print("数据预处理失败，请检查数据文件")

if __name__ == "__main__":
    import glob
    main()
