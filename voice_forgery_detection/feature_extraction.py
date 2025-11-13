#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频特征提取模块
用于从音频中提取机器学习所需的特征
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import pickle

class AudioFeatureExtractor:
    """音频特征提取器"""
    
    def __init__(self, sr=16000):
        """
        初始化特征提取器
        
        参数:
        - sr: 采样率 (Hz)
        """
        self.sr = sr
        self.scaler = StandardScaler()
    
    def extract_mfcc(self, audio, n_mfcc=13):
        """
        提取MFCC特征（梅尔倒谱系数）
        
        MFCC是什么？
        - 就像把声音的"指纹"提取出来
        - 人耳对声音的感知不是线性的，MFCC模拟了人耳的听觉特性
        - 是语音识别中最常用的特征
        
        参数:
        - audio: 音频数据
        - n_mfcc: MFCC系数数量
        
        返回:
        - mfcc: MFCC特征矩阵
        """
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sr, 
            n_mfcc=n_mfcc,
            n_fft=2048,      # 窗口大小
            hop_length=512    # 跳跃长度
        )
        return mfcc
    
    def extract_spectral_features(self, audio):
        """
        提取频谱特征
        
        参数:
        - audio: 音频数据
        
        返回:
        - features: 频谱特征字典
        """
        # 1. 频谱质心（Spectral Centroid）
        # 表示声音的"亮度"，高频成分越多，质心越高
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        
        # 2. 频谱带宽（Spectral Bandwidth）
        # 表示频谱的宽度
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        
        # 3. 频谱对比度（Spectral Contrast）
        # 表示频谱的对比度
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        
        # 4. 零交叉率（Zero Crossing Rate）
        # 表示信号穿过零点的频率
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        # 5. 色度特征（Chroma）
        # 表示音调信息
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        
        return {
            'spectral_centroid': spectral_centroids,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'zero_crossing_rate': zcr,
            'chroma': chroma
        }
    
    def extract_rhythm_features(self, audio):
        """
        提取节奏特征
        
        参数:
        - audio: 音频数据
        
        返回:
        - features: 节奏特征
        """
        # 1. 节拍跟踪
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
        
        # 2. 节奏强度
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
        
        return {
            'tempo': tempo,
            'beat_frames': beats,
            'onset_times': onset_times
        }
    
    def extract_all_features(self, audio):
        """
        提取所有特征
        
        参数:
        - audio: 音频数据
        
        返回:
        - feature_vector: 特征向量
        """
        features = []
        
        # 1. MFCC特征
        mfcc = self.extract_mfcc(audio)
        # 计算MFCC的统计特征
        for i in range(mfcc.shape[0]):
            features.extend([
                np.mean(mfcc[i]),      # 均值
                np.std(mfcc[i]),       # 标准差
                np.min(mfcc[i]),       # 最小值
                np.max(mfcc[i])        # 最大值
            ])
        
        # 2. 频谱特征
        spectral_features = self.extract_spectral_features(audio)
        
        # 频谱质心统计
        features.extend([
            np.mean(spectral_features['spectral_centroid']),
            np.std(spectral_features['spectral_centroid'])
        ])
        
        # 频谱带宽统计
        features.extend([
            np.mean(spectral_features['spectral_bandwidth']),
            np.std(spectral_features['spectral_bandwidth'])
        ])
        
        # 零交叉率统计
        features.extend([
            np.mean(spectral_features['zero_crossing_rate']),
            np.std(spectral_features['zero_crossing_rate'])
        ])
        
        # 3. 节奏特征
        rhythm_features = self.extract_rhythm_features(audio)
        features.append(rhythm_features['tempo'])
        
        return np.array(features)
    
    def visualize_features(self, audio, save_path=None):
        """
        可视化音频特征
        
        参数:
        - audio: 音频数据
        - save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 时域波形
        axes[0, 0].plot(audio)
        axes[0, 0].set_title('时域波形')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('幅度')
        
        # 2. 频谱图
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz', ax=axes[0, 1])
        axes[0, 1].set_title('频谱图')
        
        # 3. MFCC特征
        mfcc = self.extract_mfcc(audio)
        librosa.display.specshow(mfcc, sr=self.sr, x_axis='time', ax=axes[1, 0])
        axes[1, 0].set_title('MFCC特征')
        
        # 4. 色度特征
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        librosa.display.specshow(chroma, sr=self.sr, x_axis='time', y_axis='chroma', ax=axes[1, 1])
        axes[1, 1].set_title('色度特征')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def process_dataset(self, data_dir, output_dir):
        """
        处理整个数据集
        
        参数:
        - data_dir: 数据目录
        - output_dir: 输出目录
        """
        print("开始特征提取...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理真实语音
        real_dir = os.path.join(data_dir, "real")
        if os.path.exists(real_dir):
            print("处理真实语音数据...")
            real_features = self.process_class(real_dir, "real")
            np.save(os.path.join(output_dir, "real_features.npy"), real_features)
        
        # 处理伪造语音
        fake_dir = os.path.join(data_dir, "fake")
        if os.path.exists(fake_dir):
            print(" 处理伪造语音数据...")
            fake_features = self.process_class(fake_dir, "fake")
            np.save(os.path.join(output_dir, "fake_features.npy"), fake_features)
        
        # 合并特征和标签
        self.create_training_data(output_dir)
        
        print("特征提取完成！")
    
    def process_class(self, class_dir, class_name):
        """
        处理单个类别的数据
        
        参数:
        - class_dir: 类别目录
        - class_name: 类别名称
        
        返回:
        - features: 特征矩阵
        """
        features_list = []
        audio_files = []
        
        # 获取所有音频文件
        for file in os.listdir(class_dir):
            if file.endswith(('.wav', '.mp3', '.m4a')):
                audio_files.append(os.path.join(class_dir, file))
        
        print(f"找到 {len(audio_files)} 个 {class_name} 音频文件")
        
        # 处理每个音频文件
        for i, file_path in enumerate(audio_files):
            try:
                # 加载音频
                audio, sr = librosa.load(file_path, sr=self.sr)
                
                # 提取特征
                features = self.extract_all_features(audio)
                features_list.append(features)
                
                # 显示进度
                if (i + 1) % 10 == 0:
                    print(f" 已处理 {i + 1}/{len(audio_files)} 个文件")
                    
            except Exception as e:
                print(f" 处理文件失败 {file_path}: {e}")
                continue
        
        return np.array(features_list)
    
    def create_training_data(self, output_dir):
        """
        创建训练数据
        
        参数:
        - output_dir: 输出目录
        """
        # 加载特征
        real_features = np.load(os.path.join(output_dir, "real_features.npy"))
        fake_features = np.load(os.path.join(output_dir, "fake_features.npy"))
        
        # 创建标签
        real_labels = np.zeros(len(real_features))  # 真实语音标签为0
        fake_labels = np.ones(len(fake_features))    # 伪造语音标签为1
        
        # 合并特征和标签
        X = np.vstack([real_features, fake_features])
        y = np.hstack([real_labels, fake_labels])
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 保存数据
        np.save(os.path.join(output_dir, "X_scaled.npy"), X_scaled)
        np.save(os.path.join(output_dir, "y.npy"), y)
        
        # 保存标准化器
        with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"   数据集统计:")
        print(f"   总样本数: {len(X)}")
        print(f"   真实语音: {len(real_features)}")
        print(f"   伪造语音: {len(fake_features)}")
        print(f"   特征维度: {X.shape[1]}")

def main():
    """主函数"""
    print("音频特征提取工具")
    print("=" * 50)
    
    # 创建特征提取器
    extractor = AudioFeatureExtractor()
    
    # 处理数据集
    data_dir = "data/processed"
    output_dir = "features"
    
    if os.path.exists(data_dir):
        extractor.process_dataset(data_dir, output_dir)
    else:
        print(" 请先运行数据预处理脚本")
        print("运行命令: python data_preparation.py")

if __name__ == "__main__":
    main()
