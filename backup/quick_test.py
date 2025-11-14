#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试特征提取
"""

import numpy as np
import librosa
import soundfile as sf

def test_feature_extraction():
    """测试特征提取"""
    print("测试特征提取...")
    
    # 创建测试音频
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
    
    # 保存临时音频文件
    temp_audio_path = "temp_test.wav"
    sf.write(temp_audio_path, test_audio, sr)
    
    try:
        # 加载音频
        audio, sr = librosa.load(temp_audio_path, sr=16000)
        
        features = []
        
        # 1. MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        for i in range(mfcc.shape[0]):
            features.extend([
                float(np.mean(mfcc[i])),      # 均值
                float(np.std(mfcc[i])),       # 标准差
                float(np.min(mfcc[i])),       # 最小值
                float(np.max(mfcc[i]))         # 最大值
            ])
        
        # 2. 频谱特征
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        features.extend([
            float(np.mean(spectral_centroids)),
            float(np.std(spectral_centroids)),
            float(np.mean(spectral_bandwidth)),
            float(np.std(spectral_bandwidth)),
            float(np.mean(zcr)),
            float(np.std(zcr))
        ])
        
        # 3. 节奏特征
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(float(tempo))
        
        print(f"原始特征数量: {len(features)}")
        
        # 确保特征数量为52个
        if len(features) > 52:
            features = features[:52]
        elif len(features) < 52:
            features.extend([0.0] * (52 - len(features)))
        
        print(f"最终特征数量: {len(features)}")
        print(f"特征向量: {features[:10]}...")  # 显示前10个特征
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False
    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    success = test_feature_extraction()
    if success:
        print("✅ 特征提取测试通过！")
    else:
        print("❌ 特征提取测试失败")
