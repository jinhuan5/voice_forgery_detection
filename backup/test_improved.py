#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进版语音伪造检测模型
"""

import os
import numpy as np
import librosa
import joblib
from datetime import datetime

def test_improved_model():
    """测试改进版模型"""
    print("🧪 测试改进版语音伪造检测模型")
    print("=" * 50)
    
    # 检查模型文件
    model_path = "models/improved_detector.pkl"
    scaler_path = "models/improved_scaler.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    if not os.path.exists(scaler_path):
        print(f"❌ 标准化器文件不存在: {scaler_path}")
        return False
    
    try:
        # 加载模型
        print("加载改进版模型...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ 模型加载成功")
        
        # 检查模型信息
        print(f"模型类型: {type(model).__name__}")
        
        # 测试特征提取
        print("\n测试特征提取...")
        
        # 创建一个测试音频（简单的正弦波）
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
        
        # 保存临时音频文件
        temp_audio_path = "temp_test.wav"
        import soundfile as sf
        sf.write(temp_audio_path, test_audio, sr)
        
        # 提取特征
        features = extract_advanced_features(temp_audio_path)
        
        if features is not None:
            print(f"✅ 特征提取成功，特征数量: {len(features)}")
            
            # 标准化特征
            features_scaled = scaler.transform(features.reshape(1, -1))
            print(f"✅ 特征标准化成功，形状: {features_scaled.shape}")
            
            # 预测
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            print(f"\n预测结果:")
            print(f"  预测类别: {'伪造' if prediction else '真实'}")
            print(f"  真实概率: {probability[0]:.4f}")
            print(f"  伪造概率: {probability[1]:.4f}")
            print(f"  置信度: {max(probability):.4f}")
            
        else:
            print("❌ 特征提取失败")
            return False
        
        # 清理临时文件
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        print("\n✅ 改进版模型测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def extract_advanced_features(audio_path):
    """提取高级特征（与训练时保持一致）"""
    try:
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 使用与训练时相同的特征提取逻辑
        features = []
        
        # 1. MFCC特征（与训练时一致）
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        for i in range(mfcc.shape[0]):
            features.extend([
                float(np.mean(mfcc[i])),      # 均值
                float(np.std(mfcc[i])),       # 标准差
                float(np.min(mfcc[i])),       # 最小值
                float(np.max(mfcc[i]))         # 最大值
            ])
        
        # 2. 频谱特征（与训练时一致）
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
        
        # 3. 节奏特征（与训练时一致）
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(float(tempo))
        
        # 确保特征数量为52个（与训练时一致）
        if len(features) > 52:
            features = features[:52]
        elif len(features) < 52:
            # 如果特征不足，用零填充
            features.extend([0.0] * (52 - len(features)))
        
        return np.array(features)
        
    except Exception as e:
        print(f"特征提取失败: {e}")
        return None

if __name__ == "__main__":
    success = test_improved_model()
    if success:
        print("\n🎉 改进版模型测试通过！")
    else:
        print("\n❌ 改进版模型测试失败")
