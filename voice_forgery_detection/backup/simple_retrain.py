#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版重新训练脚本
"""

import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import glob

def extract_features(audio_path):
    """提取特征"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        features = []
        
        # MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        for i in range(mfcc.shape[0]):
            features.extend([
                float(np.mean(mfcc[i])),
                float(np.std(mfcc[i])),
                float(np.min(mfcc[i])),
                float(np.max(mfcc[i]))
            ])
        
        # 频谱特征
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
        
        # 节奏特征
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(float(tempo))
        
        # 确保52个特征
        if len(features) > 52:
            features = features[:52]
        elif len(features) < 52:
            features.extend([0.0] * (52 - len(features)))
        
        return np.array(features)
        
    except Exception as e:
        print(f"特征提取失败 {audio_path}: {e}")
        return None

def main():
    """主函数"""
    print("开始使用清理后的数据重新训练模型")
    
    # 加载数据
    print("加载数据...")
    real_features = []
    fake_features = []
    
    # 真实语音
    real_dir = "data/real"
    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav")) + glob.glob(os.path.join(real_dir, "*.mp3"))
        print(f"找到 {len(real_files)} 个真实语音文件")
        
        for i, file_path in enumerate(real_files[:100]):  # 限制数量避免内存问题
            if i % 20 == 0:
                print(f"处理真实语音 {i+1}/{min(100, len(real_files))}")
            features = extract_features(file_path)
            if features is not None:
                real_features.append(features)
    
    # 伪造语音
    fake_dir = "data/fake"
    if os.path.exists(fake_dir):
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav")) + glob.glob(os.path.join(fake_dir, "*.mp3"))
        print(f"找到 {len(fake_files)} 个伪造语音文件")
        
        for i, file_path in enumerate(fake_files[:100]):  # 限制数量避免内存问题
            if i % 20 == 0:
                print(f"处理伪造语音 {i+1}/{min(100, len(fake_files))}")
            features = extract_features(file_path)
            if features is not None:
                fake_features.append(features)
    
    if len(real_features) == 0 or len(fake_features) == 0:
        print("错误：没有找到有效的音频文件")
        return
    
    print(f"数据统计:")
    print(f"  真实语音: {len(real_features)}")
    print(f"  伪造语音: {len(fake_features)}")
    
    # 合并数据
    X = np.vstack([real_features, fake_features])
    y = np.hstack([np.zeros(len(real_features)), np.ones(len(fake_features))])
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 训练模型
    print("训练模型...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"模型性能:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    
    # 保存模型
    print("保存模型...")
    os.makedirs("models", exist_ok=True)
    
    model_path = "models/clean_detector.pkl"
    scaler_path = "models/clean_scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"模型保存成功: {model_path}")
    print(f"标准化器保存成功: {scaler_path}")
    
    # 保存模型信息
    with open("models/clean_model_info.txt", 'w', encoding='utf-8') as f:
        f.write(f"model_name: RandomForestClassifier\n")
        f.write(f"accuracy: {accuracy:.4f}\n")
        f.write(f"auc_score: {auc_score:.4f}\n")
        f.write(f"training_time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("重新训练完成！")

if __name__ == "__main__":
    main()
