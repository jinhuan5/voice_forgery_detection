#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音伪造检测预测脚本
用于单个音频文件的检测
"""

import sys
import os
import librosa
import numpy as np
import joblib
from feature_extraction import AudioFeatureExtractor

class VoicePredictor:
    """语音预测器"""
    
    def __init__(self, model_path="models/detector.pkl", scaler_path="models/scaler.pkl"):
        """
        初始化预测器
        
        参数:
        - model_path: 模型文件路径
        - scaler_path: 标准化器文件路径
        """
        self.model = None
        self.scaler = None
        self.feature_extractor = AudioFeatureExtractor()
        self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path, scaler_path):
        """
        加载训练好的模型
        
        参数:
        - model_path: 模型文件路径
        - scaler_path: 标准化器文件路径
        """
        try:
            # 加载模型
            self.model = joblib.load(model_path)
            print(f"✅ 模型加载成功: {model_path}")
            
            # 加载标准化器
            self.scaler = joblib.load(scaler_path)
            print(f"✅ 标准化器加载成功: {scaler_path}")
            
        except FileNotFoundError as e:
            print(f"❌ 模型文件未找到: {e}")
            print("请先运行 train_model.py 训练模型")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
    
    def predict_audio(self, audio_path):
        """
        预测音频是否为伪造
        
        参数:
        - audio_path: 音频文件路径
        
        返回:
        - result: 预测结果字典
        """
        if self.model is None or self.scaler is None:
            return {
                'error': '模型未加载，请先训练模型',
                'is_fake': None,
                'confidence': None
            }
        
        try:
            # 提取特征
            features = self.feature_extractor.extract_all_features_from_file(audio_path)
            if features is None:
                return {
                    'error': '特征提取失败',
                    'is_fake': None,
                    'confidence': None
                }
            
            # 标准化特征
            features_scaled = self.scaler.transform([features])
            
            # 预测
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # 获取置信度
            confidence = max(probability) * 100
            
            return {
                'is_fake': bool(prediction),
                'confidence': round(confidence, 2),
                'probability_real': round(probability[0] * 100, 2),
                'probability_fake': round(probability[1] * 100, 2)
            }
            
        except Exception as e:
            return {
                'error': f'预测失败: {str(e)}',
                'is_fake': None,
                'confidence': None
            }
    
    def predict_file(self, audio_path):
        """
        预测单个音频文件
        
        参数:
        - audio_path: 音频文件路径
        """
        print(f"🎵 分析音频文件: {audio_path}")
        print("-" * 50)
        
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            print(f"❌ 文件不存在: {audio_path}")
            return
        
        # 进行预测
        result = self.predict_audio(audio_path)
        
        if result['error']:
            print(f"❌ 预测失败: {result['error']}")
            return
        
        # 显示结果
        is_fake = result['is_fake']
        confidence = result['confidence']
        prob_real = result['probability_real']
        prob_fake = result['probability_fake']
        
        print(f"📊 检测结果:")
        print(f"   类型: {'🚨 伪造语音' if is_fake else '✅ 真实语音'}")
        print(f"   置信度: {confidence}%")
        print(f"   真实概率: {prob_real}%")
        print(f"   伪造概率: {prob_fake}%")
        
        # 显示建议
        if confidence > 80:
            print(f"💡 建议: 结果可信度较高")
        elif confidence > 60:
            print(f"💡 建议: 结果可信度中等，建议人工复核")
        else:
            print(f"💡 建议: 结果可信度较低，建议使用其他方法验证")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python predict.py <音频文件路径>")
        print("示例: python predict.py test_audio.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # 创建预测器
    predictor = VoicePredictor()
    
    # 进行预测
    predictor.predict_file(audio_path)

if __name__ == "__main__":
    main()
