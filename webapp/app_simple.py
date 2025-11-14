#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版语音伪造检测Web应用
避免复杂的特征提取和可视化
"""

from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
import joblib
import base64
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import io
from datetime import datetime

# 创建Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制文件大小为16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

class SimpleVoiceDetector:
    """简化的语音检测器"""
    
    def __init__(self, model_path="models/detector.pkl", scaler_path="models/scaler.pkl"):
        """初始化检测器"""
        self.model = None
        self.scaler = None
        self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path, scaler_path):
        """加载训练好的模型"""
        try:
            # 检查文件是否存在
            if not os.path.exists(model_path):
                print(f"模型文件不存在: {model_path}")
                return False
            
            if not os.path.exists(scaler_path):
                print(f"标准化器文件不存在: {scaler_path}")
                return False
            
            # 加载模型
            self.model = joblib.load(model_path)
            print(f"模型加载成功: {model_path}")
            
            # 加载标准化器
            self.scaler = joblib.load(scaler_path)
            print(f"标准化器加载成功: {scaler_path}")
            
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def extract_simple_features(self, audio_path):
        """提取简化的特征"""
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # 计算统计特征（与训练时保持一致）
            features = []
            for i in range(mfcc.shape[0]):
                features.extend([
                    float(np.mean(mfcc[i])),      # 均值
                    float(np.std(mfcc[i])),       # 标准差
                    float(np.min(mfcc[i])),       # 最小值
                    float(np.max(mfcc[i]))        # 最大值
                ])
            
            # 提取频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(audio)
            
            # 添加频谱特征
            features.extend([
                float(np.mean(spectral_centroids)),
                float(np.std(spectral_centroids)),
                float(np.mean(spectral_bandwidth)),
                float(np.std(spectral_bandwidth)),
                float(np.mean(zcr)),
                float(np.std(zcr))
            ])
            
            # 添加节奏特征
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
    
    def predict(self, audio_path):
        """预测音频是否为伪造"""
        if self.model is None or self.scaler is None:
            return {
                'error': '模型未加载，请先训练模型',
                'is_fake': None,
                'confidence': None
            }
        
        try:
            # 提取特征
            features = self.extract_simple_features(audio_path)
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

# 创建检测器实例
detector = SimpleVoiceDetector()

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和检测"""
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        try:
            # 创建上传目录
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # 保存文件
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 进行检测
            result = detector.predict(filepath)
            
            # 清理上传的文件
            os.remove(filepath)
            
            # 返回结果
            return jsonify({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            return jsonify({'error': f'处理文件时出错: {str(e)}'})
    
    return jsonify({'error': '不支持的文件格式'})

@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model is not None,
        'scaler_loaded': detector.scaler is not None
    })

@app.errorhandler(413)
def too_large(e):
    """文件过大错误处理"""
    return jsonify({'error': '文件过大，请选择小于16MB的文件'}), 413

@app.errorhandler(404)
def not_found(e):
    """404错误处理"""
    return redirect(url_for('index'))

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("启动语音伪造检测Web应用")
    print("=" * 50)
    print("访问地址: http://localhost:5000")
    print("健康检查: http://localhost:5000/health")
    print("=" * 50)
    
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000)
