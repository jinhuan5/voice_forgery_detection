#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版语音伪造检测Web应用
使用更高级的特征和模型
"""

import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, render_template, jsonify, send_from_directory
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ImprovedVoiceDetector:
    """改进版语音检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            model_path = os.path.join('..', 'models', 'improved_detector.pkl')
            scaler_path = os.path.join('..', 'models', 'improved_scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print("改进模型加载成功")
            else:
                print("改进模型文件不存在，使用基础模型")
                # 回退到基础模型
                model_path = os.path.join('..', 'models', 'detector.pkl')
                scaler_path = os.path.join('..', 'models', 'scaler.pkl')
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    print("基础模型加载成功")
                else:
                    print("模型文件不存在")
                    self.model = None
                    self.scaler = None
                    
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
            self.scaler = None
    
    def extract_advanced_features(self, audio_path):
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
    
    def predict(self, audio_path):
        """预测音频是否为伪造"""
        if self.model is None or self.scaler is None:
            return None, "模型未加载"
        
        try:
            # 提取特征
            features = self.extract_advanced_features(audio_path)
            if features is None:
                return None, "特征提取失败"
            
            # 标准化特征
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # 预测
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # 返回结果
            is_fake = bool(prediction)
            confidence = float(max(probability))
            real_prob = float(probability[0])
            fake_prob = float(probability[1])
            
            return {
                'is_fake': is_fake,
                'confidence': confidence,
                'real_probability': real_prob,
                'fake_probability': fake_prob
            }, None
            
        except Exception as e:
            return None, f"预测失败: {e}"

# 创建检测器实例
detector = ImprovedVoiceDetector()

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_advanced_spectrogram(audio_path):
    """创建高级频谱图"""
    try:
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Advanced Audio Analysis', fontsize=16)
        
        # 1. 时域波形
        time = np.linspace(0, len(audio)/sr, len(audio))
        axes[0, 0].plot(time, audio)
        axes[0, 0].set_title('Time Domain Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # 2. 频谱图
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = axes[0, 1].imshow(D, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title('Spectrogram')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Frequency')
        plt.colorbar(img, ax=axes[0, 1])
        
        # 3. MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        img2 = axes[1, 0].imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
        axes[1, 0].set_title('MFCC Features')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('MFCC')
        plt.colorbar(img2, ax=axes[1, 0])
        
        # 4. 色度特征
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        img3 = axes[1, 1].imshow(chroma, aspect='auto', origin='lower', cmap='viridis')
        axes[1, 1].set_title('Chroma Features')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Chroma')
        plt.colorbar(img3, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # 转换为base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
        
    except Exception as e:
        print(f"创建频谱图失败: {e}")
        return None

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和检测"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'}), 400
        
        # 保存文件
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 检测音频
        result, error = detector.predict(filepath)
        if error:
            return jsonify({'error': error}), 500
        
        # 创建频谱图
        spectrogram = create_advanced_spectrogram(filepath)
        
        # 清理上传的文件
        try:
            os.remove(filepath)
        except:
            pass
        
        # 返回结果
        return jsonify({
            'success': True,
            'result': result,
            'spectrogram': spectrogram
        })
        
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

if __name__ == '__main__':
    print("启动改进版语音伪造检测Web应用...")
    print("访问地址: http://localhost:5000")
    print("健康检查: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)
