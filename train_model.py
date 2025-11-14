#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练改进版语音伪造检测模型
使用更高级的特征和模型
"""

import os
import sys
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ImprovedVoiceDetector:
    """改进版语音检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = None
        self.feature_names = []
    
    def extract_advanced_features(self, audio_path):
        """提取高级特征"""
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=16000)
            
            features = []
            
            # 1. MFCC特征（更详细）
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            for i in range(mfcc.shape[0]):
                features.extend([
                    float(np.mean(mfcc[i])),
                    float(np.std(mfcc[i])),
                    float(np.min(mfcc[i])),
                    float(np.max(mfcc[i])),
                    float(np.median(mfcc[i])),
                    float(np.percentile(mfcc[i], 25)),
                    float(np.percentile(mfcc[i], 75))
                ])
            
            # 2. 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
            features.extend([
                float(np.mean(spectral_centroids)),
                float(np.std(spectral_centroids)),
                float(np.mean(spectral_bandwidth)),
                float(np.std(spectral_bandwidth)),
                float(np.mean(spectral_rolloff)),
                float(np.std(spectral_rolloff)),
                float(np.mean(spectral_contrast)),
                float(np.std(spectral_contrast))
            ])
            
            # 3. 零交叉率
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.extend([
                float(np.mean(zcr)),
                float(np.std(zcr)),
                float(np.max(zcr)),
                float(np.min(zcr))
            ])
            
            # 4. 色度特征
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend([
                float(np.mean(chroma)),
                float(np.std(chroma)),
                float(np.max(chroma)),
                float(np.min(chroma))
            ])
            
            # 5. 节奏特征
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features.extend([
                float(tempo),
                float(len(beats)),
                float(np.mean(np.diff(beats)) if len(beats) > 1 else 0)
            ])
            
            # 6. 能量特征
            rms = librosa.feature.rms(y=audio)
            features.extend([
                float(np.mean(rms)),
                float(np.std(rms)),
                float(np.max(rms)),
                float(np.min(rms))
            ])
            
            # 7. 音频质量特征
            # 信噪比估计
            signal_power = np.mean(audio**2)
            noise_floor = np.percentile(np.abs(audio), 10)
            snr = 10 * np.log10(signal_power / (noise_floor**2 + 1e-10))
            features.append(float(snr))
            
            # 动态范围
            dynamic_range = np.max(audio) - np.min(audio)
            features.append(float(dynamic_range))
            
            # 8. 谐波和感知特征
            harmonic, percussive = librosa.effects.hpss(audio)
            features.extend([
                float(np.mean(harmonic)),
                float(np.std(harmonic)),
                float(np.mean(percussive)),
                float(np.std(percussive))
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def load_data(self, features_dir="features"):
        """加载训练数据"""
        print("加载训练数据...")
        
        # 加载特征和标签
        X = np.load(os.path.join(features_dir, "X_scaled.npy"))
        y = np.load(os.path.join(features_dir, "y.npy"))
        
        # 加载标准化器
        with open(os.path.join(features_dir, "scaler.pkl"), 'rb') as f:
            self.scaler = joblib.load(f)
        
        print(f"数据加载完成:")
        print(f"   样本数量: {len(X)}")
        print(f"   特征维度: {X.shape[1]}")
        print(f"   真实样本: {np.sum(y == 0)}")
        print(f"   伪造样本: {np.sum(y == 1)}")
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """训练改进的模型"""
        print("开始训练改进的模型...")
        
        # 定义更复杂的模型
        models = {
            'Random Forest (100 trees)': RandomForestClassifier(
                n_estimators=100, 
                max_depth=20, 
                min_samples_split=5,
                random_state=42
            ),
            'Random Forest (200 trees)': RandomForestClassifier(
                n_estimators=200, 
                max_depth=25, 
                min_samples_split=3,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                random_state=42
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'SVM (Polynomial)': SVC(
                kernel='poly',
                degree=3,
                C=1.0,
                probability=True,
                random_state=42
            )
        }
        
        # 训练每个模型
        for name, model in models.items():
            print(f"训练 {name}...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 交叉验证评估
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # 保存模型和分数
            self.models[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{name} 训练完成:")
            print(f"   交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 选择最佳模型
        self.select_best_model()
    
    def select_best_model(self):
        """选择最佳模型"""
        print("\n选择最佳模型...")
        
        best_name = None
        best_score = 0
        
        for name, model_info in self.models.items():
            score = model_info['cv_mean']
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model = self.models[best_name]['model']
        self.best_score = best_score
        
        print(f"最佳模型: {best_name}")
        print(f"   交叉验证准确率: {best_score:.4f}")
    
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        print("\n模型性能评估...")
        
        # 预测
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        accuracy = (y_pred == y_test).mean()
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"测试集性能:")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   AUC: {auc_score:.4f}")
        
        # 详细分类报告
        print("\n详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        return accuracy, auc_score, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path="improved_confusion_matrix.png"):
        """绘制混淆矩阵"""
        try:
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Fake'], 
                       yticklabels=['Real', 'Fake'])
            plt.title('Improved Model Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"混淆矩阵保存成功: {save_path}")
        except Exception as e:
            print(f"保存混淆矩阵失败: {e}")
    
    def plot_roc_curve(self, y_test, y_pred_proba, save_path="improved_roc_curve.png"):
        """绘制ROC曲线"""
        try:
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Improved Model ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"ROC曲线保存成功: {save_path}")
        except Exception as e:
            print(f"保存ROC曲线失败: {e}")
    
    def save_model(self, model_dir="models"):
        """保存训练好的模型"""
        print("保存改进的模型...")
        
        try:
            # 创建模型目录
            os.makedirs(model_dir, exist_ok=True)
            
            # 保存最佳模型
            model_path = os.path.join(model_dir, "improved_detector.pkl")
            joblib.dump(self.best_model, model_path)
            print(f"改进模型保存成功: {model_path}")
            
            # 保存标准化器
            scaler_path = os.path.join(model_dir, "improved_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            print(f"改进标准化器保存成功: {scaler_path}")
            
            # 保存模型信息
            model_info = {
                'model_name': type(self.best_model).__name__,
                'best_score': self.best_score,
                'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_count': 'Advanced features'
            }
            
            info_path = os.path.join(model_dir, "improved_model_info.txt")
            with open(info_path, 'w', encoding='utf-8') as f:
                for key, value in model_info.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"改进模型信息保存成功: {info_path}")
            return True
            
        except Exception as e:
            print(f"保存模型时出错: {e}")
            return False
    
    def train_complete_pipeline(self):
        """完整的训练流程"""
        print("开始改进的模型训练流程")
        print("=" * 50)
        
        # 1. 加载数据
        X, y = self.load_data()
        
        # 2. 划分数据
        print("划分训练集和测试集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"数据划分完成:")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        
        # 3. 训练模型
        self.train_models(X_train, y_train)
        
        # 4. 评估模型
        accuracy, auc_score, y_pred, y_pred_proba = self.evaluate_model(X_test, y_test)
        
        # 5. 绘制图表
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba)
        
        # 6. 保存模型
        if self.save_model():
            print("\n改进模型训练完成！")
            print(f"最终性能: 准确率 {accuracy:.4f}, AUC {auc_score:.4f}")
        else:
            print("\n改进模型训练完成，但保存失败")

def main():
    """主函数"""
    # 创建改进检测器
    detector = ImprovedVoiceDetector()
    
    # 运行完整训练流程
    detector.train_complete_pipeline()

if __name__ == "__main__":
    main()
