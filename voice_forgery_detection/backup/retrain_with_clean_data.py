#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用清理后的数据重新训练模型
"""

import os
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
import glob

class CleanDataRetrainer:
    """使用清理后的数据重新训练模型"""
    
    def __init__(self):
        """初始化训练器"""
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = None
        self.feature_names = []
    
    def extract_features(self, audio_path):
        """提取特征（与训练时保持一致）"""
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=16000)
            
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
            
            # 确保特征数量为52个
            if len(features) > 52:
                features = features[:52]
            elif len(features) < 52:
                features.extend([0.0] * (52 - len(features)))
            
            return np.array(features)
            
        except Exception as e:
            print(f"特征提取失败 {audio_path}: {e}")
            return None
    
    def load_clean_data(self, data_dir="data"):
        """加载清理后的数据"""
        print("加载清理后的数据...")
        
        real_features = []
        fake_features = []
        
        # 加载真实语音数据
        real_dir = os.path.join(data_dir, "real")
        if os.path.exists(real_dir):
            real_files = glob.glob(os.path.join(real_dir, "*.wav")) + glob.glob(os.path.join(real_dir, "*.mp3"))
            print(f"找到 {len(real_files)} 个真实语音文件")
            
            for i, file_path in enumerate(real_files):
                print(f"处理真实语音 {i+1}/{len(real_files)}: {os.path.basename(file_path)}")
                features = self.extract_features(file_path)
                if features is not None:
                    real_features.append(features)
        
        # 加载伪造语音数据
        fake_dir = os.path.join(data_dir, "fake")
        if os.path.exists(fake_dir):
            fake_files = glob.glob(os.path.join(fake_dir, "*.wav")) + glob.glob(os.path.join(fake_dir, "*.mp3"))
            print(f"找到 {len(fake_files)} 个伪造语音文件")
            
            for i, file_path in enumerate(fake_files):
                print(f"处理伪造语音 {i+1}/{len(fake_files)}: {os.path.basename(file_path)}")
                features = self.extract_features(file_path)
                if features is not None:
                    fake_features.append(features)
        
        if len(real_features) == 0 or len(fake_features) == 0:
            print("错误：没有找到有效的音频文件")
            return None, None
        
        # 合并特征和标签
        print("\n合并特征和标签...")
        
        # 创建标签
        real_labels = np.zeros(len(real_features))  # 真实语音标签为0
        fake_labels = np.ones(len(fake_features))   # 伪造语音标签为1
        
        # 合并特征和标签
        X = np.vstack([real_features, fake_features])
        y = np.hstack([real_labels, fake_labels])
        
        print(f"数据统计:")
        print(f"  总样本数: {len(X)}")
        print(f"  真实语音: {len(real_features)}")
        print(f"  伪造语音: {len(fake_features)}")
        print(f"  特征维度: {X.shape[1]}")
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """训练模型"""
        print("开始训练模型...")
        
        # 定义模型
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=20, 
                min_samples_split=5,
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
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
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
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path="clean_confusion_matrix.png"):
        """绘制混淆矩阵"""
        try:
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Fake'], 
                       yticklabels=['Real', 'Fake'])
            plt.title('Clean Data Model Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"混淆矩阵保存成功: {save_path}")
        except Exception as e:
            print(f"保存混淆矩阵失败: {e}")
    
    def plot_roc_curve(self, y_test, y_pred_proba, save_path="clean_roc_curve.png"):
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
            plt.title('Clean Data Model ROC Curve')
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
        print("保存清理后的模型...")
        
        try:
            # 创建模型目录
            os.makedirs(model_dir, exist_ok=True)
            
            # 保存最佳模型
            model_path = os.path.join(model_dir, "clean_detector.pkl")
            joblib.dump(self.best_model, model_path)
            print(f"清理模型保存成功: {model_path}")
            
            # 保存标准化器
            scaler_path = os.path.join(model_dir, "clean_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            print(f"清理标准化器保存成功: {scaler_path}")
            
            # 保存模型信息
            model_info = {
                'model_name': type(self.best_model).__name__,
                'best_score': self.best_score,
                'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_count': 'Clean data features'
            }
            
            info_path = os.path.join(model_dir, "clean_model_info.txt")
            with open(info_path, 'w', encoding='utf-8') as f:
                for key, value in model_info.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"清理模型信息保存成功: {info_path}")
            return True
            
        except Exception as e:
            print(f"保存模型时出错: {e}")
            return False
    
    def train_complete_pipeline(self):
        """完整的训练流程"""
        print("开始使用清理后的数据重新训练模型")
        print("=" * 60)
        
        # 1. 加载清理后的数据
        X, y = self.load_clean_data()
        if X is None or y is None:
            print("数据加载失败")
            return False
        
        # 2. 特征标准化
        print("\n特征标准化...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 3. 划分数据
        print("划分训练集和测试集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"数据划分完成:")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        
        # 4. 训练模型
        self.train_models(X_train, y_train)
        
        # 5. 评估模型
        accuracy, auc_score, y_pred, y_pred_proba = self.evaluate_model(X_test, y_test)
        
        # 6. 绘制图表
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba)
        
        # 7. 保存模型
        if self.save_model():
            print("\n清理后的模型训练完成！")
            print(f"最终性能: 准确率 {accuracy:.4f}, AUC {auc_score:.4f}")
            return True
        else:
            print("\n清理后的模型训练完成，但保存失败")
            return False

def main():
    """主函数"""
    # 创建清理数据训练器
    trainer = CleanDataRetrainer()
    
    # 运行完整训练流程
    success = trainer.train_complete_pipeline()
    
    if success:
        print("\n🎉 使用清理后的数据重新训练完成！")
        print("现在可以使用更准确的模型进行检测了")
    else:
        print("\n❌ 重新训练失败")

if __name__ == "__main__":
    main()
