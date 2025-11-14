#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音伪造检测模型训练脚本
使用多种机器学习算法训练检测模型
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

class VoiceForgeryDetector:
    """语音伪造检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = None
        self.feature_names = None
    
    def load_data(self, features_dir="features"):
        """
        加载训练数据
        
        参数:
        - features_dir: 特征文件目录
        
        返回:
        - X: 特征矩阵
        - y: 标签向量
        """
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
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        划分训练集和测试集
        
        参数:
        - X: 特征矩阵
        - y: 标签向量
        - test_size: 测试集比例
        - random_state: 随机种子
        
        返回:
        - X_train, X_test, y_train, y_test: 划分后的数据
        """
        print("划分训练集和测试集...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"数据划分完成:")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        训练多个模型
        
        参数:
        - X_train: 训练特征
        - y_train: 训练标签
        """
        print("开始训练多个模型...")
        
        # 定义模型
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
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
        """
        评估模型性能
        
        参数:
        - X_test: 测试特征
        - y_test: 测试标签
        """
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
        print(classification_report(y_test, y_pred, target_names=['真实', '伪造']))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)
        
        # ROC曲线
        self.plot_roc_curve(y_test, y_pred_proba)
        
        return accuracy, auc_score
    
    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，避免显示
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，避免显示
    
    def save_model(self, model_dir="models"):
        """
        保存训练好的模型
        
        参数:
        - model_dir: 模型保存目录
        """
        print("保存模型...")
        
        try:
            # 创建模型目录
            os.makedirs(model_dir, exist_ok=True)
            
            # 保存最佳模型
            model_path = os.path.join(model_dir, "detector.pkl")
            joblib.dump(self.best_model, model_path)
            print(f"模型保存成功: {model_path}")
            
            # 保存标准化器
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            print(f"标准化器保存成功: {scaler_path}")
            
        except PermissionError:
            print("权限错误：无法保存模型文件")
            print("请检查models目录的写入权限，或关闭可能占用文件的程序")
            return False
        except Exception as e:
            print(f"保存模型时出错: {e}")
            return False
        
        # 保存模型信息
        model_info = {
            'model_name': type(self.best_model).__name__,
            'best_score': self.best_score,
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_count': len(self.feature_names) if self.feature_names else 'Unknown'
        }
        
        info_path = os.path.join(model_dir, "model_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"模型保存完成:")
        print(f"   模型文件: {model_path}")
        print(f"   标准化器: {scaler_path}")
        print(f"   模型信息: {info_path}")
        
        return True
    
    def train_complete_pipeline(self):
        """完整的训练流程"""
        print("开始完整的模型训练流程")
        print("=" * 50)
        
        # 1. 加载数据
        X, y = self.load_data()
        
        # 2. 划分数据
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 3. 训练模型
        self.train_models(X_train, y_train)
        
        # 4. 评估模型
        accuracy, auc_score = self.evaluate_model(X_test, y_test)
        
        # 5. 保存模型
        self.save_model()
        
        print("\n模型训练完成！")
        print(f"最终性能: 准确率 {accuracy:.4f}, AUC {auc_score:.4f}")

def main():
    """主函数"""
    # 创建检测器
    detector = VoiceForgeryDetector()
    
    # 运行完整训练流程
    detector.train_complete_pipeline()

if __name__ == "__main__":
    main()
