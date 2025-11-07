"""
模型评估脚本
用于计算情感分析模型的准确率、精确率、召回率等指标
"""
import json
import os
import sys
from ml_analyzer import create_ml_analyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

def evaluate_model():
    """评估ML模型的准确率"""
    
    # 加载训练数据
    training_data_file = 'training_data.json'
    if not os.path.exists(training_data_file):
        print(f"错误: {training_data_file} 文件不存在")
        return None
    
    with open(training_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("错误: 训练数据为空")
        return None
    
    # 准备数据
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    print("=" * 70)
    print("模型评估报告")
    print("=" * 70)
    print(f"总数据量: {len(data)}")
    print(f"数据分布:")
    from collections import Counter
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        print(f"  {label}: {count}条 ({count/len(data)*100:.1f}%)")
    print()
    
    # 确保有足够的数据进行划分
    if len(data) < 10:
        print("警告: 数据量太少，无法进行训练/测试集划分")
        print("将使用全部数据进行训练，无法计算测试集准确率")
        # 使用全部数据训练
        training_data = [(text, label) for text, label in zip(texts, labels)]
        analyzer = create_ml_analyzer(training_data)
        print("模型训练完成")
        return None
    
    # 划分训练集和测试集（80%训练，20%测试）
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        # 如果分层采样失败（某些类别样本太少），使用普通划分
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
    
    print(f"训练集大小: {len(X_train)} ({len(X_train)/len(data)*100:.1f}%)")
    print(f"测试集大小: {len(X_test)} ({len(X_test)/len(data)*100:.1f}%)")
    print()
    
    # 训练模型
    print("正在训练模型...")
    training_data = [(text, label) for text, label in zip(X_train, y_train)]
    analyzer = create_ml_analyzer(training_data)
    print("模型训练完成")
    print()
    
    # 在测试集上预测
    print("正在测试模型...")
    y_pred = []
    for text in X_test:
        try:
            result = analyzer.analyze_sentiment(text)
            y_pred.append(result['sentiment'])
        except Exception as e:
            print(f"警告: 分析文本时出错: {e}")
            # 使用默认值
            y_pred.append('neutral')
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=['positive', 'negative', 'neutral'], average=None, zero_division=0
    )
    
    # 打印详细报告
    print("=" * 70)
    print("分类报告:")
    print("=" * 70)
    print(classification_report(y_test, y_pred, labels=['positive', 'negative', 'neutral'], zero_division=0))
    
    print("\n" + "=" * 70)
    print(f"总体准确率: {accuracy * 100:.2f}%")
    print("=" * 70)
    
    # 按类别统计
    print("\n各类别指标:")
    labels_list = ['positive', 'negative', 'neutral']
    for i, label in enumerate(labels_list):
        if i < len(precision):
            print(f"\n{label}:")
            print(f"  精确率 (Precision): {precision[i]:.4f}")
            print(f"  召回率 (Recall): {recall[i]:.4f}")
            print(f"  F1分数: {f1[i]:.4f}")
            print(f"  样本数: {int(support[i])}")
    
    # 返回结果
    result = {
        'accuracy': float(accuracy),
        'precision': {
            'positive': float(precision[0]) if len(precision) > 0 else 0,
            'negative': float(precision[1]) if len(precision) > 1 else 0,
            'neutral': float(precision[2]) if len(precision) > 2 else 0,
        },
        'recall': {
            'positive': float(recall[0]) if len(recall) > 0 else 0,
            'negative': float(recall[1]) if len(recall) > 1 else 0,
            'neutral': float(recall[2]) if len(recall) > 2 else 0,
        },
        'f1': {
            'positive': float(f1[0]) if len(f1) > 0 else 0,
            'negative': float(f1[1]) if len(f1) > 1 else 0,
            'neutral': float(f1[2]) if len(f1) > 2 else 0,
        },
        'support': {
            'positive': int(support[0]) if len(support) > 0 else 0,
            'negative': int(support[1]) if len(support) > 1 else 0,
            'neutral': int(support[2]) if len(support) > 2 else 0,
        }
    }
    
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)
    
    return result

if __name__ == '__main__':
    try:
        result = evaluate_model()
        if result:
            print("\n评估结果摘要:")
            print(f"准确率: {result['accuracy']*100:.2f}%")
            print(f"精确率 - 正面: {result['precision']['positive']:.4f}, 负面: {result['precision']['negative']:.4f}, 中性: {result['precision']['neutral']:.4f}")
            print(f"召回率 - 正面: {result['recall']['positive']:.4f}, 负面: {result['recall']['negative']:.4f}, 中性: {result['recall']['neutral']:.4f}")
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

