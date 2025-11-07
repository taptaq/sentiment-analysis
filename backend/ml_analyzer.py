"""
机器学习分析模块
负责情感分析的ML模型训练和预测
"""
import jieba
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import os


class MLAnalyzer:
    """机器学习情感分析器"""
    
    def __init__(self, training_data=None):
        """
        初始化ML分析器
        
        Args:
            training_data: 训练数据列表，格式为 [(text, label), ...]
        """
        self.training_data = training_data or []
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        # 初始化jieba
        jieba.initialize()
    
    def train(self, training_data=None):
        """
        训练情感分析模型
        
        Args:
            training_data: 训练数据列表，格式为 [(text, label), ...]
                          如果为None，使用self.training_data
        """
        if training_data is not None:
            self.training_data = training_data
        
        if not self.training_data:
            raise ValueError("训练数据不能为空")
        
        texts = [item[0] for item in self.training_data]
        labels = [item[1] for item in self.training_data]
        
        # 文本预处理：分词
        processed_texts = [' '.join(jieba.cut(text)) for text in texts]
        
        # TF-IDF向量化 - 调整参数以适应短文本
        # min_df=1: 允许所有词，即使是只出现一次的
        # ngram_range=(1,2): 使用单字和双字组合，更好捕捉短文本特征
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=1,
            ngram_range=(1, 2),
            token_pattern=r'\b\w+\b'
        )
        X = self.vectorizer.fit_transform(processed_texts)
        
        # 训练朴素贝叶斯分类器
        # alpha=1.0: 使用拉普拉斯平滑，避免零概率问题
        self.model = MultinomialNB(alpha=1.0)
        self.model.fit(X, labels)
        
        self.is_trained = True
        
        return self.model, self.vectorizer
    
    def analyze_sentiment(self, text):
        """
        使用机器学习模型分析文本情感
        
        Args:
            text: 待分析的文本
            
        Returns:
            {
                "sentiment": "positive/negative/neutral",
                "confidence": 0.0-1.0,
                "probabilities": {
                    "positive": 0.0-1.0,
                    "negative": 0.0-1.0,
                    "neutral": 0.0-1.0
                },
                "method": "ml"
            }
        """
        if not self.is_trained or self.model is None or self.vectorizer is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # 文本预处理：分词
        processed_text = ' '.join(jieba.cut(text))
        X = self.vectorizer.transform([processed_text])
        
        # 预测
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # 获取情感标签和概率
        labels = self.model.classes_
        prob_dict = {label: float(prob) for label, prob in zip(labels, probabilities)}
        
        max_prob = float(max(probabilities))
        return {
            "sentiment": prediction,
            "confidence": max_prob,
            "probabilities": prob_dict,
            "confidence_calculation": f"基于朴素贝叶斯模型预测概率的最大值：confidence = max(probabilities) = max({', '.join([f'{k}={v:.3f}' for k, v in prob_dict.items()])}) = {max_prob:.3f}",
            "method": "ml"
        }
    
    def extract_keywords(self, text, topK=10):
        """
        使用TF-IDF提取关键词
        
        Args:
            text: 待提取关键词的文本
            topK: 返回关键词数量
            
        Returns:
            [{"word": "关键词", "weight": 权重}, ...]
        """
        keywords = jieba.analyse.extract_tags(text, topK=topK, withWeight=True)
        return [{"word": word, "weight": float(weight)} for word, weight in keywords]
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "is_trained": self.is_trained,
            "training_data_count": len(self.training_data),
            "model_type": "MultinomialNB",
            "vectorizer_type": "TfidfVectorizer"
        }


def create_ml_analyzer(training_data):
    """
    创建并训练ML分析器的便捷函数
    
    Args:
        training_data: 训练数据列表，格式为 [(text, label), ...]
        
    Returns:
        训练好的MLAnalyzer实例
    """
    analyzer = MLAnalyzer(training_data)
    analyzer.train()
    return analyzer

