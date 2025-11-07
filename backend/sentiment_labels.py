"""
情感标签定义模块
定义五分类情感标签体系（针对成人用品场景优化）
"""
# 五分类情感标签
SENTIMENT_LABELS = {
    'strongly_negative': '强烈负面',
    'weakly_negative': '轻微负面',
    'neutral': '中性',
    'weakly_positive': '轻微正面',
    'strongly_positive': '强烈正面'
}

# 标签列表（按情感强度排序）
SENTIMENT_LABEL_LIST = [
    'strongly_negative',
    'weakly_negative',
    'neutral',
    'weakly_positive',
    'strongly_positive'
]

# 标签到中文的映射
LABEL_TO_CN = {
    'strongly_negative': '强烈负面',
    'weakly_negative': '轻微负面',
    'neutral': '中性',
    'weakly_positive': '轻微正面',
    'strongly_positive': '强烈正面'
}

# 中文到标签的映射
CN_TO_LABEL = {
    '强烈负面': 'strongly_negative',
    '轻微负面': 'weakly_negative',
    '中性': 'neutral',
    '轻微正面': 'weakly_positive',
    '强烈正面': 'strongly_positive'
}

# 标签到颜色的映射（用于前端显示）
LABEL_TO_COLOR = {
    'strongly_negative': '#d32f2f',  # 深红色
    'weakly_negative': '#f44336',    # 红色
    'neutral': '#ff9800',            # 橙色
    'weakly_positive': '#4caf50',    # 绿色
    'strongly_positive': '#2e7d32'   # 深绿色
}

# 旧标签到新标签的映射（用于兼容性转换）
OLD_TO_NEW_LABEL = {
    'negative': 'weakly_negative',  # 默认映射为轻微负面，可根据实际情况调整
    'neutral': 'neutral',
    'positive': 'weakly_positive'   # 默认映射为轻微正面，可根据实际情况调整
}

def convert_old_label_to_new(old_label: str, text: str = '') -> str:
    """
    将旧的三分类标签转换为新的五分类标签
    
    Args:
        old_label: 旧标签 (positive/negative/neutral)
        text: 评论文本（可选，用于更精确的转换）
    
    Returns:
        新的五分类标签
    """
    # 基础映射
    base_mapping = OLD_TO_NEW_LABEL.get(old_label, 'neutral')
    
    # 如果提供了文本，可以根据文本内容进行更精确的转换
    if text:
        text_lower = text.lower()
        
        # 强烈负面关键词
        strong_negative_keywords = ['非常不满意', '很不满意', '太差', '垃圾', '很差劲', '失望', '非常失望', '差评', '必须差评', '退货', '不推荐', '不值得']
        # 强烈正面关键词
        strong_positive_keywords = ['非常满意', '很满意', '太棒了', '非常好', '很棒', '强烈推荐', '五星好评', '物超所值', '超出预期', '非常喜欢']
        
        if old_label == 'negative':
            # 检查是否是强烈负面
            if any(keyword in text_lower for keyword in strong_negative_keywords):
                return 'strongly_negative'
            else:
                return 'weakly_negative'
        elif old_label == 'positive':
            # 检查是否是强烈正面
            if any(keyword in text_lower for keyword in strong_positive_keywords):
                return 'strongly_positive'
            else:
                return 'weakly_positive'
    
    return base_mapping

def is_valid_sentiment_label(label: str) -> bool:
    """检查标签是否有效"""
    return label in SENTIMENT_LABEL_LIST

def get_label_cn(label: str) -> str:
    """获取标签的中文名称"""
    return LABEL_TO_CN.get(label, label)

def get_label_color(label: str) -> str:
    """获取标签对应的颜色"""
    return LABEL_TO_COLOR.get(label, '#666666')

