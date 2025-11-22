"""
Prompt构建模块
负责生成AI分析所需的prompt，包括单条分析和批量分析
"""
from typing import List, Tuple, Optional


def get_system_message() -> str:
    """获取统一的系统消息"""
    return "中文情感分析专家，擅长分析成人用品评论。理解'羞于直言'特点，识别隐晦、委婉表达的真实情感。\n\n能力：识别隐晦负面（'尴尬'、'担心'、'第一次'等）和隐晦正面（'探索'、'不错'、'可以'等）；理解语境差异（'第一次'可能是积极探索或使用困难）。\n\n原则：识别所有负面/正面（直接+隐晦）；关注安全性、舒适度、功能性、隐私；每个负面对应一个改进建议；结合语境判断情感倾向。"


def build_few_shot_section(few_shot_examples: List[Tuple[str, str]]) -> str:
    """
    构建few-shot examples部分
    
    Args:
        few_shot_examples: Few-shot示例列表，格式为 [(text, label), ...]
    
    Returns:
        few-shot部分的字符串
    """
    if not few_shot_examples or len(few_shot_examples) == 0:
        return ""
    
    few_shot_section = "\n【人工复核示例】模糊时参考：\n"
    
    label_cn_map = {
        "strongly_negative": "强烈负面",
        "weakly_negative": "轻微负面",
        "neutral": "中性",
        "weakly_positive": "轻微正面",
        "strongly_positive": "强烈正面",
        "positive": "正面",
        "negative": "负面",
    }
    
    for i, (example_text, example_label) in enumerate(few_shot_examples, 1):
        label_cn = label_cn_map.get(example_label, example_label)
        few_shot_section += f"{i}.{example_text}→{label_cn}({example_label})\n"
    
    few_shot_section += "模糊时参考相似示例标准。\n"
    
    return few_shot_section


def build_product_info_section(sku: Optional[str] = None, product_title: Optional[str] = None) -> str:
    """
    构建商品信息部分
    
    Args:
        sku: 商品SKU（可选）
        product_title: 产品标题（可选）
    
    Returns:
        商品信息部分的字符串
    """
    if not sku and not product_title:
        return ""
    
    product_info_section = "\n【商品信息】"
    if product_title:
        product_info_section += f"标题：{product_title}"
    if sku:
        product_info_section += f"，SKU：{sku}"
    product_info_section += "\n"
    
    return product_info_section


def get_analysis_prompt(
    text: str, 
    few_shot_examples: List[Tuple[str, str]] = None, 
    sku: str = None, 
    product_title: str = None
) -> str:
    """
    获取单条分析的prompt
    
    Args:
        text: 待分析的文本
        few_shot_examples: Few-shot示例列表，格式为 [(text, label), ...]
        sku: 商品SKU（可选）
        product_title: 产品标题（可选）
    
    Returns:
        完整的分析prompt
    """
    few_shot_section = build_few_shot_section(few_shot_examples) if few_shot_examples else ""
    product_info_section = build_product_info_section(sku, product_title)
    
    return f"""分析评论情感，提取关键词。{few_shot_section}{product_info_section}评论：{text}

【识别规则】
负面：隐晦("尴尬"、"担心"、"第一次"→体验不佳/使用困难)、间接("可以改进"、"一般"、"效果不明显")、直接("质量差"、"不满意")
正面：隐晦("探索"、"不错"、"可以"→较强正面)、明确("超出预期"、"推荐")、体验("舒适"、"效果")
语境："第一次"可能积极或困难；"一般"可能委婉负面；"可以/还行"可能较强正面

【输出JSON】
{{
    "sentiment": "strongly_negative/weakly_negative/neutral/weakly_positive/strongly_positive",
    "confidence": 0.0-1.0,
    "probabilities": {{"strongly_negative": 0.0-1.0, "weakly_negative": 0.0-1.0, "neutral": 0.0-1.0, "weakly_positive": 0.0-1.0, "strongly_positive": 0.0-1.0}},
    "reason": "分析原因",
    "keywords": ["关键词1", ...],
    "negative_parts": ["负面部分1", ...],
    "suggestions": ["改进建议1", ...],
    "confidence_calculation": "计算说明"
}}

【要求】
1. sentiment：五分类，混合按主要倾向
2. confidence：max(probabilities)，微调(直接>隐晦，单一>混合)，模糊时(有示例0.75-0.9，无示例0.5-0.7)
3. probabilities：五值之和≈1.0
4. keywords：5-10个
5. negative_parts：所有负面(直接+隐含)，无则[]
6. suggestions：每负面对应一建议，无则[]
7. 只返回JSON"""


def get_batch_analysis_prompt(
    texts: List[str], 
    few_shot_examples: List[Tuple[str, str]] = None, 
    skus: List[str] = None, 
    product_titles: List[str] = None
) -> str:
    """
    获取批量分析的prompt
    
    Args:
        texts: 评论文本列表
        few_shot_examples: Few-shot示例列表
        skus: 商品SKU列表（可选，与texts一一对应）
        product_titles: 产品标题列表（可选，与texts一一对应）
    
    Returns:
        完整的批量分析prompt
    """
    few_shot_section = build_few_shot_section(few_shot_examples) if few_shot_examples else ""
    
    # 构建批量评论部分（包含商品信息）
    comments_section = ""
    for i, text in enumerate(texts):
        sku = skus[i] if skus and i < len(skus) else None
        title = product_titles[i] if product_titles and i < len(product_titles) else None
        
        # 格式：评论【1】：xxxx，评论【2】：xxxx
        comment_info = f"评论【{i+1}】："
        if title:
            comment_info += f"产品标题：{title}，"
        if sku:
            comment_info += f"SKU：{sku}，"
        comment_info += f"{text}"
        comments_section += comment_info + "\n"
    
    product_info_note = ""
    if (skus and any(skus)) or (product_titles and any(product_titles)):
        product_info_note = "\n提示：结合产品标题和SKU理解评论上下文。\n"
    
    return f"""批量分析评论情感，提取关键词。{few_shot_section}{product_info_note}【评论】\n{comments_section}【识别】负面：隐晦("尴尬"、"担心"、"第一次")、间接("可以改进"、"一般")、直接("质量差")；正面：隐晦("探索"、"不错"→较强)、明确("超出预期")；语境："第一次"可能积极/困难，"一般"可能委婉负面，"可以/还行"可能较强正面

【输出JSON数组】
[
    {{
        "comment_index": 1,
        "sentiment": "strongly_negative/weakly_negative/neutral/weakly_positive/strongly_positive",
        "confidence": 0.0-1.0,
        "probabilities": {{"strongly_negative": 0.0-1.0, "weakly_negative": 0.0-1.0, "neutral": 0.0-1.0, "weakly_positive": 0.0-1.0, "strongly_positive": 0.0-1.0}},
        "reason": "分析原因",
        "keywords": ["关键词1", ...],
        "negative_parts": ["负面部分1", ...],
        "suggestions": ["改进建议1", ...],
        "confidence_calculation": "计算说明"
    }}
]

【要求】
1. 数组长度=评论数，comment_index从1开始
2. sentiment：五分类，混合按主要倾向
3. confidence：max(probabilities)，微调(直接>隐晦，单一>混合)，模糊(有示例0.75-0.9，无示例0.5-0.7)
4. probabilities：五值之和≈1.0
5. keywords：5-10个
6. negative_parts：所有负面，无则[]
7. suggestions：每负面对应一建议，无则[]
8. 只返回JSON数组"""

