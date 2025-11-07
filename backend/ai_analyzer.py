"""
AI智能分析模块
支持多种LLM API（OpenAI、DeepSeek等）
提供更智能的情感分析和关键词提取
支持few-shot learning，利用人工复核数据进行数据增强
支持五分类情感分析（强烈负面、轻微负面、中性、轻微正面、强烈正面）
"""
import os
import json
import requests
import random
from typing import Dict, List, Optional, Tuple

# 导入情感标签定义
try:
    from sentiment_labels import SENTIMENT_LABELS_AVAILABLE, convert_old_label_to_new
    SENTIMENT_LABELS_AVAILABLE = True
except ImportError:
    SENTIMENT_LABELS_AVAILABLE = False
    def convert_old_label_to_new(label, text=''): return label

class AIAnalyzer:
    """AI智能分析器"""
    
    def __init__(self):
        # 配置API密钥（从环境变量读取）
        self.openai_api_key = os.getenv('OPENAI_API_KEY', 'sk-hyY48dpf9mN09okwktrpeDvvOMzKM8I9DcmbAmZwXe6LNBOj')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.chatanywhere.tech/v1')
        
        # DeepSeek配置
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY', 'sk-f6f33f981ae343c3bbca6564f169fa1d')
        self.deepseek_base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
        self.deepseek_model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
        
        # 分析模式：'auto'（自动选择）、'ai'（仅AI）、'ml'（仅ML）、'hybrid'（混合）
        self.analysis_mode = os.getenv('ANALYSIS_MODE', 'auto').lower()
        
        # 是否启用AI分析
        self.ai_enabled = bool(self.openai_api_key or self.deepseek_api_key)
        
        # Few-shot learning配置
        self.use_few_shot = os.getenv('AI_USE_FEW_SHOT', 'true').lower() == 'true'
        self.few_shot_count = int(os.getenv('AI_FEW_SHOT_COUNT', '3'))  # 默认使用3个示例
        self.training_data_file = os.getenv('TRAINING_DATA_FILE', 'training_data.json')
    
    def _load_few_shot_examples(self, current_text: str = "", count: int = 3) -> List[Tuple[str, str]]:
        """
        从训练数据中加载few-shot examples
        
        Args:
            current_text: 当前要分析的文本（用于相似度筛选，可选）
            count: 返回的示例数量
        
        Returns:
            示例列表，格式为 [(text, label), ...]
        """
        if not self.use_few_shot:
            return []
        
        try:
            if os.path.exists(self.training_data_file):
                with open(self.training_data_file, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                
                # 转换为元组列表
                examples = [(item['text'], item['label']) for item in training_data if item.get('text') and item.get('label')]
                
                if not examples:
                    return []
                
                # 如果数据量大于需要的数量，进行筛选
                if len(examples) > count:
                    # 优先选择与当前文本相似的数据（简单实现：包含相同关键词）
                    # 如果当前文本为空或没有相似数据，则随机选择
                    if current_text:
                        # 简单的相似度筛选：查找包含相同关键词的示例
                        current_words = set(current_text)
                        scored_examples = []
                        for text, label in examples:
                            text_words = set(text)
                            # 计算交集大小作为相似度
                            similarity = len(current_words & text_words)
                            scored_examples.append((similarity, text, label))
                        
                        # 按相似度排序，选择前count个
                        scored_examples.sort(reverse=True, key=lambda x: x[0])
                        selected = [(text, label) for _, text, label in scored_examples[:count]]
                        
                        # 如果相似度高的不够，用随机数据补充
                        if len(selected) < count:
                            remaining = [(text, label) for _, text, label in scored_examples[count:]]
                            selected.extend(random.sample(remaining, min(count - len(selected), len(remaining))))
                        
                        return selected
                    else:
                        # 随机选择
                        return random.sample(examples, count)
                else:
                    return examples
            else:
                return []
        except Exception as e:
            print(f"[Few-shot] 加载示例数据失败: {e}")
            return []
    
    def _get_system_message(self) -> str:
        """获取统一的系统消息"""
        return "你是一个专业的中文情感分析专家，特别擅长分析成人用品等特殊产品的评论。你深刻理解该行业用户'羞于直言'的评论特点，能够精准识别隐晦、委婉、间接表达中的真实情感。\n\n核心能力：\n1. 识别羞于直言的隐晦表达：'尴尬'、'私密'、'难以启齿'、'担心'等语境下的真实情感\n2. 识别隐晦的积极体验：'探索'、'愉悦'、'新奇'、'超出预期'等积极表达的隐晦形式\n3. 理解语境差异：同样的词汇在不同语境下的不同含义（如'第一次'可能是探索性积极，也可能是使用困难）\n4. 透过委婉表达识别真实情感：'一般'、'可以'、'还行'在成人用品语境下的真实含义\n\n分析原则：\n- 必须识别所有负面内容，包括直接表达和隐晦表达\n- 必须识别所有正面内容，包括隐晦的积极体验表达\n- 特别关注安全性、舒适度、功能性、隐私卫生等方面\n- 为每个负面部分生成对应的改进建议\n- 结合语境和行业特点，准确判断真实情感倾向"
    
    def _get_analysis_prompt(self, text: str, few_shot_examples: List[Tuple[str, str]] = None) -> str:
        """
        获取统一的分析prompt
        
        Args:
            text: 待分析的文本
            few_shot_examples: Few-shot示例列表，格式为 [(text, label), ...]
        """
        # 构建few-shot examples部分
        few_shot_section = ""
        if few_shot_examples and len(few_shot_examples) > 0:
            few_shot_section = "\n\n【重要参考：人工复核数据示例】\n"
            few_shot_section += "以下是一些已经人工复核并标注好的评论示例，这些是经过人工验证的准确标注数据。\n"
            few_shot_section += "**特别重要**：当你对评论的情感判断感到模糊、不确定或存在歧义时，必须优先参考这些示例的分析标准和标注结果。\n\n"
            
            for i, (example_text, example_label) in enumerate(few_shot_examples, 1):
                # 支持五分类标签显示
                label_cn_map = {
                    "strongly_negative": "强烈负面",
                    "weakly_negative": "轻微负面",
                    "neutral": "中性",
                    "weakly_positive": "轻微正面",
                    "strongly_positive": "强烈正面",
                    "positive": "正面",  # 兼容旧标签
                    "negative": "负面",  # 兼容旧标签
                }
                label_cn = label_cn_map.get(example_label, example_label)
                few_shot_section += f"示例{i}：\n"
                few_shot_section += f"评论：{example_text}\n"
                few_shot_section += f"人工复核标注：{label_cn} ({example_label})\n\n"
            
            few_shot_section += "【如何使用这些示例】\n"
            few_shot_section += "1. 如果当前评论与某个示例在表达方式、用词、语境上相似，应参考该示例的标注结果\n"
            few_shot_section += "2. 如果当前评论的情感倾向不明显或存在歧义，优先参考相似示例的标注标准\n"
            few_shot_section += "3. 如果当前评论的表达方式与示例中的隐晦表达类似，应按照示例的标注逻辑进行判断\n"
            few_shot_section += "4. 当你的判断与示例标准不一致时，应重新审视并调整，确保与人工复核的标准保持一致\n"
            few_shot_section += "5. 这些示例代表了人工复核的准确标准，当判断模糊时，必须依赖这些示例而非自己的推测\n\n"
        
        return f"""请仔细分析以下商品评论的情感倾向，并提取关键词。深刻理解成人用品行业用户"羞于直言"的评论特点，精准识别隐晦表达的真实情感。{few_shot_section}

评论内容：{text}

【核心分析原则】
深刻理解成人用品行业用户"羞于直言"的评论特点，需要从隐晦、委婉、间接的表达中识别真实情感。

【负面情感识别 - 隐晦表达】
必须识别评论中的所有负面部分，特别注意以下隐晦表达：
1. 羞于直言的负面表达：
   * "尴尬"、"有点尴尬"、"不太自然"、"感觉怪怪的" → 可能表示使用体验不佳、设计不合理
   * "私密性"、"隐私"、"包装"、"担心被发现" → 可能表示包装不够私密、隐私保护不足
   * "难以启齿"、"不好意思说"、"不太方便" → 可能表示使用不便、体验不佳
   * "第一次"、"新手"、"不太懂"、"摸索" → 可能表示产品说明不足、使用困难
   * "有点担心"、"害怕"、"紧张" → 可能表示安全性担忧、使用焦虑

2. 间接负面表达：
   * "可以再改进"、"有待提升"、"有点问题"、"还可以更好"、"有点遗憾"、"一般"、"还好"、"凑合"
   * "体验一般"、"效果不明显"、"不太理想"、"没有想象中好"
   * "对比中的负面"（如"虽然外观不错，但质量一般"中的"质量一般"）
   * "条件性负面"（如"如果噪音再小点就好了"中的"噪音大"）

3. 直接负面表达：
   * "质量差"、"噪音大"、"不满意"、"材质不好"、"不舒服"、"有问题"

【正面情感识别 - 隐晦表达】
特别注意识别隐晦的积极体验表达：
1. 探索性积极表达：
   * "探索"、"尝试"、"体验"、"发现" → 可能表示对产品的积极探索和接受
   * "新奇"、"有趣"、"特别" → 可能表示产品带来的新鲜感和积极体验

2. 愉悦性隐晦表达：
   * "愉悦"、"满意"、"不错"、"可以"、"还行" → 在成人用品语境下可能是较强的正面评价
   * "超出预期"、"比想象中好"、"值得" → 明确的积极评价
   * "推荐"、"会回购"、"下次还买" → 强烈的正面信号

3. 体验性积极表达：
   * "体验感"、"感觉"、"效果" → 结合上下文判断，可能是正面体验
   * "舒适"、"柔软"、"温和" → 材质和体验的正面评价

【成人用品特殊关注点】
1. 安全性相关问题：材质安全、使用安全、卫生安全
2. 舒适度问题：材质舒适度、尺寸合适度、使用体验舒适度
3. 功能性问题：效果、性能、耐用性、使用体验
4. 隐私和卫生问题：包装、清洁、存储

【情感判断规则】
1. 结合语境判断：同样的词汇在不同语境下可能有不同含义
   * "第一次"可能是探索性积极，也可能是使用困难的负面
   * "一般"在成人用品语境下可能是委婉的负面表达
   * "可以"、"还行"在羞于直言的语境下可能是较强的正面评价

2. 识别真实情感：透过委婉表达识别真实情感
   * "有点尴尬" → 负面（使用体验不佳）
   * "探索了一下" → 正面（积极尝试和接受）
   * "私密性不错" → 正面（隐私保护满意）
   * "包装有点担心" → 负面（隐私担忧）

3. 整体情感判断：综合考虑所有表达，识别主要情感倾向

【输出要求】
- 每条负面部分都应该对应一个具体的改进建议
- 即使评论整体偏正面或中性，只要有负面内容，就必须识别出来
- 特别注意识别隐晦的正面和负面表达，不要被委婉的措辞误导

【判断模糊时的处理原则】
当遇到以下情况时，说明判断存在模糊性，必须参考上述人工复核数据示例：
1. 评论表达隐晦，难以直接判断情感倾向
2. 评论中存在混合情感，不确定主要倾向
3. 某些关键词在不同语境下可能有不同含义，需要参考示例中的标注标准
4. 置信度计算时，如果probabilities分布较为均匀（如三个值都在0.3-0.4之间），说明判断不够确定
5. 评论的表达方式与示例中的隐晦表达类似，但不确定应如何标注

**关键原则**：当判断模糊时，不要依赖自己的推测，必须参考人工复核数据示例中的标注标准，确保分析结果与人工复核的标准保持一致。如果示例中有相似的评论，应直接参考其标注结果。

请以JSON格式返回结果，格式如下：
{{
    "sentiment": "strongly_negative" 或 "weakly_negative" 或 "neutral" 或 "weakly_positive" 或 "strongly_positive",
    "confidence": 0.0-1.0之间的置信度分数,
    "probabilities": {{
        "strongly_negative": 0.0-1.0,
        "weakly_negative": 0.0-1.0,
        "neutral": 0.0-1.0,
        "weakly_positive": 0.0-1.0,
        "strongly_positive": 0.0-1.0
    }},
    "reason": "分析原因（简短说明）",
    "keywords": ["关键词1", "关键词2", ...],
    "negative_parts": ["负面部分1", "负面部分2", ...],
    "suggestions": ["改进建议1", "改进建议2", ...],
    "confidence_calculation": "置信度计算说明（可选）"
}}

【置信度计算说明】
confidence（置信度）的计算方式：
1. 基于probabilities（概率分布）的最大值：confidence = max(probabilities.values())
2. 综合考虑以下因素：
   - 评论表达的明确程度（直接表达 > 隐晦表达）
   - 情感倾向的一致性（单一情感 > 混合情感）
   - 关键词的强度（强烈词汇 > 一般词汇）
   - 上下文语境的支持度（语境一致 > 语境模糊）
   - **与人工复核示例的匹配度**（如果当前评论与示例高度相似，可适当提高confidence）
3. 计算公式示例：
   - 如果probabilities = {{"positive": 0.8, "negative": 0.15, "neutral": 0.05}}
   - 则confidence = max(0.8, 0.15, 0.05) = 0.8
   - 如果表达非常明确且强烈，可以适当提高confidence（但不超过0.95）
   - **如果表达隐晦或存在歧义，但能在人工复核示例中找到相似案例，应参考示例的标注结果，并基于示例的准确性适当提高confidence（因为示例是人工验证的）**
   - 如果表达隐晦且无法在示例中找到相似案例，应适当降低confidence（但不低于0.5）

【判断模糊时的置信度处理】
- 如果probabilities分布较为均匀（如三个值都在0.3-0.4之间），说明判断存在模糊性
- 此时应检查人工复核示例中是否有相似案例：
  * **如果有相似案例，参考其标注结果，confidence应基于示例的准确性（通常较高，0.75-0.9之间，因为示例是人工验证的）**
  * 如果没有相似案例，confidence应适当降低（0.5-0.7之间），表示需要人工复核
- **关键原则**：当判断模糊时，如果示例中有相似案例，应优先采用示例的标注结果，并相应调整confidence和probabilities

请确保：
1. sentiment必须是以下五分类之一（针对成人用品场景优化）：
   - "strongly_negative"（强烈负面）：非常不满意、强烈批评、严重问题
   - "weakly_negative"（轻微负面）：不太满意、轻微问题、可以改进
   - "neutral"（中性）：一般、还可以、无明显倾向
   - "weakly_positive"（轻微正面）：基本满意、还不错、可以接受
   - "strongly_positive"（强烈正面）：非常满意、强烈推荐、超出预期
   混合情感根据主要倾向判断，判断模糊时优先参考人工复核示例
2. confidence是0到1之间的数字，建议基于max(probabilities.values())计算，并根据表达明确程度和与示例的匹配度微调
3. probabilities中的五个值加起来应该接近1.0
4. **当判断模糊时，如果人工复核示例中有相似案例，应优先采用示例的标注结果，并相应调整confidence和probabilities**
5. keywords是5-10个最重要的关键词
6. negative_parts：必须完整列出评论中的所有负面部分，包括直接和隐含的负面表达，每个负面部分用简洁的短语概括（如"噪音大"、"质量一般"、"材质不好"、"舒适度差"、"效果不明显"、"尺寸不合适"、"安全性问题"、"可以改进"等），如果没有负面内容则为空数组
7. suggestions：必须为每个negative_parts中的负面部分生成对应的具体改进建议：
   - 对于材质/安全性问题：建议使用更安全、更舒适的材质
   - 对于舒适度问题：建议优化产品设计，提升使用舒适度
   - 对于功能性问题：建议改进产品功能，提升使用效果
   - 对于尺寸问题：建议提供更多尺寸选择或优化尺寸设计
   - 对于其他问题：根据具体情况生成针对性建议
   建议数量应与负面部分数量一致，如果没有负面内容则为空数组
8. confidence_calculation（可选）：简要说明置信度的计算依据，例如"基于概率分布最大值0.85，结合表达明确性调整为0.88"，如果参考了人工复核示例，应说明"参考了人工复核示例X的标注标准"
9. 只返回JSON，不要其他文字"""
    
    def analyze_sentiment_with_ai(self, text: str) -> Optional[Dict]:
        """
        使用AI分析情感（支持few-shot learning）
        
        Args:
            text: 评论文本
            
        Returns:
            情感分析结果，如果失败返回None
        """
        if not self.ai_enabled:
            return None
        
        # 加载few-shot examples
        few_shot_examples = []
        if self.use_few_shot:
            few_shot_examples = self._load_few_shot_examples(current_text=text, count=self.few_shot_count)
            if few_shot_examples:
                print(f"[Few-shot] 加载了 {len(few_shot_examples)} 个示例用于AI分析")
        
        # 按优先级尝试：DeepSeek > OpenAI
        # 优先使用DeepSeek（如果配置了）
        if self.deepseek_api_key:
            result = self._analyze_with_deepseek(text, few_shot_examples)
            if result:
                return result
        
        # 尝试使用OpenAI
        if self.openai_api_key:
            result = self._analyze_with_openai(text, few_shot_examples)
            if result:
                return result
        
        return None
    
    def _analyze_with_deepseek(self, text: str, few_shot_examples: List[Tuple[str, str]] = None) -> Optional[Dict]:
        """使用DeepSeek API分析（OpenAI兼容接口，支持few-shot learning）"""
        try:
            headers = {
                'Authorization': f'Bearer {self.deepseek_api_key}',
                'Content-Type': 'application/json'
            }
            
            prompt = self._get_analysis_prompt(text, few_shot_examples)
            
            payload = {
                "model": self.deepseek_model,
                "messages": [
                    {"role": "system", "content": self._get_system_message()},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.deepseek_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 尝试提取JSON
                try:
                    # 移除可能的markdown代码块标记
                    if content.startswith('```'):
                        content = content.split('```')[1]
                        if content.startswith('json'):
                            content = content[4:]
                        content = content.strip()
                    
                    ai_result = json.loads(content)
                    
                    # 转换为标准格式
                    # 如果AI没有提供confidence，基于probabilities计算
                    # 支持五分类，兼容旧的三分类
                    ai_probs = ai_result.get("probabilities", {})
                    if SENTIMENT_LABELS_AVAILABLE and any(k in ai_probs for k in ['strongly_negative', 'weakly_negative', 'weakly_positive', 'strongly_positive']):
                        # 五分类
                        probabilities = {
                            "strongly_negative": float(ai_probs.get("strongly_negative", 0.2)),
                            "weakly_negative": float(ai_probs.get("weakly_negative", 0.2)),
                            "neutral": float(ai_probs.get("neutral", 0.2)),
                            "weakly_positive": float(ai_probs.get("weakly_positive", 0.2)),
                            "strongly_positive": float(ai_probs.get("strongly_positive", 0.2))
                        }
                    else:
                        # 兼容旧的三分类，转换为五分类
                        old_positive = float(ai_probs.get("positive", 0.33))
                        old_negative = float(ai_probs.get("negative", 0.33))
                        old_neutral = float(ai_probs.get("neutral", 0.34))
                        # 简单转换：将positive/negative平均分配到weakly和strongly
                        probabilities = {
                            "strongly_negative": old_negative * 0.5,
                            "weakly_negative": old_negative * 0.5,
                            "neutral": old_neutral,
                            "weakly_positive": old_positive * 0.5,
                            "strongly_positive": old_positive * 0.5
                        }
                    confidence = float(ai_result.get("confidence", max(probabilities.values())))
                    
                    # 转换sentiment标签（如果是旧标签）
                    sentiment = ai_result.get("sentiment", "neutral")
                    if SENTIMENT_LABELS_AVAILABLE and sentiment in ['positive', 'negative', 'neutral']:
                        # 根据probabilities判断是strongly还是weakly
                        if sentiment == 'positive':
                            sentiment = 'strongly_positive' if probabilities.get('strongly_positive', 0) > probabilities.get('weakly_positive', 0) else 'weakly_positive'
                        elif sentiment == 'negative':
                            sentiment = 'strongly_negative' if probabilities.get('strongly_negative', 0) > probabilities.get('weakly_negative', 0) else 'weakly_negative'
                    
                    return {
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "reason": ai_result.get("reason", ""),
                        "keywords": ai_result.get("keywords", []),
                        "negative_parts": ai_result.get("negative_parts", []),
                        "suggestions": ai_result.get("suggestions", []),
                        "confidence_calculation": ai_result.get("confidence_calculation", f"基于概率分布最大值max({', '.join([f'{k}={v:.2f}' for k, v in probabilities.items()])}) = {max(probabilities.values()):.2f}"),
                        "method": "ai_deepseek"
                    }
                except json.JSONDecodeError:
                    # 如果JSON解析失败，尝试从文本中提取
                    return self._parse_ai_response(content, text)
            else:
                print(f"DeepSeek API错误: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"DeepSeek分析失败: {str(e)}")
            return None
    
    def _analyze_with_openai(self, text: str, few_shot_examples: List[Tuple[str, str]] = None) -> Optional[Dict]:
        """使用OpenAI API分析（支持few-shot learning）"""
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            prompt = self._get_analysis_prompt(text, few_shot_examples)
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": self._get_system_message()},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.openai_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 尝试提取JSON
                try:
                    # 移除可能的markdown代码块标记
                    if content.startswith('```'):
                        content = content.split('```')[1]
                        if content.startswith('json'):
                            content = content[4:]
                        content = content.strip()
                    
                    ai_result = json.loads(content)
                    
                    # 转换为标准格式
                    # 如果AI没有提供confidence，基于probabilities计算
                    # 支持五分类，兼容旧的三分类
                    ai_probs = ai_result.get("probabilities", {})
                    if SENTIMENT_LABELS_AVAILABLE and any(k in ai_probs for k in ['strongly_negative', 'weakly_negative', 'weakly_positive', 'strongly_positive']):
                        # 五分类
                        probabilities = {
                            "strongly_negative": float(ai_probs.get("strongly_negative", 0.2)),
                            "weakly_negative": float(ai_probs.get("weakly_negative", 0.2)),
                            "neutral": float(ai_probs.get("neutral", 0.2)),
                            "weakly_positive": float(ai_probs.get("weakly_positive", 0.2)),
                            "strongly_positive": float(ai_probs.get("strongly_positive", 0.2))
                        }
                    else:
                        # 兼容旧的三分类，转换为五分类
                        old_positive = float(ai_probs.get("positive", 0.33))
                        old_negative = float(ai_probs.get("negative", 0.33))
                        old_neutral = float(ai_probs.get("neutral", 0.34))
                        # 简单转换：将positive/negative平均分配到weakly和strongly
                        probabilities = {
                            "strongly_negative": old_negative * 0.5,
                            "weakly_negative": old_negative * 0.5,
                            "neutral": old_neutral,
                            "weakly_positive": old_positive * 0.5,
                            "strongly_positive": old_positive * 0.5
                        }
                    confidence = float(ai_result.get("confidence", max(probabilities.values())))
                    
                    # 转换sentiment标签（如果是旧标签）
                    sentiment = ai_result.get("sentiment", "neutral")
                    if SENTIMENT_LABELS_AVAILABLE and sentiment in ['positive', 'negative', 'neutral']:
                        # 根据probabilities判断是strongly还是weakly
                        if sentiment == 'positive':
                            sentiment = 'strongly_positive' if probabilities.get('strongly_positive', 0) > probabilities.get('weakly_positive', 0) else 'weakly_positive'
                        elif sentiment == 'negative':
                            sentiment = 'strongly_negative' if probabilities.get('strongly_negative', 0) > probabilities.get('weakly_negative', 0) else 'weakly_negative'
                    
                    return {
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "reason": ai_result.get("reason", ""),
                        "keywords": ai_result.get("keywords", []),
                        "negative_parts": ai_result.get("negative_parts", []),
                        "suggestions": ai_result.get("suggestions", []),
                        "confidence_calculation": ai_result.get("confidence_calculation", f"基于概率分布最大值max({', '.join([f'{k}={v:.2f}' for k, v in probabilities.items()])}) = {max(probabilities.values()):.2f}"),
                        "method": "ai_openai"
                    }
                except json.JSONDecodeError:
                    # 如果JSON解析失败，尝试从文本中提取
                    return self._parse_ai_response(content, text)
            else:
                print(f"OpenAI API错误: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"OpenAI分析失败: {str(e)}")
            return None
    
    def _parse_ai_response(self, content: str, original_text: str) -> Optional[Dict]:
        """解析AI返回的文本响应"""
        # 简单的文本解析逻辑
        sentiment_map = {
            "positive": ["正面", "积极", "好评", "满意", "positive"],
            "negative": ["负面", "消极", "差评", "不满意", "negative"],
            "neutral": ["中性", "一般", "neutral"]
        }
        
        content_lower = content.lower()
        detected_sentiment = "neutral"
        
        for sentiment, keywords in sentiment_map.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_sentiment = sentiment
                break
        
        # 尝试从内容中提取负面部分
        negative_parts = []
        suggestions = []
        
        # 简单的负面关键词识别
        negative_keywords = ["差", "不好", "慢", "问题", "失望", "不满意", "退货", "垃圾"]
        for keyword in negative_keywords:
            if keyword in original_text:
                negative_parts.append(keyword)
        
        # 生成基础建议
        if negative_parts:
            if "质量" in original_text or "差" in original_text:
                suggestions.append("建议提升产品质量")
            if "物流" in original_text or "慢" in original_text:
                suggestions.append("建议优化物流速度")
            if "服务" in original_text:
                suggestions.append("建议改善服务态度")
            if not suggestions:
                suggestions.append("建议关注用户反馈，及时改进相关问题")
        
        return {
            "sentiment": detected_sentiment,
            "confidence": 0.8,
            "probabilities": {
                "positive": 0.33,
                "negative": 0.33,
                "neutral": 0.34
            },
            "reason": content[:100],
            "keywords": [],
            "negative_parts": negative_parts,
            "suggestions": suggestions,
            "confidence_calculation": "AI返回格式异常，使用默认置信度0.8（基于文本关键词匹配的简单判断）",
            "method": "ai_fallback"
        }
    
    def extract_keywords_with_ai(self, text: str, topK: int = 10) -> Optional[List[Dict]]:
        """使用AI提取关键词"""
        if not self.ai_enabled:
            return None
        
        # 如果情感分析已经返回了关键词，直接使用
        sentiment_result = self.analyze_sentiment_with_ai(text)
        if sentiment_result and sentiment_result.get("keywords"):
            keywords = sentiment_result["keywords"][:topK]
            return [{"word": kw, "weight": 1.0 - (i * 0.1)} for i, kw in enumerate(keywords)]
        
        return None

# 全局AI分析器实例
ai_analyzer = AIAnalyzer()

