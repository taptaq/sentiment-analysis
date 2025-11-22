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
from prompt_builder import get_system_message, get_analysis_prompt, get_batch_analysis_prompt

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
        # self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY', '')
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
        
        # 批量分析配置
        self.batch_enabled = os.getenv('AI_BATCH_ENABLED', 'true').lower() == 'true'
        self.batch_size = int(os.getenv('AI_BATCH_SIZE', '10'))
        
        if self.ai_enabled:
            if self.batch_enabled:
                print(f"[AI分析器] 批量分析已启用，批量大小: {self.batch_size}")
    
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
    
    def analyze_sentiment_with_ai(self, text: str, sku: str = None, product_title: str = None) -> Optional[Dict]:
        """
        使用AI分析情感（支持few-shot learning）
        
        Args:
            text: 评论文本
            sku: 商品SKU（可选）
            product_title: 产品标题（可选）
            
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
            result = self._analyze_with_deepseek(text, few_shot_examples, sku, product_title)
            if result:
                return result
        
        # 尝试使用OpenAI
        if self.openai_api_key:
            result = self._analyze_with_openai(text, few_shot_examples, sku, product_title)
            if result:
                return result
        
        return None
    
    def analyze_sentiment_batch_with_ai(self, texts: List[str], batch_size: int = 10, skus: List[str] = None, product_titles: List[str] = None) -> List[Optional[Dict]]:
        """
        批量使用AI分析情感（优化：一次API调用处理多条评论）
        
        Args:
            texts: 评论文本列表
            batch_size: 每批处理的评论数量（默认10条）
            skus: 商品SKU列表（可选，与texts一一对应）
            product_titles: 产品标题列表（可选，与texts一一对应）
            
        Returns:
            情感分析结果列表，与输入列表一一对应，失败时对应位置为None
        """
        if not self.ai_enabled or not texts:
            return [None] * len(texts)
        
        results = []
        total = len(texts)
        
        # 分批处理，每批合并为一次API调用
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_texts = texts[batch_start:batch_end]
            batch_skus = skus[batch_start:batch_end] if skus else [None] * len(batch_texts)
            batch_titles = product_titles[batch_start:batch_end] if product_titles else [None] * len(batch_texts)
            
            batch_num = batch_start//batch_size + 1
            print(f"[批量AI分析] 批次 {batch_num}/{total//batch_size + (1 if total % batch_size else 0)}：合并 {len(batch_texts)} 条评论为一次API调用（评论 {batch_start+1}-{batch_end}/{total}）")
            
            # 尝试批量调用（优先批量，减少API调用次数）
            batch_results = None
            max_retries = 2  # 批量调用最多重试2次
            
            for retry in range(max_retries):
                # 优先使用DeepSeek
                if self.deepseek_api_key:
                    batch_results = self._analyze_batch_with_deepseek(batch_texts, batch_skus, batch_titles)
                    if batch_results:
                        results.extend(batch_results)
                        print(f"[批量AI分析] 批次 {batch_num} 成功：DeepSeek批量调用，处理 {len(batch_results)} 条评论")
                        break
                
                # 尝试使用OpenAI
                if batch_results is None and self.openai_api_key:
                    batch_results = self._analyze_batch_with_openai(batch_texts, batch_skus, batch_titles)
                    if batch_results:
                        results.extend(batch_results)
                        print(f"[批量AI分析] 批次 {batch_num} 成功：OpenAI批量调用，处理 {len(batch_results)} 条评论")
                        break
                
                # 如果失败且还有重试机会
                if batch_results is None and retry < max_retries - 1:
                    print(f"[批量AI分析] 批次 {batch_num} 失败，重试 {retry + 1}/{max_retries}...")
                    import time
                    time.sleep(1)  # 等待1秒后重试
            
            # 如果批量调用完全失败，才回退到单条调用（但尽量避免）
            if batch_results is None:
                print(f"[批量AI分析] 警告：批次 {batch_num} 批量调用失败，回退到单条调用（会增加API调用次数）")
                for text, sku, title in zip(batch_texts, batch_skus, batch_titles):
                    result = self.analyze_sentiment_with_ai(text, sku, title)
                    results.append(result)
        
        return results
    
    
    def _analyze_batch_with_deepseek(self, texts: List[str], skus: List[str] = None, product_titles: List[str] = None) -> Optional[List[Dict]]:
        """使用DeepSeek API批量分析"""
        try:
            headers = {
                'Authorization': f'Bearer {self.deepseek_api_key}',
                'Content-Type': 'application/json'
            }
            
            # 加载few-shot examples（使用第一条评论作为参考）
            few_shot_examples = []
            if self.use_few_shot and texts:
                few_shot_examples = self._load_few_shot_examples(current_text=texts[0], count=self.few_shot_count)
            
            prompt = get_batch_analysis_prompt(texts, few_shot_examples, skus, product_titles)
            # print(f"[批量DeepSeek] 提示：{prompt}")
            
            payload = {
                "model": self.deepseek_model,
                "messages": [
                    {"role": "system", "content": get_system_message()},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": min(500 * len(texts), 4000)  # 根据评论数量调整token限制
            }
            
            response = requests.post(
                f"{self.deepseek_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30  # 批量处理需要更长的超时时间
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 尝试提取JSON数组
                try:
                    # 移除可能的markdown代码块标记
                    if content.startswith('```'):
                        content = content.split('```')[1]
                        if content.startswith('json'):
                            content = content[4:]
                        content = content.strip()
                    
                    batch_results = json.loads(content)
                    
                    # 验证返回格式
                    if not isinstance(batch_results, list):
                        print(f"[批量DeepSeek] 返回格式错误：不是数组，内容：{content[:200]}")
                        return None
                    
                    if len(batch_results) != len(texts):
                        print(f"[批量DeepSeek] 返回数量不匹配：期望{len(texts)}，实际{len(batch_results)}")
                        # 尝试修复：如果返回数量少于期望，补充None；如果多于期望，截取
                        if len(batch_results) < len(texts):
                            print(f"[批量DeepSeek] 尝试修复：补充 {len(texts) - len(batch_results)} 个None结果")
                            batch_results.extend([None] * (len(texts) - len(batch_results)))
                        else:
                            print(f"[批量DeepSeek] 尝试修复：截取前 {len(texts)} 个结果")
                            batch_results = batch_results[:len(texts)]
                    
                    # 转换为标准格式
                    formatted_results = []
                    for idx, item in enumerate(batch_results):
                        # 如果item为None（修复时补充的），跳过
                        if item is None:
                            formatted_results.append(None)
                            continue
                            
                        comment_index = item.get('comment_index', 0)
                        ai_probs = item.get("probabilities", {})
                        
                        # 支持五分类，兼容旧的三分类
                        if SENTIMENT_LABELS_AVAILABLE and any(k in ai_probs for k in ['strongly_negative', 'weakly_negative', 'weakly_positive', 'strongly_positive']):
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
                            probabilities = {
                                "strongly_negative": old_negative * 0.5,
                                "weakly_negative": old_negative * 0.5,
                                "neutral": old_neutral,
                                "weakly_positive": old_positive * 0.5,
                                "strongly_positive": old_positive * 0.5
                            }
                        
                        confidence = float(item.get("confidence", max(probabilities.values())))
                        
                        # 转换sentiment标签
                        sentiment = item.get("sentiment", "neutral")
                        if SENTIMENT_LABELS_AVAILABLE and sentiment in ['positive', 'negative', 'neutral']:
                            if sentiment == 'positive':
                                sentiment = 'strongly_positive' if probabilities.get('strongly_positive', 0) > probabilities.get('weakly_positive', 0) else 'weakly_positive'
                            elif sentiment == 'negative':
                                sentiment = 'strongly_negative' if probabilities.get('strongly_negative', 0) > probabilities.get('weakly_negative', 0) else 'weakly_negative'
                        
                        formatted_results.append({
                            "sentiment": sentiment,
                            "confidence": confidence,
                            "probabilities": probabilities,
                            "reason": item.get("reason", ""),
                            "keywords": item.get("keywords", []),
                            "negative_parts": item.get("negative_parts", []),
                            "suggestions": item.get("suggestions", []),
                            "confidence_calculation": item.get("confidence_calculation", f"基于概率分布最大值max({', '.join([f'{k}={v:.2f}' for k, v in probabilities.items()])}) = {max(probabilities.values()):.2f}"),
                            "method": "ai_deepseek_batch"
                        })
                    
                    print(f"[批量DeepSeek] 成功处理 {len(formatted_results)} 条评论")
                    return formatted_results
                    
                except json.JSONDecodeError as e:
                    print(f"[批量DeepSeek] JSON解析失败: {str(e)}")
                    print(f"[批量DeepSeek] 返回内容: {content[:500]}")
                    return None
            else:
                print(f"DeepSeek批量API错误: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"DeepSeek批量分析失败: {str(e)}")
            return None
    
    def _analyze_batch_with_openai(self, texts: List[str], skus: List[str] = None, product_titles: List[str] = None) -> Optional[List[Dict]]:
        """使用OpenAI API批量分析"""
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            # 加载few-shot examples
            few_shot_examples = []
            if self.use_few_shot and texts:
                few_shot_examples = self._load_few_shot_examples(current_text=texts[0], count=self.few_shot_count)
            
            prompt = get_batch_analysis_prompt(texts, few_shot_examples, skus, product_titles)
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": get_system_message()},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": min(500 * len(texts), 4000)
            }
            
            response = requests.post(
                f"{self.openai_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                try:
                    if content.startswith('```'):
                        content = content.split('```')[1]
                        if content.startswith('json'):
                            content = content[4:]
                        content = content.strip()
                    
                    batch_results = json.loads(content)
                    
                    if not isinstance(batch_results, list):
                        print(f"[批量OpenAI] 返回格式错误：不是数组，内容：{content[:200]}")
                        return None
                    
                    if len(batch_results) != len(texts):
                        print(f"[批量OpenAI] 返回数量不匹配：期望{len(texts)}，实际{len(batch_results)}")
                        # 尝试修复：如果返回数量少于期望，补充None；如果多于期望，截取
                        if len(batch_results) < len(texts):
                            print(f"[批量OpenAI] 尝试修复：补充 {len(texts) - len(batch_results)} 个None结果")
                            batch_results.extend([None] * (len(texts) - len(batch_results)))
                        else:
                            print(f"[批量OpenAI] 尝试修复：截取前 {len(texts)} 个结果")
                            batch_results = batch_results[:len(texts)]
                    
                    # 转换为标准格式（与DeepSeek相同）
                    formatted_results = []
                    for idx, item in enumerate(batch_results):
                        # 如果item为None（修复时补充的），跳过
                        if item is None:
                            formatted_results.append(None)
                            continue
                            
                        ai_probs = item.get("probabilities", {})
                        
                        if SENTIMENT_LABELS_AVAILABLE and any(k in ai_probs for k in ['strongly_negative', 'weakly_negative', 'weakly_positive', 'strongly_positive']):
                            probabilities = {
                                "strongly_negative": float(ai_probs.get("strongly_negative", 0.2)),
                                "weakly_negative": float(ai_probs.get("weakly_negative", 0.2)),
                                "neutral": float(ai_probs.get("neutral", 0.2)),
                                "weakly_positive": float(ai_probs.get("weakly_positive", 0.2)),
                                "strongly_positive": float(ai_probs.get("strongly_positive", 0.2))
                            }
                        else:
                            old_positive = float(ai_probs.get("positive", 0.33))
                            old_negative = float(ai_probs.get("negative", 0.33))
                            old_neutral = float(ai_probs.get("neutral", 0.34))
                            probabilities = {
                                "strongly_negative": old_negative * 0.5,
                                "weakly_negative": old_negative * 0.5,
                                "neutral": old_neutral,
                                "weakly_positive": old_positive * 0.5,
                                "strongly_positive": old_positive * 0.5
                            }
                        
                        confidence = float(item.get("confidence", max(probabilities.values())))
                        
                        sentiment = item.get("sentiment", "neutral")
                        if SENTIMENT_LABELS_AVAILABLE and sentiment in ['positive', 'negative', 'neutral']:
                            if sentiment == 'positive':
                                sentiment = 'strongly_positive' if probabilities.get('strongly_positive', 0) > probabilities.get('weakly_positive', 0) else 'weakly_positive'
                            elif sentiment == 'negative':
                                sentiment = 'strongly_negative' if probabilities.get('strongly_negative', 0) > probabilities.get('weakly_negative', 0) else 'weakly_negative'
                        
                        formatted_results.append({
                            "sentiment": sentiment,
                            "confidence": confidence,
                            "probabilities": probabilities,
                            "reason": item.get("reason", ""),
                            "keywords": item.get("keywords", []),
                            "negative_parts": item.get("negative_parts", []),
                            "suggestions": item.get("suggestions", []),
                            "confidence_calculation": item.get("confidence_calculation", f"基于概率分布最大值max({', '.join([f'{k}={v:.2f}' for k, v in probabilities.items()])}) = {max(probabilities.values()):.2f}"),
                            "method": "ai_openai_batch"
                        })
                    
                    print(f"[批量OpenAI] 成功处理 {len(formatted_results)} 条评论")
                    return formatted_results
                    
                except json.JSONDecodeError as e:
                    print(f"[批量OpenAI] JSON解析失败: {str(e)}")
                    return None
            else:
                print(f"OpenAI批量API错误: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"OpenAI批量分析失败: {str(e)}")
            return None
    
    def _analyze_with_deepseek(self, text: str, few_shot_examples: List[Tuple[str, str]] = None, sku: str = None, product_title: str = None) -> Optional[Dict]:
        """使用DeepSeek API分析（OpenAI兼容接口，支持few-shot learning）"""
        try:
            headers = {
                'Authorization': f'Bearer {self.deepseek_api_key}',
                'Content-Type': 'application/json'
            }
            
            prompt = get_analysis_prompt(text, few_shot_examples, sku, product_title)
            # print(f"[DeepSeek分析] 提示：{prompt}")
            
            payload = {
                "model": self.deepseek_model,
                "messages": [
                    {"role": "system", "content": get_system_message()},
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
    
    def _analyze_with_openai(self, text: str, few_shot_examples: List[Tuple[str, str]] = None, sku: str = None, product_title: str = None) -> Optional[Dict]:
        """使用OpenAI API分析（支持few-shot learning）"""
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            prompt = get_analysis_prompt(text, few_shot_examples, sku, product_title)
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": get_system_message()},
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

