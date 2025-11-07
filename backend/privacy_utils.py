"""
隐私和安全工具模块
用于处理敏感信息、数据脱敏、隐私保护等
"""
import os
import re
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class PrivacyUtils:
    """隐私和安全工具类"""
    
    # 敏感信息模式（可根据行业扩展）
    SENSITIVE_PATTERNS = {
        # 个人信息
        'phone': r'1[3-9]\d{9}',  # 手机号
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # 邮箱
        'id_card': r'\d{17}[\dXx]',  # 身份证号
        'address': r'[\u4e00-\u9fa5]{2,}(省|市|区|县|街道|路|号)',  # 地址
        
        # 支付信息
        'bank_card': r'\d{16,19}',  # 银行卡号
        'alipay': r'支付宝|alipay',
        'wechat_pay': r'微信支付|wechat',
        
        # 隐私相关词汇（针对特殊行业）
        'privacy_terms': [
            '真实姓名', '姓名', '电话', '地址', '收货地址',
            '个人信息', '隐私', '保密', '私密',
            '个人', '本人', '我的', '自己'
        ]
    }
    
    def __init__(self, enable_desensitization: bool = True, enable_logging: bool = True):
        """
        初始化隐私工具
        
        Args:
            enable_desensitization: 是否启用数据脱敏
            enable_logging: 是否启用日志记录（脱敏后）
        """
        self.enable_desensitization = enable_desensitization
        self.enable_logging = enable_logging
        self.sensitive_data_cache = {}  # 敏感数据缓存（用于审计）
    
    def detect_sensitive_info(self, text: str) -> Dict[str, List[str]]:
        """
        检测文本中的敏感信息
        
        Args:
            text: 待检测的文本
            
        Returns:
            检测到的敏感信息字典
        """
        detected = {
            'phone': [],
            'email': [],
            'id_card': [],
            'address': [],
            'bank_card': [],
            'privacy_terms': []
        }
        
        # 检测手机号
        phones = re.findall(self.SENSITIVE_PATTERNS['phone'], text)
        if phones:
            detected['phone'] = phones
        
        # 检测邮箱
        emails = re.findall(self.SENSITIVE_PATTERNS['email'], text)
        if emails:
            detected['email'] = emails
        
        # 检测身份证号
        id_cards = re.findall(self.SENSITIVE_PATTERNS['id_card'], text)
        if id_cards:
            detected['id_card'] = id_cards
        
        # 检测地址
        addresses = re.findall(self.SENSITIVE_PATTERNS['address'], text)
        if addresses:
            detected['address'] = addresses
        
        # 检测银行卡号
        bank_cards = re.findall(self.SENSITIVE_PATTERNS['bank_card'], text)
        if bank_cards:
            detected['bank_card'] = bank_cards
        
        # 检测隐私相关词汇
        for term in self.SENSITIVE_PATTERNS['privacy_terms']:
            if term in text:
                detected['privacy_terms'].append(term)
        
        return detected
    
    def desensitize_text(self, text: str, keep_context: bool = True) -> Tuple[str, Dict]:
        """
        对文本进行脱敏处理
        
        Args:
            text: 待脱敏的文本
            keep_context: 是否保留上下文（部分脱敏）
            
        Returns:
            (脱敏后的文本, 脱敏信息记录)
        """
        if not self.enable_desensitization:
            return text, {}
        
        desensitized_text = text
        desensitization_log = {
            'original_length': len(text),
            'replacements': []
        }
        
        # 检测敏感信息
        sensitive_info = self.detect_sensitive_info(text)
        
        # 脱敏处理
        for info_type, values in sensitive_info.items():
            if not values:
                continue
            
            for value in values:
                if info_type == 'phone':
                    # 手机号：保留前3位和后4位，中间用*代替
                    masked = value[:3] + '****' + value[-4:] if keep_context else '***'
                    desensitized_text = desensitized_text.replace(value, masked)
                    desensitization_log['replacements'].append({
                        'type': 'phone',
                        'original': value,
                        'masked': masked
                    })
                
                elif info_type == 'email':
                    # 邮箱：保留用户名前2位，域名保留
                    parts = value.split('@')
                    if len(parts) == 2:
                        username = parts[0]
                        domain = parts[1]
                        masked_username = username[:2] + '***' if len(username) > 2 else '***'
                        masked = f"{masked_username}@{domain}" if keep_context else '***@***'
                        desensitized_text = desensitized_text.replace(value, masked)
                        desensitization_log['replacements'].append({
                            'type': 'email',
                            'original': value,
                            'masked': masked
                        })
                
                elif info_type == 'id_card':
                    # 身份证号：保留前6位和后4位
                    masked = value[:6] + '********' + value[-4:] if keep_context else '***'
                    desensitized_text = desensitized_text.replace(value, masked)
                    desensitization_log['replacements'].append({
                        'type': 'id_card',
                        'original': value,
                        'masked': masked
                    })
                
                elif info_type == 'bank_card':
                    # 银行卡号：保留前4位和后4位
                    masked = value[:4] + '****' + value[-4:] if keep_context else '***'
                    desensitized_text = desensitized_text.replace(value, masked)
                    desensitization_log['replacements'].append({
                        'type': 'bank_card',
                        'original': value,
                        'masked': masked
                    })
                
                elif info_type == 'address':
                    # 地址：部分脱敏
                    if keep_context:
                        # 保留省市区，脱敏详细地址
                        masked = re.sub(r'[\u4e00-\u9fa5]{2,}(街道|路|号|小区|单元|室)', '***', value)
                    else:
                        masked = '***'
                    desensitized_text = desensitized_text.replace(value, masked)
                    desensitization_log['replacements'].append({
                        'type': 'address',
                        'original': value,
                        'masked': masked
                    })
        
        desensitization_log['desensitized_length'] = len(desensitized_text)
        desensitization_log['has_sensitive_info'] = len(desensitization_log['replacements']) > 0
        
        return desensitized_text, desensitization_log
    
    def hash_text(self, text: str, algorithm: str = 'sha256') -> str:
        """
        对文本进行哈希处理（用于去重，不泄露原文）
        
        Args:
            text: 待哈希的文本
            algorithm: 哈希算法（sha256, md5等）
            
        Returns:
            哈希值
        """
        if algorithm == 'sha256':
            return hashlib.sha256(text.encode('utf-8')).hexdigest()
        elif algorithm == 'md5':
            return hashlib.md5(text.encode('utf-8')).hexdigest()
        else:
            raise ValueError(f"不支持的哈希算法: {algorithm}")
    
    def should_use_local_analysis(self, text: str, industry: str = None) -> bool:
        """
        判断是否应该使用本地分析（避免敏感数据上传到第三方AI）
        
        Args:
            text: 评论文本
            industry: 行业类型（如'adult_products'）
            
        Returns:
            是否应该使用本地分析
        """
        # 检测敏感信息
        sensitive_info = self.detect_sensitive_info(text)
        has_sensitive = any(sensitive_info.values())
        
        # 特殊行业强制使用本地分析
        if industry == 'adult_products':
            return True
        
        # 如果包含敏感信息，建议使用本地分析
        if has_sensitive:
            return True
        
        return False
    
    def sanitize_log_message(self, message: str) -> str:
        """
        清理日志消息中的敏感信息
        
        Args:
            message: 日志消息
            
        Returns:
            清理后的日志消息
        """
        if not self.enable_logging:
            return "[日志已禁用]"
        
        # 对日志中的敏感信息进行脱敏
        desensitized, _ = self.desensitize_text(message, keep_context=False)
        return desensitized
    
    def create_audit_log(self, action: str, text_hash: str, 
                        has_sensitive: bool, desensitization_log: Dict) -> Dict:
        """
        创建审计日志（不包含敏感信息）
        
        Args:
            action: 操作类型（analyze, batch_analyze等）
            text_hash: 文本哈希值
            has_sensitive: 是否包含敏感信息
            desensitization_log: 脱敏日志
            
        Returns:
            审计日志字典
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'text_hash': text_hash,
            'has_sensitive_info': has_sensitive,
            'desensitization_applied': len(desensitization_log.get('replacements', [])) > 0,
            'sensitive_types_detected': [r['type'] for r in desensitization_log.get('replacements', [])]
        }
    
    def filter_privacy_terms(self, text: str, replace_with: str = '[已过滤]') -> str:
        """
        过滤隐私相关词汇（针对特殊行业）
        
        Args:
            text: 待过滤的文本
            replace_with: 替换文本
            
        Returns:
            过滤后的文本
        """
        filtered_text = text
        for term in self.SENSITIVE_PATTERNS['privacy_terms']:
            # 只过滤完整的词汇，避免误伤
            pattern = r'\b' + re.escape(term) + r'\b'
            filtered_text = re.sub(pattern, replace_with, filtered_text)
        
        return filtered_text

# 全局隐私工具实例
privacy_utils = PrivacyUtils(
    enable_desensitization=os.getenv('ENABLE_DESENSITIZATION', 'true').lower() == 'true',
    enable_logging=os.getenv('ENABLE_LOGGING', 'true').lower() == 'true'
)

