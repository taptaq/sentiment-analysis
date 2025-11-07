import React, { useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config';
import './CommentAnalyzer.css';

const CommentAnalyzer = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showReviewForm, setShowReviewForm] = useState(false);
  const [reviewSentiment, setReviewSentiment] = useState('');
  const [reviewNotes, setReviewNotes] = useState('');
  const [reviewSubmitting, setReviewSubmitting] = useState(false);
  const [reviewSubmitted, setReviewSubmitted] = useState(false);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('请输入评论内容');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/analyze`, {
        text: text.trim()
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || '分析失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    // 支持五分类
    const colorMap = {
      'strongly_positive': '#2e7d32',  // 深绿色
      'weakly_positive': '#4caf50',    // 绿色
      'neutral': '#ff9800',             // 橙色
      'weakly_negative': '#f44336',     // 红色
      'strongly_negative': '#d32f2f',   // 深红色
      // 兼容旧标签
      'positive': '#4caf50',
      'negative': '#f44336'
    };
    return colorMap[sentiment] || '#ff9800';
  };

  const getSentimentText = (sentiment) => {
    // 支持五分类
    const textMap = {
      'strongly_positive': '强烈正面',
      'weakly_positive': '轻微正面',
      'neutral': '中性',
      'weakly_negative': '轻微负面',
      'strongly_negative': '强烈负面',
      // 兼容旧标签
      'positive': '正面',
      'negative': '负面'
    };
    return textMap[sentiment] || '中性';
  };

  const handleSubmitReview = async () => {
    if (!reviewSentiment) {
      setError('请选择复核情感');
      return;
    }

    setReviewSubmitting(true);
    setError(null);

    try {
      await axios.post(`${API_BASE_URL}/human-review`, {
        text: result.text,
        reviewed_sentiment: reviewSentiment,
        reviewed_confidence: 1.0,
        review_notes: reviewNotes
      });
      
      setReviewSubmitted(true);
      setShowReviewForm(false);
      // 可以显示成功消息
      setTimeout(() => {
        setReviewSubmitted(false);
        setReviewSentiment('');
        setReviewNotes('');
      }, 3000);
    } catch (err) {
      setError(err.response?.data?.error || '提交复核结果失败，请稍后重试');
    } finally {
      setReviewSubmitting(false);
    }
  };

  return (
    <div className="comment-analyzer">
      <div className="input-section">
        <h2>输入评论内容</h2>
        <textarea
          className="comment-input"
          placeholder="请输入商品评论..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={6}
        />
        <button 
          className="analyze-button"
          onClick={handleAnalyze}
          disabled={loading}
        >
          {loading ? '分析中...' : '开始分析'}
        </button>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {result && (
        <div className="result-section">
          <div className="result-card">
            <h3>评论内容</h3>
            <p className="comment-text">{result.text}</p>
          </div>

          <div className="result-card">
            <h3>情感分析</h3>
            <div className="sentiment-info">
              <div 
                className="sentiment-badge"
                style={{ backgroundColor: getSentimentColor(result.sentiment.sentiment) }}
              >
                {getSentimentText(result.sentiment.sentiment)}
              </div>
              <div className="confidence">
                置信度: {(result.sentiment.confidence * 100).toFixed(1)}%
                {result.sentiment?.confidence_calculation && (
                  <div className="confidence-tooltip" title={result.sentiment.confidence_calculation}>
                    <span className="info-icon">ℹ️</span>
                  </div>
                )}
              </div>
              {result.sentiment?.confidence_calculation && (
                <div className="confidence-calculation">
                  <span className="calculation-label">置信度计算：</span>
                  <span className="calculation-text">{result.sentiment.confidence_calculation}</span>
                </div>
              )}
              {/* 人工复核标记 */}
              {result.analysis_info?.review_status && (
                <div className={`review-status-badge ${result.analysis_info.review_status}`}>
                  {result.analysis_info.review_status === 'invalid' && (
                    <>
                      <span className="review-icon">❌</span>
                      <span>无效/待观察</span>
                    </>
                  )}
                  {result.analysis_info.review_status === 'needs_review' && (
                    <>
                      <span className="review-icon">⚠️</span>
                      <span>需要人工复核</span>
                    </>
                  )}
                  {result.analysis_info.review_status === 'auto_accepted' && (
                    <>
                      <span className="review-icon">✅</span>
                      <span>自动采纳</span>
                    </>
                  )}
                </div>
              )}
            </div>
            <div className="probabilities">
              {Object.entries(result.sentiment.probabilities).map(([label, prob]) => (
                <div key={label} className="probability-item">
                  <span>{getSentimentText(label)}</span>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ 
                        width: `${prob * 100}%`,
                        backgroundColor: getSentimentColor(label)
                      }}
                    />
                  </div>
                  <span>{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
            {/* 分析原因（AI分析特有） */}
            {(result.sentiment?.reason || result.analysis_reason) && (
              <div className="analysis-reason">
                <span className="reason-label">分析原因：</span>
                <span className="reason-text">{result.sentiment?.reason || result.analysis_reason}</span>
              </div>
            )}
            
            {/* 负面部分分析（AI分析特有） */}
            {(result.negative_parts && result.negative_parts.length > 0) && (
              <div className="negative-parts-section">
                <h4 className="section-subtitle">负面部分识别</h4>
                <div className="negative-parts-list">
                  {result.negative_parts.map((part, index) => (
                    <span key={index} className="negative-part-tag">
                      {part}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {/* 改进建议（AI分析特有） */}
            {(result.suggestions && result.suggestions.length > 0) && (
              <div className="suggestions-section">
                <h4 className="section-subtitle">改进建议</h4>
                <ul className="suggestions-list">
                  {result.suggestions.map((suggestion, index) => (
                    <li key={index} className="suggestion-item">
                      <span className="suggestion-icon">💡</span>
                      <span className="suggestion-text">{suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="result-card">
            <h3>关键词提取</h3>
            <div className="keywords-list">
              {result.keywords.map((kw, index) => (
                <div key={index} className="keyword-item">
                  <span className="keyword-word">{kw.word}</span>
                  <span className="keyword-weight">权重: {kw.weight.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* 人工复核区域 */}
          {result.analysis_info?.review_status === 'needs_review' && (
            <div className="result-card human-review-card">
              <h3>人工复核</h3>
              <div className="review-info">
                <p className="review-tip">
                  <span className="review-icon">⚠️</span>
                  {result.analysis_info.review_reason || 
                    `该评论的置信度为 ${(result.analysis_info.confidence * 100).toFixed(1)}%，在复核区间内（${(result.analysis_info.confidence_thresholds?.min * 100).toFixed(0)}%-${(result.analysis_info.confidence_thresholds?.max * 100).toFixed(0)}%），建议进行人工复核以确保准确性。`}
                </p>
                {!showReviewForm && !reviewSubmitted && (
                  <button 
                    className="review-button"
                    onClick={() => setShowReviewForm(true)}
                  >
                    开始复核
                  </button>
                )}
                
                {showReviewForm && !reviewSubmitted && (
                  <div className="review-form">
                    <div className="review-form-group">
                      <label>复核情感：</label>
                      <div className="review-sentiment-options">
                        <button
                          className={`review-option ${reviewSentiment === 'strongly_positive' ? 'active' : ''}`}
                          onClick={() => setReviewSentiment('strongly_positive')}
                        >
                          强烈正面
                        </button>
                        <button
                          className={`review-option ${reviewSentiment === 'weakly_positive' ? 'active' : ''}`}
                          onClick={() => setReviewSentiment('weakly_positive')}
                        >
                          轻微正面
                        </button>
                        <button
                          className={`review-option ${reviewSentiment === 'neutral' ? 'active' : ''}`}
                          onClick={() => setReviewSentiment('neutral')}
                        >
                          中性
                        </button>
                        <button
                          className={`review-option ${reviewSentiment === 'weakly_negative' ? 'active' : ''}`}
                          onClick={() => setReviewSentiment('weakly_negative')}
                        >
                          轻微负面
                        </button>
                        <button
                          className={`review-option ${reviewSentiment === 'strongly_negative' ? 'active' : ''}`}
                          onClick={() => setReviewSentiment('strongly_negative')}
                        >
                          强烈负面
                        </button>
                      </div>
                    </div>
                    <div className="review-form-group">
                      <label>复核备注（可选）：</label>
                      <textarea
                        className="review-notes-input"
                        placeholder="请输入复核备注..."
                        value={reviewNotes}
                        onChange={(e) => setReviewNotes(e.target.value)}
                        rows={3}
                      />
                    </div>
                    <div className="review-form-actions">
                      <button
                        className="submit-review-button"
                        onClick={handleSubmitReview}
                        disabled={reviewSubmitting || !reviewSentiment}
                      >
                        {reviewSubmitting ? '提交中...' : '提交复核结果'}
                      </button>
                      <button
                        className="cancel-review-button"
                        onClick={() => {
                          setShowReviewForm(false);
                          setReviewSentiment('');
                          setReviewNotes('');
                        }}
                      >
                        取消
                      </button>
                    </div>
                  </div>
                )}

                {reviewSubmitted && (
                  <div className="review-success">
                    <span className="success-icon">✅</span>
                    <span>复核结果已提交，感谢您的反馈！</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CommentAnalyzer;

