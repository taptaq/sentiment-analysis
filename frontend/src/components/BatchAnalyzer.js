import React, { useState } from 'react';
import axios from 'axios';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { API_BASE_URL } from '../config';
import './BatchAnalyzer.css';

const BatchAnalyzer = () => {
  const [comments, setComments] = useState([
    { text: '' }
  ]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [reviewingIndex, setReviewingIndex] = useState(null);
  const [reviewData, setReviewData] = useState({});
  const [reviewSubmitting, setReviewSubmitting] = useState({});
  const [reviewSubmitted, setReviewSubmitted] = useState(new Set());

  const handleAddComment = () => {
    setComments([...comments, { text: '' }]);
  };

  const handleRemoveComment = (index) => {
    setComments(comments.filter((_, i) => i !== index));
  };

  const handleCommentChange = (index, value) => {
    const newComments = [...comments];
    newComments[index].text = value;
    setComments(newComments);
  };

  const handleDownloadTemplate = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/download-template`, {
        responseType: 'blob'
      });
      
      // 创建下载链接
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', '评论数据导入模板.xlsx');
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError('模板下载失败，请稍后重试');
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // 检查文件类型
    if (!file.name.toLowerCase().endsWith('.xlsx') && !file.name.toLowerCase().endsWith('.xls')) {
      setError('请上传 .xlsx 或 .xls 格式的Excel文件');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.post(`${API_BASE_URL}/upload-excel`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (response.data.success && response.data.comments) {
        // 将导入的评论添加到现有列表
        const newComments = response.data.comments;
        setComments([...comments.filter(c => c.text.trim()), ...newComments]);
        setError(null);
        // 显示成功消息
        alert(`成功导入 ${response.data.total} 条评论！`);
      } else {
        setError(response.data.error || '导入失败');
      }
    } catch (err) {
      setError(err.response?.data?.error || '文件上传失败，请稍后重试');
    } finally {
      setLoading(false);
      // 清空文件输入
      event.target.value = '';
    }
  };

  const handleAnalyze = async () => {
    const validComments = comments.filter(c => c.text.trim());
    if (validComments.length === 0) {
      setError('请至少输入一条评论');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/analyze-batch`, {
        comments: validComments
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || '分析失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  const COLORS = {
    // 五分类颜色
    strongly_positive: '#2e7d32',  // 深绿色
    weakly_positive: '#4caf50',    // 绿色
    neutral: '#ff9800',             // 橙色
    weakly_negative: '#f44336',     // 红色
    strongly_negative: '#d32f2f',   // 深红色
    // 兼容旧标签
    positive: '#4caf50',
    negative: '#f44336'
  };

  const getSentimentText = (sentiment) => {
    const map = {
      // 五分类
      strongly_positive: '强烈正面',
      weakly_positive: '轻微正面',
      neutral: '中性',
      weakly_negative: '轻微负面',
      strongly_negative: '强烈负面',
      // 兼容旧标签
      positive: '正面',
      negative: '负面'
    };
    return map[sentiment] || sentiment;
  };

  const prepareChartData = () => {
    if (!result) return null;

    const sentimentData = Object.entries(result.statistics.sentiment_distribution).map(([key, value]) => ({
      name: getSentimentText(key),
      value: value,
      color: COLORS[key]
    }));

    const keywordData = result.statistics.top_keywords.slice(0, 10).map(kw => ({
      name: kw.word,
      count: kw.count
    }));

    return { sentimentData, keywordData };
  };

  const chartData = prepareChartData();

  const handleSubmitReview = async (index, item) => {
    const reviewSentiment = reviewData[index]?.sentiment;
    const reviewNotes = reviewData[index]?.notes || '';

    if (!reviewSentiment) {
      setError('请选择复核情感');
      return;
    }

    setReviewSubmitting({ ...reviewSubmitting, [index]: true });
    setError(null);

    try {
      await axios.post(`${API_BASE_URL}/human-review`, {
        text: item.text,
        reviewed_sentiment: reviewSentiment,
        reviewed_confidence: 1.0,
        review_notes: reviewNotes
      });
      
      setReviewSubmitted(new Set([...reviewSubmitted, index]));
      setReviewingIndex(null);
      setReviewData({ ...reviewData, [index]: null });
      
      setTimeout(() => {
        const newSubmitted = new Set(reviewSubmitted);
        newSubmitted.delete(index);
        setReviewSubmitted(newSubmitted);
      }, 3000);
    } catch (err) {
      setError(err.response?.data?.error || '提交复核结果失败，请稍后重试');
    } finally {
      setReviewSubmitting({ ...reviewSubmitting, [index]: false });
    }
  };

  const handleReviewDataChange = (index, field, value) => {
    setReviewData({
      ...reviewData,
      [index]: {
        ...reviewData[index],
        [field]: value
      }
    });
  };

  return (
    <div className="batch-analyzer">
      <div className="input-section">
        <h2>批量评论输入</h2>
        
        {/* Excel导入功能 */}
        <div className="excel-import-section">
          <div className="excel-actions">
            <button 
              className="template-button"
              onClick={handleDownloadTemplate}
              type="button"
            >
              📥 下载Excel模板
            </button>
            <label className="upload-button">
              📤 导入Excel文件
              <input
                type="file"
                accept=".xlsx,.xls"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
            </label>
          </div>
          <p className="excel-tip">支持 .xlsx 和 .xls 格式，Excel文件需包含"评论"列</p>
        </div>

        <div className="comments-list">
          {comments.map((comment, index) => (
            <div key={index} className="comment-row">
              <textarea
                className="comment-input"
                placeholder={`评论 ${index + 1}...`}
                value={comment.text}
                onChange={(e) => handleCommentChange(index, e.target.value)}
                rows={3}
              />
              {comments.length > 1 && (
                <button
                  className="remove-button"
                  onClick={() => handleRemoveComment(index)}
                >
                  删除
                </button>
              )}
            </div>
          ))}
        </div>
        <div className="actions">
          <button className="add-button" onClick={handleAddComment}>
            + 添加评论
          </button>
          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={loading}
          >
            {loading ? '分析中...' : '开始批量分析'}
          </button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {result && (
        <div className="result-section">
          {/* 满意度评分 */}
          <div className="result-card satisfaction-card">
            <h3>满意度评分</h3>
            <div className="satisfaction-content">
              <div className="score-circle">
                <div className="score-value">{result.statistics.satisfaction.score}</div>
                <div className="score-label">分</div>
              </div>
              <div className="satisfaction-info">
                <div className="satisfaction-level">
                  满意度等级: <span>{result.statistics.satisfaction.level}</span>
                </div>
                <div className="total-comments">
                  评论总数: {result.statistics.satisfaction.total_comments}
                </div>
              </div>
            </div>
          </div>

          {/* 图表区域 */}
          <div className="charts-container">
            {/* 情感分布饼图 */}
            {chartData && (
              <div className="chart-card">
                <h3>情感分布</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={chartData.sentimentData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {chartData.sentimentData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* 关键词统计柱状图 */}
            {chartData && (
              <div className="chart-card">
                <h3>热门关键词</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData.keywordData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="count" fill="#667eea" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* 人工复核统计 */}
          {result.statistics?.human_review && (
            <div className="result-card human-review-stats-card">
              <h3>人工复核统计</h3>
              <div className="review-stats-content">
                <div className="stat-item">
                  <span className="stat-label">需要复核：</span>
                  <span className="stat-value highlight">{result.statistics.human_review.needed_count || 0}</span>
                  <span className="stat-label">/ {result.statistics.human_review.total_count}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">无效/待观察：</span>
                  <span className="stat-value invalid-value">{result.statistics.human_review.invalid_count || 0}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">自动采纳：</span>
                  <span className="stat-value accepted-value">{result.statistics.human_review.auto_accepted_count || 0}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">复核率：</span>
                  <span className="stat-value">{result.statistics.human_review.review_rate}%</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">置信度区间：</span>
                  <span className="stat-value">
                    {(result.analysis_info?.confidence_thresholds?.min || 0.5) * 100}% - {(result.analysis_info?.confidence_thresholds?.max || 0.85) * 100}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* 详细结果列表 */}
          <div className="result-card">
            <h3>详细分析结果</h3>
            <div className="results-list">
              {result.results.map((item, index) => (
                <div 
                  key={index} 
                  className={`result-item ${
                    item.review_status === 'needs_review' ? 'needs-review' : 
                    item.review_status === 'invalid' ? 'invalid-status' : 
                    item.review_status === 'auto_accepted' ? 'auto-accepted' : ''
                  }`}
                >
                  <div className="result-header">
                    <div className="result-text">{item.text}</div>
                    {item.review_status && (
                      <div className={`review-status-badge ${item.review_status}`}>
                        {item.review_status === 'invalid' && (
                          <>
                            <span className="review-icon">❌</span>
                            <span>无效/待观察 ({(item.confidence * 100).toFixed(1)}%)</span>
                          </>
                        )}
                        {item.review_status === 'needs_review' && (
                          <>
                            <span className="review-icon">⚠️</span>
                            <span>需复核 ({(item.confidence * 100).toFixed(1)}%)</span>
                          </>
                        )}
                        {item.review_status === 'auto_accepted' && (
                          <>
                            <span className="review-icon">✅</span>
                            <span>自动采纳 ({(item.confidence * 100).toFixed(1)}%)</span>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="result-tags">
                    <span
                      className="sentiment-tag"
                      style={{ backgroundColor: COLORS[item.sentiment.sentiment] }}
                    >
                      {getSentimentText(item.sentiment.sentiment)}
                    </span>
                    {item.keywords.slice(0, 5).map((kw, i) => (
                      <span key={i} className="keyword-tag">
                        {kw.word}
                      </span>
                    ))}
                  </div>
                  {/* 分析原因（AI分析特有） */}
                  {item.sentiment?.reason && (
                    <div className="analysis-reason">
                      <span className="reason-label">分析原因：</span>
                      <span className="reason-text">{item.sentiment.reason}</span>
                    </div>
                  )}
                  
                  {/* 负面部分分析（AI分析特有） */}
                  {(item.sentiment?.negative_parts && item.sentiment.negative_parts.length > 0) && (
                    <div className="negative-parts-section">
                      <span className="section-label">负面部分：</span>
                      <div className="negative-parts-list">
                        {item.sentiment.negative_parts.map((part, i) => (
                          <span key={i} className="negative-part-tag">
                            {part}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* 改进建议（AI分析特有） */}
                  {(item.sentiment?.suggestions && item.sentiment.suggestions.length > 0) && (
                    <div className="suggestions-section">
                      <span className="section-label">改进建议：</span>
                      <ul className="suggestions-list">
                        {item.sentiment.suggestions.map((suggestion, i) => (
                          <li key={i} className="suggestion-item">
                            <span className="suggestion-icon">💡</span>
                            <span className="suggestion-text">{suggestion}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {/* 置信度计算说明 */}
                  {item.sentiment?.confidence_calculation && (
                    <div className="confidence-calculation">
                      <span className="calculation-label">置信度计算：</span>
                      <span className="calculation-text">{item.sentiment.confidence_calculation}</span>
                    </div>
                  )}
                  
                  {/* 人工复核区域 */}
                  {item.review_status === 'needs_review' && (
                    <div className="human-review-section">
                      <div className="review-tip">
                        <span className="review-icon">⚠️</span>
                        <span>{item.review_reason || `置信度 ${(item.confidence * 100).toFixed(1)}% 在复核区间内，建议人工复核`}</span>
                      </div>
                      
                      {reviewingIndex !== index && !reviewSubmitted.has(index) && (
                        <button
                          className="review-button"
                          onClick={() => setReviewingIndex(index)}
                        >
                          开始复核
                        </button>
                      )}
                      
                      {reviewingIndex === index && !reviewSubmitted.has(index) && (
                        <div className="review-form">
                          <div className="review-form-group">
                            <label>复核情感：</label>
                            <div className="review-sentiment-options">
                              <button
                                className={`review-option ${reviewData[index]?.sentiment === 'strongly_positive' ? 'active' : ''}`}
                                onClick={() => handleReviewDataChange(index, 'sentiment', 'strongly_positive')}
                              >
                                强烈正面
                              </button>
                              <button
                                className={`review-option ${reviewData[index]?.sentiment === 'weakly_positive' ? 'active' : ''}`}
                                onClick={() => handleReviewDataChange(index, 'sentiment', 'weakly_positive')}
                              >
                                轻微正面
                              </button>
                              <button
                                className={`review-option ${reviewData[index]?.sentiment === 'neutral' ? 'active' : ''}`}
                                onClick={() => handleReviewDataChange(index, 'sentiment', 'neutral')}
                              >
                                中性
                              </button>
                              <button
                                className={`review-option ${reviewData[index]?.sentiment === 'weakly_negative' ? 'active' : ''}`}
                                onClick={() => handleReviewDataChange(index, 'sentiment', 'weakly_negative')}
                              >
                                轻微负面
                              </button>
                              <button
                                className={`review-option ${reviewData[index]?.sentiment === 'strongly_negative' ? 'active' : ''}`}
                                onClick={() => handleReviewDataChange(index, 'sentiment', 'strongly_negative')}
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
                              value={reviewData[index]?.notes || ''}
                              onChange={(e) => handleReviewDataChange(index, 'notes', e.target.value)}
                              rows={2}
                            />
                          </div>
                          <div className="review-form-actions">
                            <button
                              className="submit-review-button"
                              onClick={() => handleSubmitReview(index, item)}
                              disabled={reviewSubmitting[index] || !reviewData[index]?.sentiment}
                            >
                              {reviewSubmitting[index] ? '提交中...' : '提交复核结果'}
                            </button>
                            <button
                              className="cancel-review-button"
                              onClick={() => {
                                setReviewingIndex(null);
                                setReviewData({ ...reviewData, [index]: null });
                              }}
                            >
                              取消
                            </button>
                          </div>
                        </div>
                      )}
                      
                      {reviewSubmitted.has(index) && (
                        <div className="review-success">
                          <span className="success-icon">✅</span>
                          <span>复核结果已提交，感谢您的反馈！</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchAnalyzer;

