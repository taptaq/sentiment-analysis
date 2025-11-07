# 产品评论分析工具

一个基于 React + Flask + Jieba + Sklearn 的电商平台商品评论情感分析工具。

## 功能特性

- ✅ **情感分析**: 使用机器学习模型分析评论情感（正面/负面/中性）
- ✅ **关键词提取**: 自动提取评论中的关键词并统计权重
- ✅ **满意度评分**: 基于情感分析计算用户满意度评分
- ✅ **批量分析**: 支持批量分析多条评论并生成统计报告
- ✅ **可视化展示**: 使用图表展示情感分布和关键词统计

## 技术栈

### 前端
- React 18
- Axios (HTTP请求)
- Recharts (数据可视化)

### 后端
- Flask (Python Web框架)
- Jieba (中文分词)
- Scikit-learn (机器学习)
- TF-IDF (特征提取)
- 朴素贝叶斯 (情感分类)

## 项目结构

```
产品评论分析工具/
├── backend/              # Flask后端
│   ├── app.py           # 主应用文件
│   └── requirements.txt # Python依赖
├── frontend/            # React前端
│   ├── public/
│   ├── src/
│   │   ├── components/  # React组件
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
└── README.md
```

## 安装和运行

### 后端设置

1. 进入后端目录：
```bash
cd backend
```

2. 创建虚拟环境（推荐）：
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 启动Flask服务器：
```bash
python app.py
```

后端将在 `http://localhost:8080` 运行

### 前端设置

1. 进入前端目录：
```bash
cd frontend
```

2. 安装依赖：
```bash
npm install
```

3. 启动开发服务器：
```bash
npm start
```

前端将在 `http://localhost:3000` 运行

## API接口

### 1. 单条评论分析
- **URL**: `/api/analyze`
- **方法**: POST
- **请求体**:
```json
{
  "text": "这个商品质量很好，非常满意！"
}
```
- **响应**:
```json
{
  "text": "评论内容",
  "sentiment": {
    "sentiment": "positive",
    "confidence": 0.95,
    "probabilities": {
      "positive": 0.95,
      "negative": 0.03,
      "neutral": 0.02
    }
  },
  "keywords": [
    {"word": "商品", "weight": 0.5},
    {"word": "质量", "weight": 0.4}
  ]
}
```

### 2. 批量评论分析
- **URL**: `/api/analyze-batch`
- **方法**: POST
- **请求体**:
```json
{
  "comments": [
    {"text": "评论1"},
    {"text": "评论2"}
  ]
}
```
- **响应**: 包含详细分析结果和统计信息

### 3. 健康检查
- **URL**: `/api/health`
- **方法**: GET

## 使用说明

1. **单条评论分析**:
   - 在输入框中输入一条评论
   - 点击"开始分析"按钮
   - 查看情感分析结果、关键词提取和置信度

2. **批量评论分析**:
   - 切换到"批量评论分析"标签
   - 添加多条评论
   - 点击"开始批量分析"
   - 查看满意度评分、情感分布图表和关键词统计

## 注意事项

- 确保后端服务在端口 8080 运行
- 如果遇到 CORS 问题，检查 Flask-CORS 配置
- 模型使用简单的训练数据，实际项目中建议使用更大的数据集进行训练

## 许可证

MIT License

