import React, { useState } from 'react';
import './App.css';
import CommentAnalyzer from './components/CommentAnalyzer';
import BatchAnalyzer from './components/BatchAnalyzer';

function App() {
  const [activeTab, setActiveTab] = useState('single');

  return (
    <div className="App">
      <header className="app-header">
        <h1>
          <span className="emoji">📊</span>
          <span className="title-text">产品评论情感分析工具</span>
        </h1>
        <p>基于AI的情感分析、关键词提取和满意度评分</p>
      </header>
      
      <div className="tab-container">
        <button 
          className={`tab-button ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          单条评论分析
        </button>
        <button 
          className={`tab-button ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          批量评论分析
        </button>
      </div>

      <main className="app-main">
        {activeTab === 'single' ? <CommentAnalyzer /> : <BatchAnalyzer />}
      </main>
    </div>
  );
}

export default App;

