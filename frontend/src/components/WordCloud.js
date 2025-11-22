import React, { useEffect } from 'react';
import WordCloud from 'react-wordcloud2';
import './WordCloud.css';

const WordCloudComponent = ({ words, width = 600, height = 400 }) => {
  // 准备词云数据：将关键词数组转换为 react-wordcloud2 需要的格式
  // react-wordcloud2 需要的数据格式：[[word, weight], [word, weight], ...]
  const wordList = React.useMemo(() => {
    if (!words || words.length === 0) return [];
    
    return words.map(item => {
      // 支持两种格式：{word, weight} 或 {word, count}
      const word = typeof item === 'string' ? item : (item.word || item.name || '');
      let weight = typeof item === 'string' ? 1 : (item.weight || item.count || 1);
      
      if (!word) {
        return null;
      }
      
      // 确保权重是数字且足够大（wordcloud2 需要较大的权重值才能显示）
      weight = typeof weight === 'number' ? weight : parseFloat(weight) || 1;
      // 将权重放大，确保词云能显示（如果权重太小，词云可能不显示）
      // 对于 count/weight 为 1 的词，也要确保能显示，所以最小权重设为 20
      weight = Math.max(weight * 20, 20); // 至少放大20倍，最小值为20，确保所有词都能显示
      
      // 返回 [word, weight] 格式
      return [word, weight];
    }).filter(item => item !== null);
  }, [words]);

  // 调试：打印接收到的数据
  useEffect(() => {
    console.log('[WordCloud] 接收到的 words 数据:', words);
    console.log('[WordCloud] 转换后的 wordList:', wordList);
  }, [words, wordList]);

  // 添加渲染后的调试
  useEffect(() => {
    if (wordList.length === 0) return;
    
    // 延迟检查，确保组件已渲染
    const timer = setTimeout(() => {
      const container = document.querySelector('.wordcloud-container');
      const canvas = container?.querySelector('canvas');
      console.log('[WordCloud] 容器元素:', container);
      console.log('[WordCloud] Canvas 元素:', canvas);
      if (canvas) {
        console.log('[WordCloud] Canvas 尺寸:', {
          width: canvas.width,
          height: canvas.height,
          offsetWidth: canvas.offsetWidth,
          offsetHeight: canvas.offsetHeight,
          display: window.getComputedStyle(canvas).display,
          visibility: window.getComputedStyle(canvas).visibility,
          opacity: window.getComputedStyle(canvas).opacity
        });
        // 检查 canvas 是否有内容
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, Math.min(canvas.width, 100), Math.min(canvas.height, 100));
        let pixelCount = 0;
        for (let i = 3; i < imageData.data.length; i += 4) {
          if (imageData.data[i] > 0) pixelCount++;
        }
        console.log('[WordCloud] Canvas 像素检查:', pixelCount > 0 ? '有内容' : '无内容', '像素数:', pixelCount);
      } else {
        console.warn('[WordCloud] 未找到 canvas 元素！');
      }
    }, 1000);
    return () => clearTimeout(timer);
  }, [wordList]);

  if (!words || words.length === 0 || wordList.length === 0) {
    return (
      <div className="wordcloud-container empty">
        <p>暂无关键词数据</p>
      </div>
    );
  }

  // 颜色函数：根据单词生成颜色
  const getColor = (word) => {
    const colors = [
      '#667eea', '#764ba2', '#f093fb', '#4facfe', 
      '#43e97b', '#fa709a', '#fee140', '#30cfd0'
    ];
    // 使用单词的哈希值来确保相同词使用相同颜色
    let hash = 0;
    for (let i = 0; i < word.length; i++) {
      hash = word.charCodeAt(i) + ((hash << 5) - hash);
    }
    return colors[Math.abs(hash) % colors.length];
  };

  return (
    <div className="wordcloud-container">
      <WordCloud
        list={wordList}
        width={width}
        height={height}
        component="canvas"
        color={getColor}
        fontFamily="Arial, 'Microsoft YaHei', '微软雅黑', sans-serif"
        minSize={12}
        gridSize={8}
        weightFactor={(size) => size * 1.2}
        rotateRatio={0.3}
        rotationSteps={2}
        shape="circle"
        ellipticity={0.65}
        backgroundColor="transparent"
        onStart={() => console.log('[WordCloud] 开始渲染')}
        onStop={() => console.log('[WordCloud] 渲染完成')}
      />
    </div>
  );
};

export default WordCloudComponent;
