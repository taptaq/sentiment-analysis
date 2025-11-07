"""
可用性监控脚本
用于统计系统可用性和API调用成功率
"""
import requests
import time
import json
from datetime import datetime
from statistics import mean

API_BASE_URL = "http://localhost:8080/api"

def monitor_availability(test_count=100, interval=0.1):
    """监控系统可用性"""
    print("=" * 70)
    print("系统可用性监控")
    print("=" * 70)
    print(f"API地址: {API_BASE_URL}")
    print(f"测试次数: {test_count}")
    print(f"测试间隔: {interval}秒")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success_count = 0
    fail_count = 0
    response_times = []
    errors = []
    ai_count = 0
    ml_count = 0
    
    test_comments = [
        "这个商品质量很好，非常满意！",
        "物流很快，包装精美，五星好评！",
        "质量不错，价格实惠",
        "商品质量差，不推荐",
        "物流太慢，等了很久",
        "一般般，还可以"
    ]
    
    for i in range(test_count):
        comment = test_comments[i % len(test_comments)]
        
        start = time.time()
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                json={
                    'text': comment,
                    'analysis_mode': 'auto'
                },
                timeout=10
            )
            end = time.time()
            
            response_time = end - start
            response_times.append(response_time)
            
            if response.status_code == 200:
                success_count += 1
                data = response.json()
                
                # 统计使用的分析方法
                method = data.get('sentiment', {}).get('method', '')
                if 'ai' in method:
                    ai_count += 1
                elif 'ml' in method:
                    ml_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{test_count} - 成功: {success_count}, 失败: {fail_count}", end='\r')
            else:
                fail_count += 1
                errors.append({
                    'index': i+1,
                    'status_code': response.status_code,
                    'error': f"HTTP {response.status_code}"
                })
                if (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{test_count} - 成功: {success_count}, 失败: {fail_count}", end='\r')
                    
        except requests.exceptions.Timeout:
            fail_count += 1
            errors.append({
                'index': i+1,
                'error': '请求超时'
            })
            if (i + 1) % 10 == 0:
                print(f"进度: {i+1}/{test_count} - 成功: {success_count}, 失败: {fail_count}", end='\r')
        except Exception as e:
            fail_count += 1
            errors.append({
                'index': i+1,
                'error': str(e)
            })
            if (i + 1) % 10 == 0:
                print(f"进度: {i+1}/{test_count} - 成功: {success_count}, 失败: {fail_count}", end='\r')
        
        time.sleep(interval)
    
    print()  # 换行
    
    # 计算统计结果
    availability = (success_count / test_count) * 100 if test_count > 0 else 0
    
    print("\n" + "=" * 70)
    print("监控结果")
    print("=" * 70)
    print(f"总请求数: {test_count}")
    print(f"成功请求: {success_count}")
    print(f"失败请求: {fail_count}")
    print(f"可用性: {availability:.2f}%")
    print()
    
    if response_times:
        print("响应时间统计:")
        print(f"  平均响应时间: {mean(response_times):.3f}秒")
        print(f"  最快响应时间: {min(response_times):.3f}秒")
        print(f"  最慢响应时间: {max(response_times):.3f}秒")
        print()
    
    print("分析方法统计:")
    print(f"  AI分析: {ai_count}次 ({ai_count/success_count*100 if success_count > 0 else 0:.1f}%)")
    print(f"  ML分析: {ml_count}次 ({ml_count/success_count*100 if success_count > 0 else 0:.1f}%)")
    print()
    
    if errors:
        print(f"错误详情 (前10个):")
        for error in errors[:10]:
            print(f"  第{error['index']}次: {error.get('error', '未知错误')}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")
        print()
    
    # 计算降级情况
    if success_count > 0:
        ai_rate = ai_count / success_count * 100
        ml_rate = ml_count / success_count * 100
        
        print("降级机制分析:")
        print(f"  AI直接成功: {ai_rate:.1f}%")
        print(f"  ML降级使用: {ml_rate:.1f}%")
        print(f"  降级触发率: {ml_rate:.1f}%")
        print()
        
        # 计算最终可用性（考虑降级）
        if ml_rate > 0:
            print("可用性分析:")
            print(f"  如果AI服务100%可用，可用性应该接近100%")
            print(f"  实际AI可用率: {ai_rate:.1f}%")
            print(f"  通过ML降级，最终可用性: {availability:.2f}%")
            print(f"  降级机制提升了 {availability - ai_rate:.2f}% 的可用性")
    
    print("=" * 70)
    print(f"监控完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return {
        'total': test_count,
        'success': success_count,
        'fail': fail_count,
        'availability': availability,
        'avg_response_time': mean(response_times) if response_times else 0,
        'ai_count': ai_count,
        'ml_count': ml_count,
        'errors': errors
    }

def check_server():
    """检查服务器是否运行"""
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api', '')}/api/health", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

if __name__ == '__main__':
    import sys
    
    print("系统可用性监控脚本")
    print()
    
    # 检查服务器是否运行
    print("检查服务器状态...")
    if not check_server():
        print("错误: 后端服务器未运行，请先启动后端服务")
        print("启动命令: cd backend && python app.py")
        sys.exit(1)
    
    print("服务器运行正常\n")
    
    # 默认测试100次
    test_count = 100
    if len(sys.argv) > 1:
        try:
            test_count = int(sys.argv[1])
        except:
            pass
    
    result = monitor_availability(test_count=test_count)
    
    print("\n监控结果摘要:")
    print(f"可用性: {result['availability']:.2f}%")
    print(f"平均响应时间: {result['avg_response_time']:.3f}秒")
    print(f"AI使用率: {result['ai_count']/result['success']*100 if result['success'] > 0 else 0:.1f}%")
    print(f"ML降级率: {result['ml_count']/result['success']*100 if result['success'] > 0 else 0:.1f}%")

