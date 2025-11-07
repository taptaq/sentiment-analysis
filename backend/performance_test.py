"""
性能测试脚本
用于测试批量分析的效率和耗时
"""
import time
import requests
import json
from statistics import mean, median

API_BASE_URL = "http://localhost:8080/api"

def test_single_analysis():
    """测试单条分析性能"""
    print("=" * 70)
    print("单条分析性能测试")
    print("=" * 70)
    
    test_comment = "这个商品质量很好，非常满意！"
    
    times = []
    success_count = 0
    
    print(f"测试评论: {test_comment}")
    print(f"测试次数: 10次")
    print()
    
    for i in range(10):
        start = time.time()
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                json={'text': test_comment, 'analysis_mode': 'auto'},
                timeout=10
            )
            end = time.time()
            
            if response.status_code == 200:
                elapsed = end - start
                times.append(elapsed)
                success_count += 1
                print(f"第{i+1}次: {elapsed:.3f}秒 - 成功")
            else:
                print(f"第{i+1}次: 失败 (状态码: {response.status_code})")
        except Exception as e:
            print(f"第{i+1}次: 失败 ({str(e)})")
    
    if times:
        print()
        print("=" * 70)
        print("统计结果:")
        print("=" * 70)
        print(f"成功次数: {success_count}/10")
        print(f"平均耗时: {mean(times):.3f}秒")
        print(f"中位数耗时: {median(times):.3f}秒")
        print(f"最快耗时: {min(times):.3f}秒")
        print(f"最慢耗时: {max(times):.3f}秒")
        print(f"成功率: {success_count/10*100:.1f}%")
    else:
        print("\n所有测试都失败了，请检查后端服务是否运行")
    
    return times

def test_batch_analysis(sizes=[10, 50, 100]):
    """测试批量分析性能"""
    print("\n" + "=" * 70)
    print("批量分析性能测试")
    print("=" * 70)
    
    # 生成测试评论
    test_comments = [
        "这个商品质量很好，非常满意！",
        "物流很快，包装精美，五星好评！",
        "质量不错，价格实惠",
        "商品质量差，不推荐",
        "物流太慢，等了很久",
        "一般般，还可以",
        "普通商品，没什么特别",
        "非常喜欢，下次还会购买",
        "价格虚高，不值这个价",
        "服务态度好，包装精美"
    ]
    
    results = []
    
    for size in sizes:
        print(f"\n测试批量分析 {size} 条评论...")
        
        # 生成指定数量的评论列表
        comments = []
        for i in range(size):
            comments.append({
                'text': test_comments[i % len(test_comments)]
            })
        
        times = []
        success_count = 0
        
        # 测试3次取平均值
        for test_num in range(3):
            start = time.time()
            try:
                response = requests.post(
                    f"{API_BASE_URL}/analyze-batch",
                    json={
                        'comments': comments,
                        'analysis_mode': 'auto'
                    },
                    timeout=60
                )
                end = time.time()
                
                if response.status_code == 200:
                    elapsed = end - start
                    times.append(elapsed)
                    success_count += 1
                    print(f"  第{test_num+1}次: {elapsed:.2f}秒")
                else:
                    print(f"  第{test_num+1}次: 失败 (状态码: {response.status_code})")
            except Exception as e:
                print(f"  第{test_num+1}次: 失败 ({str(e)})")
        
        if times:
            avg_time = mean(times)
            avg_per_comment = avg_time / size
            results.append({
                'size': size,
                'avg_time': avg_time,
                'avg_per_comment': avg_per_comment,
                'success_count': success_count
            })
            print(f"  平均耗时: {avg_time:.2f}秒")
            print(f"  平均每条: {avg_per_comment:.3f}秒")
    
    if results:
        print("\n" + "=" * 70)
        print("批量分析性能统计:")
        print("=" * 70)
        print(f"{'评论数量':<10} {'总耗时(秒)':<15} {'平均每条(秒)':<15} {'成功率':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['size']:<10} {r['avg_time']:<15.2f} {r['avg_per_comment']:<15.3f} {r['success_count']/3*100:<10.1f}%")
        
        # 计算效率提升
        if len(results) >= 2:
            print("\n效率分析:")
            first = results[0]
            last = results[-1]
            print(f"从 {first['size']} 条到 {last['size']} 条:")
            print(f"  总耗时: {first['avg_time']:.2f}秒 → {last['avg_time']:.2f}秒")
            print(f"  平均每条: {first['avg_per_comment']:.3f}秒 → {last['avg_per_comment']:.3f}秒")
            
            # 估算手动分析时间（假设每条评论需要30秒）
            manual_time_per_comment = 30
            for r in results:
                manual_total = r['size'] * manual_time_per_comment
                efficiency_gain = manual_total / r['avg_time']
                print(f"\n  {r['size']}条评论:")
                print(f"    手动分析时间: {manual_total}秒 ({manual_total/60:.1f}分钟)")
                print(f"    系统分析时间: {r['avg_time']:.2f}秒 ({r['avg_time']/60:.2f}分钟)")
                print(f"    效率提升: {efficiency_gain:.1f}倍")
    
    return results

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
    print("性能测试脚本")
    print("=" * 70)
    print(f"API地址: {API_BASE_URL}")
    print()
    
    # 检查服务器是否运行
    print("检查服务器状态...")
    if not check_server():
        print("错误: 后端服务器未运行，请先启动后端服务")
        print("启动命令: cd backend && python app.py")
        sys.exit(1)
    
    print("服务器运行正常\n")
    
    # 测试单条分析
    single_times = test_single_analysis()
    
    # 测试批量分析
    batch_results = test_batch_analysis([10, 50, 100])
    
    print("\n" + "=" * 70)
    print("性能测试完成！")
    print("=" * 70)

