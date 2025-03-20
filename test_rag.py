import asyncio
import httpx
import json
import logging
from typing import List, Dict, Any
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RAG服务端点
RAG_ENDPOINT = "http://localhost:8084/generate"

async def test_rag_service():
    """测试RAG服务的基本功能"""
    headers = {"Content-Type": "application/json"}
    test_data = {
        "query": "请介绍一下深度学习",
        "model": "deepseekr1",
        "stream": True,
        "max_tokens": 200,
        "temperature": 0.6,
        "top_p": 0.9
    }
    
    logger.info(f"测试RAG服务: {RAG_ENDPOINT}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            start_time = time.time()
            async with client.stream(
                "POST",
                RAG_ENDPOINT,
                headers=headers,
                json=test_data,
                timeout=60.0
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"RAG服务错误: {response.status_code} - {error_text}")
                    return False
                
                logger.info("RAG服务返回状态码 200")
                logger.info("流式响应内容:")
                
                # 处理流式响应
                full_response = ""
                retrieved_docs = []
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        # 记录完整响应用于调试
                        logger.debug(f"接收到数据: {json.dumps(data, ensure_ascii=False)}")
                        
                        # 检查是否是最终消息
                        if data.get("data") == "[done]":
                            logger.info("流结束")
                            break
                            
                        # 处理常规消息
                        if data.get("code") == 200 and "data" in data:
                            text_chunk = data["data"].get("text", "")
                            if text_chunk:
                                full_response += text_chunk
                                logger.info(f"收到文本: {text_chunk}")
                                
                                # 尝试识别是否返回了上下文信息
                                if "[置信度:" in text_chunk:
                                    retrieved_docs.append(text_chunk)
                        # 处理错误消息
                        elif data.get("code") != 200:
                            logger.warning(f"收到错误响应: {data}")
                    except json.JSONDecodeError:
                        logger.warning(f"解析JSON失败: {line}")
                    except Exception as e:
                        logger.error(f"处理块时出错: {str(e)}")
                
                end_time = time.time()
                
                # 打印结果
                logger.info(f"总响应时间: {end_time - start_time:.2f} 秒")
                logger.info(f"完整响应长度: {len(full_response)} 字符")
                
                # 分析上下文是否被使用
                if retrieved_docs:
                    logger.info(f"检索到 {len(retrieved_docs)} 个文档块")
                else:
                    logger.info("响应中未检测到明确的文档块")
                    
                return True
                
    except httpx.ConnectError as e:
        logger.error(f"连接错误: {str(e)}")
        logger.error("可能原因:\n"
                    "1. 服务未启动\n"
                    "2. 防火墙阻止\n"
                    "3. IP/端口错误")
        return False
    except httpx.ReadTimeout:
        logger.error("请求超时 - RAG服务响应时间可能过长")
        return False
    except Exception as e:
        logger.error(f"意外错误: {str(e)}")
        return False

# 测试多个查询
async def run_multiple_tests(queries: List[str]):
    """使用多个不同查询运行测试"""
    headers = {"Content-Type": "application/json"}
    
    logger.info(f"使用 {len(queries)} 个查询运行多个RAG测试")
    
    results = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, query in enumerate(queries):
            test_data = {
                "query": query,
                "model": "deepseekr1",
                "stream": True,
                "max_tokens": 200
            }
            
            logger.info(f"测试 #{i+1}: 查询: '{query}'")
            start_time = time.time()
            
            try:
                async with client.stream(
                    "POST",
                    RAG_ENDPOINT,
                    headers=headers,
                    json=test_data,
                    timeout=60.0
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"测试 #{i+1} 失败，状态码 {response.status_code}: {error_text}")
                        results.append({"query": query, "success": False, "error": error_text})
                        continue
                    
                    # 处理流式响应（最小处理以关注成功/失败）
                    response_length = 0
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        try:
                            data = json.loads(line)
                            if data.get("data") == "[done]":
                                break
                                
                            if data.get("code") == 200 and "data" in data:
                                text = data["data"].get("text", "")
                                response_length += len(text)
                        except:
                            pass
                    
                    end_time = time.time()
                    results.append({
                        "query": query,
                        "success": True,
                        "time": end_time - start_time,
                        "response_length": response_length
                    })
                    logger.info(f"测试 #{i+1} 成功。响应时间: {end_time - start_time:.2f}秒, 长度: {response_length}")
            
            except Exception as e:
                logger.error(f"测试 #{i+1} 失败，错误: {str(e)}")
                results.append({"query": query, "success": False, "error": str(e)})
    
    # 结果摘要
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"测试摘要: {success_count}/{len(queries)} 个测试通过")
    
    return results

if __name__ == "__main__":
    logger.info("开始 RAG 服务测试...")
    
    # 基础测试
    basic_result = asyncio.run(test_rag_service())
    
    if basic_result:
        logger.info("✅ 基础 RAG 服务测试通过")
        
        # 运行扩展测试
        logger.info("使用多个查询运行扩展测试...")
        test_queries = [
            "人工智能的历史是什么？",
            "机器学习和深度学习有什么区别？",
            "神经网络是如何工作的？",
            "什么是自然语言处理？",
            "强化学习的应用场景有哪些？"
        ]
        
        extended_results = asyncio.run(run_multiple_tests(test_queries))
        
        # 输出摘要
        passed = sum(1 for r in extended_results if r["success"])
        logger.info(f"扩展测试: {passed}/{len(test_queries)} 个测试通过")
        
        if passed == len(test_queries):
            logger.info("✅ 所有扩展 RAG 测试通过")
        else:
            logger.warning("⚠️ 部分扩展 RAG 测试失败")
    else:
        logger.error("❌ 基础 RAG 服务测试失败")