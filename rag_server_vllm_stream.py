import multiprocessing
import torch
import httpx  # 改用异步HTTP客户端
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
from typing import Dict, AsyncGenerator, Optional
import json
import asyncio
from transformers import AutoTokenizer
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 在全局添加
MAX_CONTEXT_LENGTH = 2048  # 根据实际模型调整
MIN_OUTPUT_TOKENS = 50  # 至少保留的生成token数


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
VLLM_ENDPOINT = "http://vllm_1:8000/v1/completions"
MODEL_NAME = "deepseekr1"

# 提示模板管理
PROMPT_TEMPLATES = {
    "rag": {
        "template": """上下文: {context}
            如果上述上下文为空或与问题无关，请基于你的知识和能力回答问题。
            如果上下文相关，请严格基于上述上下文提供准确、一致的回答。
            问题是: {query}
            请给出清晰、确定的答案:""",
        "description": "用于RAG检索增强生成的提示模板"
    }
}

# 多进程设置
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)


async def load_vector_store(vectorstore_path: str) -> FAISS:
    """异步包装同步加载方法"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="/workspace/models/bge_large_zh_v1.5")
        # 使用异步执行器包装同步方法
        loop = asyncio.get_event_loop()
        vectorstore = await loop.run_in_executor(
            None, 
            lambda: FAISS.load_local(
                vectorstore_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        )
        logger.info("向量存储加载成功！")
        return vectorstore
    except Exception as e:
        logger.error(f"加载向量存储失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="向量存储加载失败")

async def search_documents_with_scores(vectorstore: FAISS, query: str, k: int = 5):
    """异步搜索文档块"""
    try:
        # 使用同步方法并包装在run_in_executor中
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: vectorstore.similarity_search_with_score(query, k=k)
        )
        logger.info(f"找到 {len(results)} 个相关文档块")
        return results
    except Exception as e:
        logger.error(f"搜索文档块失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="搜索文档块失败")

class QueryRequest(BaseModel):
    query: str
    model: str = MODEL_NAME
    stream: bool = True
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.05  # 新增重复惩罚参数
    timeout: Optional[float] = 60.0  # 添加超时参数


async def call_vllm_service_streaming(prompt: str, params: QueryRequest) -> AsyncGenerator[str, None]:
    """流式调用服务"""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": params.model,
        "prompt": prompt,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "repetition_penalty": params.repetition_penalty,  # 添加重复惩罚参数
        "stream": True  # 始终为流式
    }
    logger.debug(f"请求参数: {data}")
    try:
        async with httpx.AsyncClient(timeout=params.timeout) as client:
            # 添加详细日志
            logger.info(f"连接到 {VLLM_ENDPOINT} 使用参数: {params}")
            try:
                async with client.stream(
                    "POST",
                    VLLM_ENDPOINT,
                    headers=headers,
                    json=data,
                    timeout=params.timeout
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"vLLM服务返回错误: {response.status_code} - {error_text}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"vLLM服务错误: {error_text}"
                        )

                    async for chunk in response.aiter_lines():
                        if not chunk.strip():
                            continue
                            
                        # 处理data:前缀
                        if chunk.startswith("data:"):
                            chunk = chunk[5:].strip()
                        
                        # 如果是[DONE]，跳过
                        if chunk == "[DONE]":
                            continue
                                
                        try:
                            json_chunk = json.loads(chunk)
                            # 记录完整JSON结构用于调试
                            logger.debug(f"接收到JSON结构: {json.dumps(json_chunk, ensure_ascii=False)}")
                            
                            content = None
                            
                            # 尝试OpenAI格式: choices[0].delta.content
                            if "choices" in json_chunk and len(json_chunk["choices"]) > 0:
                                choice = json_chunk["choices"][0]
                                
                                # 尝试delta.content格式
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                # 尝试text格式
                                elif "text" in choice:
                                    content = choice["text"]
                                # 尝试直接content格式
                                elif "content" in choice:
                                    content = choice["content"]
                            
                            if content:
                                yield content
                            
                        except json.JSONDecodeError:
                            logger.warning(f"解析JSON失败: {chunk}")
                        except Exception as e:
                            logger.error(f"处理响应块失败: {str(e)}")
                            logger.debug(f"问题块: {chunk}")

            except httpx.ReadTimeout:
                logger.error("流式读取超时")
                yield json.dumps({
                    "code": 504,
                    "error": "模型响应超时"
                })
            except httpx.RequestError as e:
                logger.error(f"请求vLLM服务失败: {str(e)}")
                yield json.dumps({
                    "code": 503,
                    "error": f"服务暂时不可用: {str(e)}"
                })
            except Exception as e:
                logger.error(f"流式处理未知错误: {str(e)}")
                yield json.dumps({
                    "code": 500,
                    "error": f"内部服务器错误: {str(e)}"
                })

    except httpx.ConnectError as e:
        logger.error(f"连接失败: {str(e)}")
        logger.error("可能原因：\n"
                     "1. 服务未启动\n"
                     "2. 防火墙阻止\n"
                     "3. IP/端口错误")
        yield json.dumps({
            "code": 503,
            "error": f"连接服务失败: {str(e)}"
        })


async def generate_streaming_answer(query: str, params: QueryRequest) -> AsyncGenerator[str, None]:
    """生成流式回答（带结构化包装）"""
    try:
        # RAG处理流程
        results = await search_documents_with_scores(vectorstore, query, k=3)

        # 1. 计算固定部分的token数
        template = PROMPT_TEMPLATES["rag"]["template"]
        empty_context_prompt = template.format(context="", query=query)
        fixed_tokens = calculate_tokens(empty_context_prompt)

        # 2. 计算可用token空间
        max_available_tokens = MAX_CONTEXT_LENGTH - MIN_OUTPUT_TOKENS
        remaining_tokens = max_available_tokens - fixed_tokens
        
        if remaining_tokens <= 0:
            yield json.dumps({
                "code": 400,
                "msg": "问题过长，请缩短问题后重试",
                "data": None
            })
            return
        
        # 3. 动态构建上下文
        context_parts = []
        current_tokens = 0
        for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
            doc_text = f"[置信度: {score:.4f}] {doc.page_content}"
            doc_tokens = calculate_tokens(doc_text)
            
            if current_tokens + doc_tokens > remaining_tokens:
                # 截断最后一个文档
                truncated_text = tokenizer.decode(
                    tokenizer.encode(
                        doc_text, 
                        max_length=remaining_tokens - current_tokens,
                        truncation=True,
                        add_special_tokens=False
                    )
                )
                if truncated_text:
                    context_parts.append(truncated_text)
                break
            else:
                context_parts.append(doc_text)
                current_tokens += doc_tokens
        
        context = "\n".join(context_parts)
        prompt = template.format(context=context, query=query)

        # 4. 计算实际使用的token数
        used_tokens = calculate_tokens(prompt)
        available_output_tokens = MAX_CONTEXT_LENGTH - used_tokens
        
        # 自动调整max_tokens
        adjusted_max_tokens = min(
            available_output_tokens,
            params.max_tokens or 300,
            MAX_CONTEXT_LENGTH - used_tokens
        )
        params.max_tokens = max(adjusted_max_tokens, 1)  # 至少生成1个token
        
        logger.info(f"Token使用情况 | 输入: {used_tokens} | 输出: {params.max_tokens}")

        async for raw_chunk in call_vllm_service_streaming(prompt, params):
            try:
                # 检查是否已经是JSON结构（错误信息）
                if isinstance(raw_chunk, str) and raw_chunk.startswith("{") and "code" in raw_chunk:
                    yield raw_chunk
                    continue
                
                # 第一步：强制类型转换确保安全
                processed_chunk = _process_chunk(raw_chunk)
                
                # 第二步：构建标准响应结构
                response = {
                    "code": 200,
                    "msg": "成功",
                     "data": {
                        "text": processed_chunk  # 修改字段名
                    }
                }
                yield json.dumps(response, ensure_ascii=False)

            except Exception as e:
                logger.error(f"块处理失败: {str(e)}")
                yield json.dumps({
                    "code": 500,
                    "msg": f"数据处理错误: {str(e)}",
                    "data": None
                })

    except Exception as e:
        logger.error(f"生成流式答案失败: {str(e)}", exc_info=True)
        yield json.dumps({
            "code": 500,
            "msg": str(e),
            "data": None
        })

def _process_chunk(chunk):
    """统一数据类型处理"""
    # 处理字节类型
    if isinstance(chunk, bytes):
        return chunk.decode('utf-8', errors='replace')
    
    # 处理数值类型
    if isinstance(chunk, (int, float)):
        return str(chunk)
    
    # 处理其他可序列化类型
    try:
        json.dumps(chunk)  # 测试可序列化性
        return chunk
    except TypeError:
        return repr(chunk)  # 不可序列化时转为字符串表示

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global vectorstore, tokenizer
    vectorstore_path = "/workspace/code/vectorstore"
    vectorstore = await load_vector_store(vectorstore_path)
    # 初始化tokenizer（需要与vLLM服务端模型一致）
    try:
        tokenizer = AutoTokenizer.from_pretrained("/workspace/models/DeepSeek-R1-Distill-Qwen-14B")
        logger.info("Tokenizer加载成功")
    except Exception as e:
        logger.error(f"Tokenizer加载失败: {str(e)}")
        raise
def calculate_tokens(text: str) -> int:
    """计算文本的token数"""
    return len(tokenizer.encode(text, add_special_tokens=False))

@app.post("/generate")
async def generate_answer_endpoint(request: QueryRequest):
    """统一生成入口，支持流式和非流式"""  
    async def sse_event_generator():
        """SSE 格式事件生成器"""
        async for chunk in generate_streaming_answer(request.query, request):
            # 添加 SSE 格式包装
            yield f"{chunk}\n"
        yield f"{json.dumps({'code': 200, 'msg': '成功', 'data': '[done]'}, ensure_ascii=False)}\n\n"
        # yield "[DONE]\n"  # 流结束标记
    
    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={"X-Stream-Type": "structured"}  # 自定义头标识结构化流
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("启动服务...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8084,
        log_config=None,  # 禁用uvicorn默认日志
        timeout_keep_alive=300  # 保持长连接
    )