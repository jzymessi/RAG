import multiprocessing
import torch
import httpx
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
from typing import AsyncGenerator, Optional, List, Dict
import json
import asyncio
from transformers import AutoTokenizer

# 全局配置
MAX_CONTEXT_LENGTH = 4096
MIN_OUTPUT_TOKENS = 200
MAX_HISTORY_TURNS = 2

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

# 提示模板
PROMPT_TEMPLATES = {
    "rag": {
        "template": """[历史对话]
{history}

[相关上下文]
{context}

[当前问题]
{query}

请根据上述信息给出专业回答：""",
        "description": "支持前端传入历史的提示模板"
    }
}

# 多进程设置
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = None  # 格式: [{"user": "问题", "assistant": "回答"}]
    model: str = MODEL_NAME
    stream: bool = True
    max_tokens: Optional[int] = 400
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.2
    timeout: Optional[float] = 60.0

def format_history(history: List[Dict], tokenizer, max_tokens: int) -> str:
    """正确处理历史顺序的版本"""
    formatted = []
    current_tokens = 0
    
    # 从最早开始保留，但优先保留最近的（取最后N个）
    valid_history = history[-MAX_HISTORY_TURNS:]  # 取最后2倍轮数作为处理池
    
    # 正序处理但控制token
    for item in valid_history:
        if not all(key in item for key in ['user', 'assistant']):
            continue
            
        interaction = f"用户：{item['user']}\n助手：{item['assistant']}"
        tokens = len(tokenizer.encode(interaction, add_special_tokens=False))
        
        if current_tokens + tokens > max_tokens:
            break
            
        formatted.append(interaction)
        current_tokens += tokens
    
    return "\n".join(formatted)

async def generate_streaming_answer(query: str, params: QueryRequest) -> AsyncGenerator[str, None]:
    try:
        # 1. 处理历史对话
        history_text = ""
        if params.history:
            # 计算历史可用token数（总token的30%）
            base_prompt = PROMPT_TEMPLATES["rag"]["template"].format(
                history="", context="", query=query)
            base_tokens = calculate_tokens(base_prompt)
            
            if params.history and len(params.history) > 2:
                history_ratio = 0.4  # 历史较长时分配更多token
            else:
                history_ratio = 0.25
            history_max_tokens = int((MAX_CONTEXT_LENGTH - MIN_OUTPUT_TOKENS - base_tokens) * history_ratio)
            
            history_text = format_history(params.history, tokenizer, history_max_tokens)

        # 2. 检索文档
        results = await search_documents_with_scores(vectorstore, query, k=3)
        
        # 3. 构建上下文
        context_parts = []
        available_tokens = MAX_CONTEXT_LENGTH - MIN_OUTPUT_TOKENS - calculate_tokens(history_text)
        context_max_tokens = max(available_tokens, 0)  # 确保不小于0
        
        for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
            doc_text = f"[相关度:{score:.2f}] {doc.page_content}"
            doc_tokens = calculate_tokens(doc_text)
            
            if context_max_tokens <= 0:
                break
                
            if doc_tokens > context_max_tokens:
                truncated = tokenizer.decode(
                    tokenizer.encode(
                        doc_text,
                        max_length=context_max_tokens,
                        truncation=True,
                        add_special_tokens=False
                    )
                )
                if truncated:
                    context_parts.append(truncated)
                    context_max_tokens -= calculate_tokens(truncated)
                break
            else:
                context_parts.append(doc_text)
                context_max_tokens -= doc_tokens
        
        context = "\n".join(context_parts)
        
        # 4. 构建完整提示
        prompt = PROMPT_TEMPLATES["rag"]["template"].format(
            history=history_text,
            context=context,
            query=query
        )
        
        # 5. 调整生成参数
        used_tokens = calculate_tokens(prompt)
        adjusted_max = MAX_CONTEXT_LENGTH - used_tokens
        safe_max_tokens = max(min(adjusted_max, params.max_tokens or 400), 1)  # 确保≥1
        params.max_tokens = safe_max_tokens
        
        logger.info(f"Token使用 | 总上下文: {used_tokens} | 可生成: {params.max_tokens}")

        # 6. 调用vLLM生成
        async for raw_chunk in call_vllm_service_streaming(prompt, params):
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
                    "text": processed_chunk
                }
            }
            yield json.dumps(response, ensure_ascii=False)

    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
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
    

# 异步加载向量存储，并返回FAISS实例
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

# 异步调用vLLM服务，并返回异步生成器
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

# 异步调用vLLM服务，并返回异步生成器
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