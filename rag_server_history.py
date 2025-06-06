import multiprocessing
import torch
import httpx  # 改用异步HTTP客户端
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
from typing import Dict, AsyncGenerator, Optional, List
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
VLLM_ENDPOINT = "http://125.122.39.29:8000/v1/chat/completions"
MODEL_NAME = "Qwen-QwQ-32B"
API_KEY = "vllm"
REQUEST_TIMEOUT = 120  # 增加默认超时时间为120秒

# 提示模板管理
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


async def load_vector_store(vectorstore_path: str) -> FAISS:
    """异步包装同步加载方法"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="./models/bge_large_zh_1.5/")
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
    history: Optional[List[Dict[str, str]]] = None  # 格式: [{"user": "问题", "assistant": "回答"}]
    model: str = MODEL_NAME
    stream: bool = False
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.9
    use_rag: bool = True  # 是否使用RAG
    timeout: Optional[int] = REQUEST_TIMEOUT  # 允许客户端自定义超时时间

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

async def call_vllm_service_non_streaming(prompt: str, params: QueryRequest) -> str:
    """非流式调用服务"""
    headers = {"Authorization": f"Bearer {API_KEY}","Content-Type": "application/json"}
    data = {
        "model": params.model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "stream": False  # 始终为非流式
    }

    # 使用流式方式组装非流式结果，以避免超时问题
    async with httpx.AsyncClient(timeout=params.timeout) as client:
        try:
            # 使用流式请求但手动组装完整响应
            async with client.stream(
                "POST",
                VLLM_ENDPOINT,
                headers=headers,
                json={**data, "stream": True},  # 强制使用流式
                timeout=params.timeout
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"vLLM服务返回错误: {response.status_code} - {error_text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"vLLM服务错误: {error_text}"
                    )
                
                # 手动收集所有块
                full_response = ""
                async for chunk in response.aiter_lines():
                    # print(chunk)
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
                        if "choices" in json_chunk:
                            content = json_chunk["choices"][0]["delta"].get("content", "")
                            full_response += content
                    except json.JSONDecodeError:
                        logger.warning(f"解析JSON失败: {chunk}")
                    except Exception as e:
                        logger.error(f"处理响应块失败: {str(e)}")
                
                return full_response
                
        except httpx.ReadTimeout:
            logger.error("读取超时，可能是模型生成时间过长")
            raise HTTPException(status_code=504, detail="模型响应超时，请尝试减少输出长度或使用流式请求")
        except httpx.RequestError as e:
            logger.error(f"请求vLLM服务失败: {str(e)}")
            raise HTTPException(status_code=503, detail=f"服务暂时不可用: {str(e)}")
        except Exception as e:
            logger.error(f"未知错误: {str(e)}")
            raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

async def call_vllm_service_streaming(prompt: str, params: QueryRequest) -> AsyncGenerator[str, None]:
    """流式调用服务"""
    headers = {"Authorization": f"Bearer {API_KEY}","Content-Type": "application/json"}
    data = {
        "model": params.model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "stream": True  # 始终为流式
    }

    async with httpx.AsyncClient(timeout=params.timeout) as client:
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
                    # print(chunk)
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
                        if "choices" in json_chunk:
                            content = json_chunk["choices"][0]["delta"].get("content", "")
                            if content:  # 只在有内容时才yield
                                yield content
                    except json.JSONDecodeError:
                        logger.warning(f"解析JSON失败: {chunk}")
                    except Exception as e:
                        logger.error(f"处理响应块失败: {str(e)}")

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

async def generate_answer(query: str, params: QueryRequest):
    """生成非流式回答"""
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
        
        # 使用非流式调用
        final_answer = await call_vllm_service_non_streaming(prompt, params)
        return final_answer

    except Exception as e:
        logger.error(f"生成答案失败: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


async def generate_streaming_answer(query: str, params: QueryRequest) -> AsyncGenerator[str, None]:
    """生成流式回答（带结构化包装）"""
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
            try:
                # 第一步：强制类型转换确保安全
                processed_chunk = _process_chunk(raw_chunk)
                
                # 第二步：构建标准响应结构
                response = {
                    "code": 200,
                    "msg": "成功",
                    "data": processed_chunk
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
    vectorstore_path = "./vectorstore"
    vectorstore = await load_vector_store(vectorstore_path)
    # 初始化tokenizer（需要与vLLM服务端模型一致）
    try:
        tokenizer = AutoTokenizer.from_pretrained("./models/QwQ-32B")
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
    if request.stream:
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
    
    try:
        # 直接使用非流式调用
        final_answer = await generate_answer(request.query, request)
        
        return {
            "code": 200,
            "msg": "成功",
            "data": {
                "answer": final_answer,
                "model": request.model
            }
        }
    except HTTPException as e:
        logger.error(f"处理请求失败: {e.detail}")
        return {"code": e.status_code, "msg": e.detail}
    except Exception as e:
        logger.error(f"处理请求失败: {str(e)}")
        return {"code": 500, "msg": f"内部服务器错误: {str(e)}"}

# 添加OpenAI兼容接口，方便Chatbox等客户端接入
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 300
    stream: bool = False

@app.post("/v1/chat/completions")
async def openai_compatible_endpoint(request: OpenAIRequest):
    """OpenAI兼容接口，方便Chatbox等客户端接入"""
    history = []
    # 提取用户问题（最后一条）
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    user_message = user_messages[-1].content if user_messages else "请提供问题"
    
    # 构建历史记录（兼容 OpenAI 格式）
    history = []
    for msg in request.messages:
        if msg.role == "user":
            history.append({"user": msg.content, "assistant": ""})
        elif msg.role == "assistant" and history:
            history[-1]["assistant"] = msg.content

    # 转换为内部请求格式
    internal_request = QueryRequest(
        query=user_message,
        history=history,
        model=request.model,
        stream=request.stream,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        use_rag=True  # 默认使用RAG
    )
    
    if request.stream:
        async def event_generator():
            # SSE格式前缀
            yield "data: " + json.dumps({
                "id": "chatcmpl-" + str(hash(user_message))[:10],
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            }) + "\n\n"
            
            try:
                async for chunk in generate_streaming_answer(user_message, internal_request):
                    # 检查是否为错误JSON
                    if chunk.startswith("{") and "error" in chunk:
                        error_data = json.loads(chunk)
                        yield "data: " + json.dumps({
                            "error": {
                                "message": error_data.get("error", "未知错误"),
                                "type": "server_error",
                                "code": error_data.get("code", 500)
                            }
                        }) + "\n\n"
                        return
                    
                    # 新增部分：解析并提取data字段内容
                    try:
                        chunk_data = json.loads(chunk)
                        content = chunk_data.get("data", "")  # 提取data字段
                    except json.JSONDecodeError:
                        content = chunk  # 容错处理


                    # 正常内容块
                    yield "data: " + json.dumps({
                        "id": "chatcmpl-" + str(hash(user_message))[:10],
                        "object": "chat.completion.chunk",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None
                        }]
                    },ensure_ascii=False) + "\n\n"
                
                # 发送结束标记
                yield "data: " + json.dumps({
                    "id": "chatcmpl-" + str(hash(user_message))[:10],
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }) + "\n\n"
                
                # 标记流结束
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"OpenAI兼容流处理错误: {str(e)}")
                yield "data: " + json.dumps({
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": 500
                    }
                }) + "\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    else:
        try:
            # 非流式响应
            answer = await generate_answer(user_message, internal_request)
            
            return {
                "id": "chatcmpl-" + str(hash(user_message))[:10],
                "object": "chat.completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(answer.split()),
                    "total_tokens": len(user_message.split()) + len(answer.split())
                }
            }
        except HTTPException as e:
            return {
                "error": {
                    "message": e.detail,
                    "type": "server_error",
                    "code": e.status_code
                }
            }
        except Exception as e:
            logger.error(f"OpenAI兼容接口错误: {str(e)}")
            return {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }

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

'''
curl -N  -X POST "http://localhost:8084/generate" -H "Content-Type: application/json" -d '{ "query": "你好","history":["user:你好，你叫什么名字？","assistant:我叫李琪，你好，很高兴认识你！"], "stream": true, "max_tokens": 200}'

curl -X POST "http://localhost:8084/v1/chat/completions" -H "Content-Type: application/json" -d '{ "model": "deepseekr1", "messages": [{"role": "user", "content": "luqia"}],"temperature": 0.7,"max_tokens": 200}'

'''

