import multiprocessing
import torch
import httpx
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
from typing import Dict, AsyncGenerator, Optional, TypedDict, List
import json
import asyncio
from langgraph.graph import StateGraph, END
import datetime

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
VLLM_ENDPOINT = "http://10.100.0.3:1025/v1/chat/completions"
MODEL_NAME = "deepseekr1"
REQUEST_TIMEOUT = 120

# 智能体状态定义
class AgentState(TypedDict):
    query: str
    context: str
    prompt: str
    response: str
    use_rag: bool
    model_params: dict
    stream: bool
    memory: Dict[str, List[dict]]  # 记忆字段存储历史对话

# 多进程设置
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

async def load_vector_store(vectorstore_path: str) -> FAISS:
    """异步加载向量存储"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="/data02/bge_large_zh_1.5/")
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
        logger.error(f"加载向量存储失败: {str(e)}")
        raise HTTPException(status_code=500, detail="向量存储加载失败")

async def search_documents_with_scores(vectorstore: FAISS, query: str, k: int = 5):
    """异步搜索文档块"""
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: vectorstore.similarity_search_with_score(query, k=k)
        )
        return results
    except Exception as e:
        logger.error(f"搜索文档块失败: {str(e)}")
        raise HTTPException(status_code=500, detail="搜索文档块失败")

# 智能体节点定义
async def retrieve_node(state: AgentState):
    """检索上下文节点"""
    try:
        if state["use_rag"]:
            results = await search_documents_with_scores(vectorstore, state["query"], 3)
            context = "\n".join(
                [f"[置信度: {score:.4f}] {doc.page_content}" 
                 for doc, score in results]
            )
            return {"context": context}
        return {"context": "未使用RAG检索"}
    except Exception as e:
        logger.error(f"检索节点失败: {str(e)}")
        raise

async def prompt_node(state: AgentState):
    """动态生成提示词（结合记忆和RAG上下文）"""
    try:
        # 从记忆获取历史对话（最多保留3轮）
        history = state.get("memory", {}).get("history", [])[-3:]
        history_context = "\n".join(
            [f"用户: {h['query']}\n助手: {h['response']}" for h in history]
        )

        # 动态生成提示词
        if state["use_rag"] and state["context"]:
            prompt = f"""历史对话:\n{history_context}\n\n检索到的上下文:\n{state['context']}\n\n请基于以上信息回答问题: {state['query']}\n回答:"""
        else:
            prompt = f"""历史对话:\n{history_context}\n\n请直接回答问题: {state['query']}\n回答（200字以内）:"""
        
        return {"prompt": prompt}
    except Exception as e:
        logger.error(f"提示生成失败: {str(e)}")
        raise

async def llm_node(state: AgentState):
    """LLM调用节点"""
    try:
        async def stream_generator():
            async for chunk in call_vllm_service_streaming(
                state["prompt"], 
                QueryRequest(**state["model_params"])
            ):
                yield chunk

        if state["model_params"].get("stream", False):
            return {"response": stream_generator()}
        else:
            response = await call_vllm_service_non_streaming(
                state["prompt"], 
                QueryRequest(**state["model_params"])
            )
            return {"response": response}
    except Exception as e:
        logger.error(f"LLM调用失败: {str(e)}")
        raise

async def memory_node(state: AgentState):
    """记忆节点，存储对话历史"""
    try:
        if "memory" not in state:
            state["memory"] = {"history": []}
        
        # 添加当前对话到记忆
        state["memory"]["history"].append({
            "query": state["query"],
            "response": state.get("response", "")
        })
        
        # 限制记忆长度（最多10轮）
        if len(state["memory"]["history"]) > 10:
            state["memory"]["history"].pop(0)
        
        return {"memory": state["memory"]}
    except Exception as e:
        logger.error(f"记忆节点失败: {str(e)}")
        raise

async def log_node(state: AgentState):
    """日记记录节点"""
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": state["query"],
            "response": state.get("response", ""),
            "context": state["context"],
            "use_rag": state["use_rag"],
            "model_params": state["model_params"]
        }
        
        # 异步写入日志文件
        with open("conversation_logs.json", "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
        
        logger.info(f"对话日志已记录: {log_entry}")
        return {}
    except Exception as e:
        logger.error(f"日记记录失败: {str(e)}")
        raise

# 构建智能体工作流
def build_agent_workflow():
    """构建LangGraph工作流"""
    builder = StateGraph(AgentState)
    
    # 添加节点
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("build_prompt", prompt_node)
    builder.add_node("generate_response", llm_node)
    builder.add_node("update_memory", memory_node)
    builder.add_node("log_conversation", log_node)
    
    # 设置边
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "build_prompt")
    builder.add_edge("build_prompt", "generate_response")
    builder.add_edge("generate_response", "update_memory")
    builder.add_edge("update_memory", "log_conversation")
    builder.add_edge("log_conversation", END)
    
    return builder.compile()

class QueryRequest(BaseModel):
    query: str
    model: str = MODEL_NAME
    stream: bool = False
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.9
    use_rag: bool = True
    timeout: Optional[int] = REQUEST_TIMEOUT

async def call_vllm_service_non_streaming(prompt: str, params: QueryRequest) -> str:
    """非流式调用vLLM服务（原有实现保持不变）"""
    # 实现代码...

async def call_vllm_service_streaming(prompt: str, params: QueryRequest) -> AsyncGenerator[str, None]:
    """流式调用vLLM服务（原有实现保持不变）"""
    # 实现代码...

app = FastAPI()
agent = None
vectorstore = None

@app.on_event("startup")
async def startup_event():
    global vectorstore, agent
    vectorstore_path = "/root/rag/vectorstore"
    vectorstore = await load_vector_store(vectorstore_path)
    agent = build_agent_workflow()
    logger.info("智能体初始化完成")

@app.post("/generate")
async def generate_answer_endpoint(request: QueryRequest):
    """统一生成入口"""
    initial_state = {
        "query": request.query,
        "use_rag": request.use_rag,
        "model_params": {
            "model": request.model,
            "stream": request.stream,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "timeout": request.timeout
        },
        "stream": request.stream,
        "memory": {"history": []}  # 初始化记忆
    }

    if request.stream:
        async def stream_wrapper():
            async for event in agent.astream(initial_state):
                if "generate_response" in event:
                    chunk = event["generate_response"]["response"]
                    if isinstance(chunk, AsyncGenerator):
                        async for real_chunk in chunk:
                            yield f"{json.dumps(real_chunk)}\n"
                    else:
                        yield f"{json.dumps(chunk)}\n"
            yield "[DONE]\n"

        return StreamingResponse(
            stream_wrapper(),
            media_type="text/event-stream"
        )
    else:
        try:
            result = await agent.ainvoke(initial_state)
            return {
                "code": 200,
                "msg": "成功",
                "data": {
                    "answer": result["response"],
                    "model": request.model
                }
            }
        except Exception as e:
            logger.error(f"请求处理失败: {str(e)}")
            return {"code": 500, "msg": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)