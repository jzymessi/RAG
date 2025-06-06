# RAG

一个基于 Python 的检索增强生成（Retrieval-Augmented Generation, RAG）系统，结合了 LangChain 框架与 OpenAI 的 GPT 模型，支持历史上下文对话和本地文档问答。

## 📌 项目简介

本项目旨在构建一个集成文档检索与大语言模型生成能力的智能问答系统，支持以下功能：

- 基于本地文档的问答系统
- 支持对话历史的多轮问答
- 集成 OpenAI GPT 模型进行响应生成
- 使用 LangChain 框架管理检索与生成流程

## 🧱 项目结构

```
├── dataprocess.py              # 文档预处理脚本
├── dataprocess_improve.py      # 优化后的文档预处理脚本
├── rag_agent.py                # RAG 代理核心逻辑
├── rag_server.py               # 启动 RAG 服务的主程序
├── rag_server_history.py       # 支持对话历史的服务版本
├── test_rag.py                 # 测试脚本
├── requirements.txt            # 项目依赖列表
└── README.md                   # 项目说明文档
```

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/jzymessi/RAG.git
cd RAG
```

### 2. 创建并激活虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Windows 用户使用 venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 设置环境变量

创建一个 `.env` 文件，并添加你的 OpenAI API 密钥：

```env
OPENAI_API_KEY=你的OpenAI密钥
```

### 5. 运行服务

```bash
python rag_server.py
```

服务启动后，可通过指定的端口访问 RAG 系统。

## 🧪 示例测试

运行测试脚本以验证系统功能：

```bash
python test_rag.py
```

该脚本将加载示例文档，并通过 RAG 系统进行问答测试。

## 📄 文档预处理

使用 `dataprocess.py` 或 `dataprocess_improve.py` 脚本对本地文档进行预处理，生成适用于检索的文档向量。

```bash
python dataprocess.py --input_dir ./docs --output_file processed_docs.json
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request，共同完善本项目。

## 📄 许可证

本项目采用 MIT 许可证。

## 🔗 参考链接

- [LangChain 官方文档](https://docs.langchain.com/)
- [OpenAI API 文档](https://platform.openai.com/docs)