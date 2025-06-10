# RAG

一个基于 Python 的检索增强生成（Retrieval-Augmented Generation, RAG）系统，结合了 LangChain 框架与 OpenAI 的 GPT 模型，支持历史上下文对话和本地文档问答。

## 📌 项目简介

本项目旨在构建一个集成文档检索与大语言模型生成能力的智能问答系统，支持以下功能：

- 基于本地文档的问答系统
- 支持对话历史的多轮问答
- 集成 OpenAI GPT 模型进行响应生成
- 使用 LangChain 框架管理检索与生成流程
- 支持文档的父子级结构处理

## 🧱 项目结构

```
├── dataprocess.py              # 文档预处理脚本
├── dataprocess_improve.py      # 优化后的文档预处理脚本
├── rag_agent.py                # RAG 代理核心逻辑
├── rag_server.py               # 启动 RAG 服务的主程序
├── rag_server_history.py       # 支持对话历史的服务版本
├── test_rag.py                 # 测试脚本
├── hierarchy_processor.py      # 文档父子级处理框架
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

## 📄 文档处理

### 1. 基础文档处理

使用 `dataprocess.py` 或 `dataprocess_improve.py` 脚本对本地文档进行预处理，生成适用于检索的文档向量。

```bash
python dataprocess.py --input_dir ./docs --output_file processed_docs.json
```

### 2. 父子级文档处理

使用 `hierarchy_processor.py` 进行文档的父子级结构处理：

```python
from hierarchy_processor import process_document, NodeConfig

# 准备文本
text = """
第一章 介绍
    1.1 背景
        这是一个示例文档。
        用于测试层级结构。
    
    1.2 目标
        展示父子级关系。
        说明实现方式。
"""

# 使用缩进策略处理
documents = process_document(text, strategy_type='indentation')

# 使用语义策略处理
documents = process_document(text, strategy_type='semantic')
```

#### 自定义配置

```python
# 创建自定义配置
config = NodeConfig(
    min_length=20,           # 最小段落长度
    max_length=2000,         # 最大段落长度
    similarity_threshold=0.7, # 相似度阈值
    indent_size=2,           # 缩进大小
    level_limit=6           # 最大层级数
)

# 使用自定义配置处理文档
documents = process_document(text, strategy_type='semantic', config=config)
```

## 📊 父子级处理策略

### 1. 缩进策略 (IndentationHierarchyStrategy)

基于文本缩进确定父子级关系：

```python
# 示例文本
text = """
第一章 介绍
    1.1 背景
        这是背景内容
    1.2 目标
        这是目标内容
"""

# 使用缩进策略
documents = process_document(text, strategy_type='indentation')
```

### 2. 语义策略 (SemanticHierarchyStrategy)

基于段落间的语义相似度确定父子级关系：

```python
# 示例文本
text = """
人工智能概述
人工智能是计算机科学的一个分支。

机器学习
机器学习是人工智能的核心技术。

深度学习
深度学习是机器学习的一个子领域。
"""

# 使用语义策略
documents = process_document(text, strategy_type='semantic')
```

## ⚙️ 配置参数说明

### NodeConfig 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| min_length | int | 10 | 最小段落长度 |
| max_length | int | 1000 | 最大段落长度 |
| similarity_threshold | float | 0.8 | 语义相似度阈值 |
| indent_size | int | 4 | 缩进单位大小 |
| level_limit | int | 5 | 最大层级数 |

## 🔍 实际应用场景

### 1. 文档结构化

```python
# 处理长文档
with open('long_document.txt', 'r', encoding='utf-8') as f:
    text = f.read()
documents = process_document(text, strategy_type='semantic')
```

### 2. 知识库构建

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 处理文档
documents = process_document(text, strategy_type='semantic')

# 创建向量存储
embeddings = HuggingFaceEmbeddings(model_name="./models/bge_large_zh_1.5/")
vectorstore = FAISS.from_documents(documents, embeddings)
```

### 3. 文档分析

```python
def analyze_document_structure(documents):
    level_count = {}
    for doc in documents:
        level = doc.metadata['level']
        level_count[level] = level_count.get(level, 0) + 1
    
    print("文档层级统计：")
    for level, count in sorted(level_count.items()):
        print(f"层级 {level}: {count} 个节点")
```

## ⚠️ 注意事项

1. **策略选择**
   - 对于格式规范的文档，建议使用缩进策略
   - 对于非结构化文档，建议使用语义策略
   - 可以尝试两种策略，选择效果更好的

2. **参数调优**
   - 根据文档特点调整配置参数
   - 注意相似度阈值的设置
   - 控制最大层级数

3. **性能考虑**
   - 语义策略计算量较大
   - 大文档处理时注意内存使用
   - 考虑使用批处理

## 🔧 扩展开发

### 添加新的策略

```python
from hierarchy_processor import HierarchyStrategy

class CustomStrategy(HierarchyStrategy):
    def build_hierarchy(self, text: str) -> BaseNode:
        # 实现自定义的层级构建逻辑
        pass
    
    def validate_hierarchy(self, node: BaseNode) -> bool:
        # 实现自定义的验证逻辑
        pass
```

## ❓ 常见问题

1. **Q: 如何处理非结构化文档？**
   A: 使用语义策略，并适当调整相似度阈值。

2. **Q: 为什么某些段落没有被正确分类？**
   A: 检查相似度阈值设置，可能需要调整配置参数。

3. **Q: 处理大文档时内存占用过高？**
   A: 考虑使用批处理方式，或增加段落长度限制。

## 🤝 贡献指南

欢迎提交问题和改进建议。如果您想贡献代码，请遵循以下步骤：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🔗 参考链接

- [LangChain 官方文档](https://docs.langchain.com/)
- [OpenAI API 文档](https://platform.openai.com/docs)
