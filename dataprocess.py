# pip install langchain python-docx pandas PyPDF2 sentence-transformers faiss-cpu openpyxl
# pip install pypdf docx2txt
# pip install -U langchain-community
# pip install -U langchain-huggingface
# pip install --upgrade cryptography

import os

# 设置环境变量（必须在导入其他库之前）
os.environ['OPENBLAS_NUM_THREADS'] = '64'  # 或更低，如 32 或 16
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

from langchain.document_loaders import Docx2txtLoader, CSVLoader, PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS
import pandas as pd
from langchain.docstore.document import Document

# 1. 加载文档
def load_documents(file_path):
    """
    根据文件类型加载文档
    """
    if file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return load_excel(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_path}")
    
    documents = loader.load()
    return documents

# 2. 加载 Excel 文件
def load_excel(file_path):
    """
    加载 Excel 文件并转换为文档
    """
    df = pd.read_excel(file_path, engine='openpyxl')
    documents = []
    for index, row in df.iterrows():
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(Document(page_content=text))
    return documents

# 3. 分段处理
def split_documents(documents):
    """
    将文档分割成适合处理的块
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个块的大小
        chunk_overlap=200  # 块之间的重叠部分
    )
    texts = text_splitter.split_documents(documents)
    return texts

# 4. 向量化并保存
def vectorize_and_save(texts, save_path):
    """
    将文本块向量化并保存到本地
    """
    # 使用 HuggingFace 嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name="/data02/bge_large_zh_1.5/")

    # 创建向量存储
    vectorstore = FAISS.from_documents(texts, embeddings)

    # 保存向量存储到本地
    vectorstore.save_local(save_path)
    print(f"向量存储已保存到: {save_path}")

# 5. 遍历文件夹并加载所有文件
def load_all_documents_from_folder(folder_path):
    """
    遍历文件夹，加载所有支持的文件
    """
    all_documents = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                print(f"正在加载文件: {file_path}")
                documents = load_documents(file_path)
                all_documents.extend(documents)
            except ValueError as e:
                print(f"跳过不支持的文件: {file_path} ({e})")
            except Exception as e:
                print(f"加载文件失败: {file_path} ({e})")
    return all_documents

# 主函数
def main():
    # 数据文件夹路径
    data_folder = "./data/"  # 替换为实际的文件夹路径

    # 向量存储保存路径
    save_path = "./vectorstore"

    # 加载所有文档
    all_documents = load_all_documents_from_folder(data_folder)

    # 分段处理
    all_texts = split_documents(all_documents)

    # 向量化并保存
    vectorize_and_save(all_texts, save_path)

if __name__ == "__main__":
    main()