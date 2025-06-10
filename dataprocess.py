# pip install langchain python-docx pandas PyPDF2 sentence-transformers faiss-cpu openpyxl
# pip install pypdf docx2txt
# pip install -U langchain-community langchain-huggingface
# pip install --upgrade cryptography

import os
import re

# 设置环境变量
os.environ['OPENBLAS_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

from langchain.document_loaders import Docx2txtLoader, CSVLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd
from langchain.docstore.document import Document

# 简单的句子分割器（不依赖NLTK）
class SimpleSentenceTokenizer:
    """简单的句子分割器"""
    
    def __init__(self):
        # 中文句子结束标点
        self.zh_sentence_endings = r'[。！？；!?;]'
        # 英文句子结束标点
        self.en_sentence_endings = r'[.!?;](?:\s|$)'
        # 综合句子分割模式
        self.sentence_pattern = f'({self.zh_sentence_endings}|{self.en_sentence_endings})'
    
    def split_sentences(self, text):
        """分割句子"""
        if not text.strip():
            return []
        
        # 使用正则表达式分割句子
        sentences = re.split(self.sentence_pattern, text)
        
        # 重新组合句子和标点
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence_content = sentences[i].strip()
            if i + 1 < len(sentences):
                sentence_ending = sentences[i + 1]
                if sentence_content:
                    result.append(sentence_content + sentence_ending)
            elif sentence_content:
                result.append(sentence_content)
        
        # 处理最后一个片段
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return [s.strip() for s in result if s.strip()]

class SmartTextSplitter:
    """智能文本分割器 - 修复版"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_tokenizer = SimpleSentenceTokenizer()
        
    def split_by_semantic_structure(self, text):
        """基于语义结构分割文本"""
        chunks = []
        
        # 1. 首先按段落分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        for paragraph in paragraphs:
            if not paragraph:
                continue
                
            # 如果当前段落加入后超过限制，先保存当前块
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # 保留重叠部分
                current_chunk = self._get_overlap_text(current_chunk) + "\n\n" + paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def split_by_sentences(self, text):
        """基于句子分割文本"""
        chunks = []
        sentences = self.sentence_tokenizer.split_sentences(text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # 保留最后几个句子作为重叠
                overlap_sentences = self.sentence_tokenizer.split_sentences(current_chunk)
                if len(overlap_sentences) >= 2:
                    current_chunk = " ".join(overlap_sentences[-2:]) + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def split_by_headings(self, text):
        """基于标题结构分割文本"""
        chunks = []
        
        # 检测标题模式（支持 Markdown 和数字编号）
        heading_patterns = [
            r'^#{1,6}\s+.+$',  # Markdown标题
            r'^第[一二三四五六七八九十\d]+[章节部分].*$',  # 中文章节
            r'^[一二三四五六七八九十\d]+[、.]\s*.+$',  # 中文编号
            r'^\d+\.\s*.+$',  # 数字编号
            r'^[A-Z][A-Z\s]*$',  # 全大写标题
        ]
        
        lines = text.split('\n')
        current_section = ""
        current_heading = ""
        
        for line in lines:
            line_strip = line.strip()
            is_heading = any(re.match(pattern, line_strip) for pattern in heading_patterns)
            
            if is_heading:
                # 遇到新标题，保存前一个部分
                if current_section.strip():
                    section_text = f"{current_heading}\n{current_section}".strip()
                    if len(section_text) > self.chunk_size:
                        # 大段落进一步分割
                        sub_chunks = self.split_by_semantic_structure(section_text)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(section_text)
                
                current_heading = line_strip
                current_section = ""
            else:
                current_section += "\n" + line
        
        # 添加最后一个部分
        if current_section.strip():
            section_text = f"{current_heading}\n{current_section}".strip()
            if len(section_text) > self.chunk_size:
                sub_chunks = self.split_by_semantic_structure(section_text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section_text)
            
        return chunks if chunks else [text]
    
    def _get_overlap_text(self, text):
        """获取重叠文本"""
        sentences = self.sentence_tokenizer.split_sentences(text)
        overlap_text = ""
        
        # 从后往前取句子，直到达到重叠长度
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.chunk_overlap:
                overlap_text = sentence + " " + overlap_text
            else:
                break
                
        return overlap_text.strip()

def detect_document_type(text):
    """检测文档类型以选择最佳分割策略"""
    # 检测标题数量
    heading_patterns = [
        r'^#{1,6}\s+.+$',
        r'^第[一二三四五六七八九十\d]+[章节部分].*$',
        r'^[一二三四五六七八九十\d]+[、.]\s*.+$',
        r'^\d+\.\s*.+$'
    ]
    
    lines = text.split('\n')
    heading_count = sum(1 for line in lines 
                       if any(re.match(pattern, line.strip()) for pattern in heading_patterns))
    
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    total_length = len(text)
    
    if heading_count > 3 and total_length > 2000:
        return "structured"  # 结构化文档
    elif paragraph_count > 5 and total_length > 1000:
        return "narrative"   # 叙述性文档
    else:
        return "simple"      # 简单文档

class DocumentNode:
    """文档节点类，用于构建文档树结构"""
    def __init__(self, content, metadata=None, parent=None):
        self.content = content
        self.metadata = metadata or {}
        self.parent = parent
        self.children = []
        self.level = 0 if parent is None else parent.level + 1
        
    def add_child(self, child):
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)
        
    def to_document(self):
        """转换为LangChain Document对象"""
        metadata = self.metadata.copy()
        metadata.update({
            'level': self.level,
            'parent_id': self.parent.metadata.get('node_id') if self.parent else None,
            'has_children': len(self.children) > 0
        })
        return Document(page_content=self.content, metadata=metadata)

class HierarchicalTextSplitter:
    """支持层级结构的文本分割器"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_tokenizer = SimpleSentenceTokenizer()
        
    def split_with_hierarchy(self, text, metadata=None):
        """基于层级结构分割文本"""
        root_node = DocumentNode(text, metadata)
        self._process_node(root_node)
        return root_node
    
    def _process_node(self, node):
        """处理节点内容，创建子节点"""
        text = node.content
        
        # 检测标题
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown标题
            r'^第[一二三四五六七八九十\d]+[章节部分](.+)$',  # 中文章节
            r'^[一二三四五六七八九十\d]+[、.]\s*(.+)$',  # 中文编号
            r'^\d+\.\s*(.+)$',  # 数字编号
        ]
        
        lines = text.split('\n')
        current_section = []
        current_heading = None
        
        for line in lines:
            line_strip = line.strip()
            is_heading = False
            heading_content = None
            
            for pattern in heading_patterns:
                match = re.match(pattern, line_strip)
                if match:
                    is_heading = True
                    heading_content = match.group(1).strip()
                    break
            
            if is_heading:
                # 处理前一个部分
                if current_section:
                    section_text = '\n'.join(current_section)
                    if len(section_text) > self.chunk_size:
                        # 需要进一步分割
                        sub_chunks = self._split_large_section(section_text)
                        for chunk in sub_chunks:
                            child = DocumentNode(chunk, node.metadata.copy())
                            node.add_child(child)
                    else:
                        child = DocumentNode(section_text, node.metadata.copy())
                        node.add_child(child)
                
                # 创建新的标题节点
                current_heading = heading_content
                current_section = [line_strip]
            else:
                current_section.append(line)
        
        # 处理最后一个部分
        if current_section:
            section_text = '\n'.join(current_section)
            if len(section_text) > self.chunk_size:
                sub_chunks = self._split_large_section(section_text)
                for chunk in sub_chunks:
                    child = DocumentNode(chunk, node.metadata.copy())
                    node.add_child(child)
            else:
                child = DocumentNode(section_text, node.metadata.copy())
                node.add_child(child)
    
    def _split_large_section(self, text):
        """分割大段文本"""
        chunks = []
        sentences = self.sentence_tokenizer.split_sentences(text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = self._get_overlap_text(current_chunk) + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text):
        """获取重叠文本"""
        sentences = self.sentence_tokenizer.split_sentences(text)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.chunk_overlap:
                overlap_text = sentence + " " + overlap_text
            else:
                break
        
        return overlap_text.strip()

def smart_split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """智能分割文档（支持层级结构）"""
    splitter = HierarchicalTextSplitter(chunk_size, chunk_overlap)
    all_chunks = []
    
    for doc in documents:
        text = doc.page_content
        if not text.strip():
            continue
        
        # 创建文档树
        root_node = splitter.split_with_hierarchy(text, doc.metadata)
        
        # 遍历文档树，收集所有节点
        def collect_nodes(node, node_id=0):
            node.metadata['node_id'] = node_id
            all_chunks.append(node.to_document())
            for i, child in enumerate(node.children):
                collect_nodes(child, f"{node_id}_{i}")
        
        collect_nodes(root_node)
    
    return all_chunks

def load_documents_enhanced(file_path):
    """增强的文档加载器"""
    try:
        if file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            return load_excel_enhanced(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")
        
        documents = loader.load()
        
        # 添加文件信息
        for doc in documents:
            doc.metadata.update({
                'file_type': file_path.split('.')[-1],
                'source_file': os.path.basename(file_path),
                'file_path': file_path
            })
        
        return documents
    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return []

def load_excel_enhanced(file_path):
    """增强的 Excel 加载器"""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        documents = []
        
        # 处理Excel数据
        if len(df.columns) <= 3:  # 简单表格，按行处理
            for index, row in df.iterrows():
                text = "\n".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notna(row[col])])
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            'file_type': 'excel',
                            'source_file': os.path.basename(file_path),
                            'row_index': index,
                            'file_path': file_path
                        }
                    )
                    documents.append(doc)
        else:  # 复杂表格，整体处理
            text = df.to_string(index=False)
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        'file_type': 'excel',
                        'source_file': os.path.basename(file_path),
                        'shape': f"{df.shape[0]}x{df.shape[1]}",
                        'file_path': file_path
                    }
                )
                documents.append(doc)
        
        return documents
    except Exception as e:
        print(f"加载Excel文件失败 {file_path}: {e}")
        return []

def preprocess_text(text):
    """文本预处理"""
    if not text:
        return ""
    
    # 清理多余的空白字符
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 规范化段落分隔
    text = re.sub(r'[ \t]+', ' ', text)       # 规范化空格
    text = text.strip()
    
    return text

def vectorize_and_save_enhanced(texts, save_path, batch_size=32):
    """增强的向量化保存（支持层级结构）"""
    if not texts:
        print("没有文档需要处理")
        return
    
    # 预处理文本
    processed_texts = []
    for doc in texts:
        processed_content = preprocess_text(doc.page_content)
        if processed_content:
            doc.page_content = processed_content
            processed_texts.append(doc)
    
    if not processed_texts:
        print("预处理后没有有效文档")
        return
    
    print(f"开始向量化 {len(processed_texts)} 个文档块...")
    
    # 使用 HuggingFace 嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/bge_large_zh_1.5/",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'batch_size': batch_size}
    )

    # 创建向量存储
    vectorstore = FAISS.from_documents(processed_texts, embeddings)

    # 保存向量存储到本地
    vectorstore.save_local(save_path)
    print(f"向量存储已保存到: {save_path}")
    print(f"共处理文档块数: {len(processed_texts)}")
    print(f"文档层级结构已保存")

def load_all_documents_from_folder_enhanced(folder_path):
    """增强的文件夹遍历加载"""
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return []
    
    all_documents = []
    file_count = 0
    
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                print(f"正在加载文件: {file_path}")
                documents = load_documents_enhanced(file_path)
                if documents:
                    all_documents.extend(documents)
                    file_count += 1
                    print(f"  - 成功加载 {len(documents)} 个文档块")
                else:
                    print(f"  - 文件为空或加载失败")
            except Exception as e:
                print(f"处理文件异常: {file_path} - {e}")
    
    print(f"总共加载了 {file_count} 个文件，{len(all_documents)} 个文档块")
    return all_documents

# 主函数
def main():
    # 数据文件夹路径
    data_folder = "./data/"
    
    # 向量存储保存路径
    save_path = "./vectorstore"
    
    # 分割参数
    chunk_size = 800
    chunk_overlap = 150
    
    # 加载所有文档
    print("开始加载文档...")
    all_documents = load_all_documents_from_folder_enhanced(data_folder)
    exit(0)
    if not all_documents:
        print("没有找到任何文档，请检查文件夹路径和文件格式。")
        return
    
    # 智能分段处理
    print("开始智能分割文档...")
    all_texts = smart_split_documents(all_documents, chunk_size, chunk_overlap)
    
    if not all_texts:
        print("文档分割后没有有效内容")
        return
    
    # 向量化并保存
    print("开始向量化处理...")
    vectorize_and_save_enhanced(all_texts, save_path)
    
    print("处理完成！")

if __name__ == "__main__":
    main()
