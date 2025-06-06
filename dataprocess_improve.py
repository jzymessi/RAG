# 先进的文档处理方案
# pip install langchain sentence-transformers faiss-cpu numpy scikit-learn
# pip install transformers torch nltk spacy
# pip install -U langchain-community langchain-huggingface

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from collections import defaultdict
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

# 设置环境变量
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '8'

from langchain.document_loaders import Docx2txtLoader, CSVLoader, PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd
from langchain.docstore.document import Document

# 方案1：基于语义相似度的动态分块
class SemanticChunker:
    """基于语义相似度的智能分块器"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
    
    def chunk_by_semantic_similarity(self, text, max_chunk_size=1000):
        """基于语义相似度进行分块"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return [text]
        
        # 计算句子嵌入
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            # 计算当前句子与当前块的平均相似度
            chunk_embedding = np.mean([embeddings[j] for j in range(i-len(current_chunk), i)], axis=0)
            similarity = cosine_similarity([embeddings[i]], [chunk_embedding])[0][0]
            
            # 检查是否应该开始新块
            current_chunk_text = ' '.join(current_chunk)
            if (similarity < self.similarity_threshold or 
                len(current_chunk_text) + len(sentences[i]) > max_chunk_size):
                
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

# 方案2：基于主题建模的分块
class TopicBasedChunker:
    """基于主题建模的分块器"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', n_topics=5):
        self.model = SentenceTransformer(model_name)
        self.n_topics = n_topics
    
    def chunk_by_topics(self, text, max_chunk_size=1000):
        """基于主题聚类进行分块"""
        sentences = sent_tokenize(text)
        if len(sentences) <= self.n_topics:
            return [text]
        
        # 获取句子嵌入
        embeddings = self.model.encode(sentences)
        
        # 使用K-means聚类识别主题
        n_clusters = min(self.n_topics, len(sentences))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # 按主题分组句子
        topic_sentences = defaultdict(list)
        for i, cluster in enumerate(clusters):
            topic_sentences[cluster].append((i, sentences[i]))
        
        # 将同主题的句子组合成块
        chunks = []
        for topic_id, sentence_list in topic_sentences.items():
            # 按原文顺序排序
            sentence_list.sort(key=lambda x: x[0])
            topic_text = ' '.join([s[1] for s in sentence_list])
            
            # 如果主题块太大，进一步分割
            if len(topic_text) > max_chunk_size:
                sub_chunks = self._split_large_chunk(topic_text, max_chunk_size)
                chunks.extend(sub_chunks)
            else:
                chunks.append(topic_text)
        
        return chunks
    
    def _split_large_chunk(self, text, max_size):
        """分割过大的块"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            if len(' '.join(current_chunk)) + len(sentence) > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
            else:
                current_chunk.append(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

# 方案3：基于图的文档结构分析
class GraphBasedChunker:
    """基于图结构的文档分块器"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', min_community_size=3):
        self.model = SentenceTransformer(model_name)
        self.min_community_size = min_community_size
    
    def chunk_by_graph_communities(self, text, similarity_threshold=0.6):
        """基于图社区检测进行分块"""
        sentences = sent_tokenize(text)
        if len(sentences) <= self.min_community_size:
            return [text]
        
        # 计算句子嵌入
        embeddings = self.model.encode(sentences)
        
        # 构建相似度图
        G = nx.Graph()
        for i in range(len(sentences)):
            G.add_node(i, text=sentences[i])
        
        # 添加边（基于相似度）
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity > similarity_threshold:
                    G.add_edge(i, j, weight=similarity)
        
        # 检测社区
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # 将社区转换为文本块
        chunks = []
        for community in communities:
            if len(community) >= self.min_community_size:
                # 按原文顺序排序
                sorted_indices = sorted(list(community))
                chunk_text = ' '.join([sentences[i] for i in sorted_indices])
                chunks.append(chunk_text)
        
        # 处理未分组的句子
        assigned_sentences = set()
        for community in communities:
            assigned_sentences.update(community)
        
        unassigned = [i for i in range(len(sentences)) if i not in assigned_sentences]
        if unassigned:
            unassigned_text = ' '.join([sentences[i] for i in unassigned])
            chunks.append(unassigned_text)
        
        return chunks if chunks else [text]

# 方案4：层次化文档结构
class HierarchicalChunker:
    """层次化文档结构分块器"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def create_hierarchical_chunks(self, text, levels=3):
        """创建层次化的文档块"""
        # 第一层：原始文本的主要段落
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # 第二层：段落内的关键句子
        key_sentences = []
        for para in paragraphs:
            sentences = sent_tokenize(para)
            if len(sentences) > 3:
                # 选择最重要的句子
                key_sentences.extend(self._extract_key_sentences(sentences, max_sentences=2))
            else:
                key_sentences.extend(sentences)
        
        # 第三层：整体摘要
        try:
            summary = self.summarizer(text[:1000], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        except:
            summary = ' '.join(key_sentences[:3])
        
        return {
            'level_1_paragraphs': paragraphs,
            'level_2_key_sentences': key_sentences,
            'level_3_summary': summary,
            'full_text': text
        }
    
    def _extract_key_sentences(self, sentences, max_sentences=2):
        """提取关键句子"""
        if len(sentences) <= max_sentences:
            return sentences
        
        # 计算句子嵌入
        embeddings = self.model.encode(sentences)
        
        # 计算句子重要性（基于与其他句子的平均相似度）
        importance_scores = []
        for i, emb in enumerate(embeddings):
            avg_similarity = np.mean([cosine_similarity([emb], [other_emb])[0][0] 
                                    for j, other_emb in enumerate(embeddings) if i != j])
            importance_scores.append(avg_similarity)
        
        # 选择最重要的句子
        top_indices = np.argsort(importance_scores)[-max_sentences:]
        return [sentences[i] for i in sorted(top_indices)]

# 方案5：自适应分块策略
class AdaptiveChunker:
    """自适应分块策略"""
    
    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.topic_chunker = TopicBasedChunker()
        self.graph_chunker = GraphBasedChunker()
        self.hierarchical_chunker = HierarchicalChunker()
    
    def adaptive_chunk(self, text, doc_type='auto'):
        """根据文档特征自适应选择分块策略"""
        if doc_type == 'auto':
            doc_type = self._detect_document_type(text)
        
        if doc_type == 'narrative':
            # 叙述性文档使用语义分块
            return self.semantic_chunker.chunk_by_semantic_similarity(text)
        elif doc_type == 'technical':
            # 技术文档使用主题分块
            return self.topic_chunker.chunk_by_topics(text)
        elif doc_type == 'academic':
            # 学术文档使用图结构分块
            return self.graph_chunker.chunk_by_graph_communities(text)
        elif doc_type == 'complex':
            # 复杂文档使用层次化结构
            hierarchical = self.hierarchical_chunker.create_hierarchical_chunks(text)
            return hierarchical['level_1_paragraphs']
        else:
            # 默认使用语义分块
            return self.semantic_chunker.chunk_by_semantic_similarity(text)
    
    def _detect_document_type(self, text):
        """检测文档类型"""
        # 检测技术术语密度
        technical_terms = len(re.findall(r'\b(?:API|HTTP|JSON|XML|SQL|Python|JavaScript|algorithm|function|class|method)\b', text, re.IGNORECASE))
        
        # 检测学术特征
        academic_features = len(re.findall(r'\b(?:research|study|analysis|hypothesis|conclusion|methodology|literature|citation)\b', text, re.IGNORECASE))
        
        # 检测叙述特征
        narrative_features = len(re.findall(r'\b(?:story|narrative|character|plot|once upon|beginning|end|happened)\b', text, re.IGNORECASE))
        
        word_count = len(text.split())
        
        if technical_terms / word_count > 0.02:
            return 'technical'
        elif academic_features / word_count > 0.01:
            return 'academic'
        elif narrative_features / word_count > 0.01:
            return 'narrative'
        elif word_count > 2000:
            return 'complex'
        else:
            return 'general'

# 统一的文档处理接口
class AdvancedDocumentProcessor:
    """先进的文档处理器"""
    
    def __init__(self, strategy='adaptive'):
        self.strategy = strategy
        self.adaptive_chunker = AdaptiveChunker()
        
    def process_documents(self, documents, save_path):
        """处理文档并保存向量存储"""
        all_chunks = []
        
        for doc in documents:
            print(f"处理文档: {doc.metadata.get('source', 'Unknown')}")
            
            # 选择处理策略
            if self.strategy == 'adaptive':
                chunks = self.adaptive_chunker.adaptive_chunk(doc.page_content)
            elif self.strategy == 'semantic':
                chunks = SemanticChunker().chunk_by_semantic_similarity(doc.page_content)
            elif self.strategy == 'topic':
                chunks = TopicBasedChunker().chunk_by_topics(doc.page_content)
            elif self.strategy == 'graph':
                chunks = GraphBasedChunker().chunk_by_graph_communities(doc.page_content)
            elif self.strategy == 'hierarchical':
                hierarchical = HierarchicalChunker().create_hierarchical_chunks(doc.page_content)
                chunks = hierarchical['level_1_paragraphs']
            else:
                chunks = [doc.page_content]  # 回退到原始文档
            
            # 创建文档对象
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata.update({
                    'chunk_id': i,
                    'chunk_strategy': self.strategy,
                    'chunk_count': len(chunks)
                })
                all_chunks.append(Document(page_content=chunk, metadata=metadata))
        
        # 向量化并保存
        print(f"开始向量化 {len(all_chunks)} 个文档块...")
        embeddings = HuggingFaceEmbeddings(model_name="/data02/bge_large_zh_1.5/")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(save_path)
        
        print(f"处理完成，共生成 {len(all_chunks)} 个文档块")
        return vectorstore

# 使用示例
def main():
    # 加载文档（使用你原来的加载函数）
    data_folder = "./data/"
    
    # 选择处理策略：'adaptive', 'semantic', 'topic', 'graph', 'hierarchical'
    processor = AdvancedDocumentProcessor(strategy='adaptive')
    
    # 加载所有文档
    all_documents = []
    for root, _, files in os.walk(data_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                if file_path.endswith(('.txt', '.pdf', '.docx')):
                    if file_path.endswith('.txt'):
                        loader = TextLoader(file_path)
                    elif file_path.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif file_path.endswith('.docx'):
                        loader = Docx2txtLoader(file_path)
                    
                    documents = loader.load()
                    all_documents.extend(documents)
                    print(f"加载文件: {file_path}")
            except Exception as e:
                print(f"加载失败: {file_path} - {e}")
    
    if all_documents:
        # 处理文档
        processor.process_documents(all_documents, "./advanced_vectorstore")
    else:
        print("没有找到任何支持的文档文件")

if __name__ == "__main__":
    main()