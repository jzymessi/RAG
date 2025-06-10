from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import re

@dataclass
class NodeConfig:
    """节点配置类"""
    min_length: int = 10          # 最小长度
    max_length: int = 1000        # 最大长度
    similarity_threshold: float = 0.8  # 相似度阈值
    indent_size: int = 4          # 缩进大小
    level_limit: int = 5          # 最大层级数

class BaseNode:
    """基础节点类"""
    def __init__(self, 
                 content: str, 
                 metadata: Optional[Dict[str, Any]] = None,
                 parent: Optional['BaseNode'] = None):
        self.content = content
        self.metadata = metadata or {}
        self.parent = parent
        self.children: List['BaseNode'] = []
        self.level = 0 if parent is None else parent.level + 1
        self.node_id = self._generate_node_id()
    
    def add_child(self, child: 'BaseNode') -> None:
        """添加子节点"""
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)
    
    def _generate_node_id(self) -> str:
        """生成节点ID"""
        if self.parent is None:
            return "0"
        return f"{self.parent.node_id}_{len(self.parent.children)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'level': self.level,
            'node_id': self.node_id,
            'parent_id': self.parent.node_id if self.parent else None,
            'children': [child.to_dict() for child in self.children]
        }
    
    def to_document(self) -> Document:
        """转换为LangChain Document对象"""
        metadata = self.metadata.copy()
        metadata.update({
            'level': self.level,
            'node_id': self.node_id,
            'parent_id': self.parent.node_id if self.parent else None,
            'has_children': len(self.children) > 0
        })
        return Document(page_content=self.content, metadata=metadata)

class HierarchyStrategy(ABC):
    """层级策略基类"""
    def __init__(self, config: NodeConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def build_hierarchy(self, text: str) -> BaseNode:
        """构建层级结构"""
        pass
    
    @abstractmethod
    def validate_hierarchy(self, node: BaseNode) -> bool:
        """验证层级结构"""
        pass

class SemanticHierarchyStrategy(HierarchyStrategy):
    """基于语义的层级策略"""
    def __init__(self, config: NodeConfig):
        super().__init__(config)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="./models/bge_large_zh_1.5/",
            model_kwargs={'device': 'cpu'}
        )
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """分割段落"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return [p for p in paragraphs if len(p) >= self.config.min_length]
    
    def _find_parent(self, embedding: List[float], 
                     parent_embeddings: List[List[float]], 
                     root: BaseNode) -> BaseNode:
        """找到最相似的父节点"""
        if not parent_embeddings:
            return root
        
        # 计算相似度
        similarities = []
        for parent_emb in parent_embeddings:
            similarity = self._cosine_similarity(embedding, parent_emb)
            similarities.append(similarity)
        
        # 找到最相似的父节点
        max_similarity = max(similarities)
        if max_similarity >= self.config.similarity_threshold:
            # 获取对应的节点
            nodes = self._get_all_nodes(root)
            return nodes[similarities.index(max_similarity)]
        
        return root
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _get_all_nodes(self, node: BaseNode) -> List[BaseNode]:
        """获取所有节点"""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def build_hierarchy(self, text: str) -> BaseNode:
        """构建层级结构"""
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return BaseNode("Empty Document")
        
        embeddings = self.embeddings.embed_documents(paragraphs)
        root = BaseNode("Root")
        
        for i, (para, emb) in enumerate(zip(paragraphs, embeddings)):
            if i == 0:
                root.add_child(BaseNode(para))
            else:
                parent = self._find_parent(emb, embeddings[:i], root)
                parent.add_child(BaseNode(para))
        
        return root
    
    def validate_hierarchy(self, node: BaseNode) -> bool:
        """验证层级结构"""
        if node.level > self.config.level_limit:
            return False
        
        for child in node.children:
            if not self.validate_hierarchy(child):
                return False
        
        return True

class IndentationHierarchyStrategy(HierarchyStrategy):
    """基于缩进的层级策略"""
    def build_hierarchy(self, text: str) -> BaseNode:
        lines = text.split('\n')
        root = BaseNode("Root")
        current_path = [root]
        
        for line in lines:
            if not line.strip():
                continue
                
            indent_level = len(line) - len(line.lstrip())
            level = indent_level // self.config.indent_size
            
            # 确保层级不超过限制
            level = min(level, self.config.level_limit)
            
            while len(current_path) > level + 1:
                current_path.pop()
            
            node = BaseNode(line.strip())
            current_path[-1].add_child(node)
            current_path.append(node)
        
        return root
    
    def validate_hierarchy(self, node: BaseNode) -> bool:
        """验证层级结构"""
        if node.level > self.config.level_limit:
            return False
        
        for child in node.children:
            if not self.validate_hierarchy(child):
                return False
        
        return True

class HierarchyBuilder:
    """层级构建器"""
    def __init__(self, strategy: HierarchyStrategy):
        self.strategy = strategy
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build(self, text: str) -> BaseNode:
        """构建层级结构"""
        try:
            hierarchy = self.strategy.build_hierarchy(text)
            if self.strategy.validate_hierarchy(hierarchy):
                return hierarchy
            else:
                self.logger.warning("层级结构验证失败，使用备用策略")
                return self._fallback_strategy(text)
        except Exception as e:
            self.logger.error(f"构建层级结构失败: {e}")
            return self._fallback_strategy(text)
    
    def _fallback_strategy(self, text: str) -> BaseNode:
        """备用策略"""
        root = BaseNode("Root")
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                root.add_child(BaseNode(para.strip()))
        return root
    
    def to_documents(self, node: BaseNode) -> List[Document]:
        """转换为文档列表"""
        documents = []
        
        def collect_documents(n: BaseNode):
            if n.content != "Root":  # 跳过根节点
                documents.append(n.to_document())
            for child in n.children:
                collect_documents(child)
        
        collect_documents(node)
        return documents

class HierarchyStrategyFactory:
    """策略工厂类"""
    @staticmethod
    def create_strategy(strategy_type: str, config: NodeConfig) -> HierarchyStrategy:
        strategies = {
            'semantic': SemanticHierarchyStrategy,
            'indentation': IndentationHierarchyStrategy,
        }
        
        if strategy_type not in strategies:
            raise ValueError(f"不支持的策略类型: {strategy_type}")
        
        return strategies[strategy_type](config)

def process_document(text: str, 
                    strategy_type: str = 'semantic',
                    config: Optional[NodeConfig] = None) -> List[Document]:
    """处理文档的主函数"""
    # 默认配置
    if config is None:
        config = NodeConfig()
    
    # 创建策略
    strategy = HierarchyStrategyFactory.create_strategy(strategy_type, config)
    
    # 创建构建器
    builder = HierarchyBuilder(strategy)
    
    # 构建层级结构
    hierarchy = builder.build(text)
    
    # 转换为文档列表
    return builder.to_documents(hierarchy)

# 使用示例
if __name__ == "__main__":
    # 示例文本
    sample_text = """
第一章 介绍
    1.1 背景
        这是一个示例文档。
        用于测试层级结构。
    
    1.2 目标
        展示父子级关系。
        说明实现方式。
    """
    
    # 处理文档
    documents = process_document(sample_text, strategy_type='indentation')
    
    # 打印结果
    for doc in documents:
        print(f"Level: {doc.metadata['level']}")
        print(f"Content: {doc.page_content}")
        print(f"Parent ID: {doc.metadata['parent_id']}")
        print("---") 
