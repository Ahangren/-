import logging
from typing import Optional, Callable, List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from .chroma_conn import ChromaDB


class RAGManager:
    """
    RAG(检索增强生成)系统管理器

    功能:
    - 管理向量数据库连接
    - 构建RAG处理流程
    - 执行检索和生成操作

    特性:
    - 支持可配置的相似度阈值
    - 详细的日志记录
    - 类型安全的接口
    - 灵活的提示模板配置
    """

    def __init__(
            self,
            chroma_server_type: str = "http",
            host: str = "localhost",
            port: int = 8000,
            persist_path: str = "chroma_db",
            llm: Optional[BaseLanguageModel] = None,
            embed: Optional[Embeddings] = None,
            prompt_template: Optional[str] = None
    ):
        """
        初始化RAG系统

        参数:
            chroma_server_type: ChromaDB服务类型("http"或"local")
            host: ChromaDB服务器地址
            port: ChromaDB服务端口
            persist_path: 向量数据持久化存储目录
            llm: 语言模型实例
            embed: 文本嵌入模型
            prompt_template: 自定义提示模板
        """
        self.llm = llm
        self.embed = embed
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # 设置默认提示模板
        self.default_prompt_template = """你是一个问答助手。请使用以下检索到的上下文来回答问题。
            如果不知道答案，就直接说不知道。
            回答最多三句话，保持简洁明了。
            问题: {question} 
            上下文: {context} 
            回答:"""
        self.prompt_template = prompt_template or self.default_prompt_template

        # 初始化向量存储
        try:
            chrom_db = ChromaDB(
                chromadb_server_type=chroma_server_type,
                host=host,
                port=port,
                persist_path=persist_path,
                embed=embed
            )
            self.store = chrom_db.get_store()
            self.logger.info("ChromaDB连接初始化成功")
        except Exception as e:
            self.logger.error(f"初始化ChromaDB失败: {str(e)}")
            raise

    def _setup_logging(self) -> None:
        """配置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def get_chain(self, retriever: BaseRetriever) -> Callable:
        """
        构建完整的RAG处理链

        参数:
            retriever: 文档检索器实例

        返回:
            配置好的RAG处理链

        异常:
            ValueError: 如果语言模型未初始化
        """
        if not self.llm:
            error_msg = "语言模型未初始化"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("human", self.prompt_template)
        ])

        # 构建处理链
        rag_chain = (
                {
                    "context": retriever | RunnableLambda(self.format_docs),
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

        self.logger.info("RAG处理链构建完成")
        return rag_chain

    def format_docs(self, docs: List[Document]) -> str:
        """
        格式化检索到的文档

        参数:
            docs: 检索到的文档列表

        返回:
            拼接后的文档内容字符串
        """
        if not docs:
            self.logger.warning("未检索到任何相关文档")
            return ""

        # 记录检索信息
        retrieved_files = "\n".join([doc.metadata.get("source", "未知文件") for doc in docs])
        self.logger.info(f"检索到 {len(docs)} 个相关文档:\n{retrieved_files}")

        # 拼接文档内容
        return "\n\n".join(doc.page_content for doc in docs)

    def get_retriever(
            self,
            k: int = 4,
            score_threshold: float = 0.3
    ) -> BaseRetriever:
        """
        获取带阈值的文档检索器

        参数:
            k: 返回的文档数量
            score_threshold: 相似度阈值(0-1)

        返回:
            配置好的检索器实例
        """
        retriever = self.store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold
            }
        )
        self.logger.info(f"创建检索器: k={k}, score_threshold={score_threshold}")
        return retriever

    def get_result(
            self,
            question: str,
            k: int = 4,
            score_threshold: float = 0.3
    ) -> str:
        """
        执行RAG查询并返回结果

        参数:
            question: 用户问题
            k: 返回的文档数量
            score_threshold: 相似度阈值

        返回:
            语言模型生成的回答

        异常:
            RuntimeError: 如果处理过程中出现错误
        """
        try:
            self.logger.info(f"开始处理问题: {question}")

            retriever = self.get_retriever(k, score_threshold)
            rag_chain = self.get_chain(retriever)

            result = rag_chain.invoke(question)
            self.logger.info("问题处理完成")

            return result
        except Exception as e:
            self.logger.error(f"处理问题时出错: {str(e)}")
            raise RuntimeError(f"无法处理问题: {str(e)}")