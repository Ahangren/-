# 导入必要的库和模块
import logging  # Python标准日志库，用于记录程序运行信息
from langchain_core.prompts import ChatPromptTemplate  # 用于创建聊天式提示模板
from langchain_core.runnables import RunnablePassthrough  # 用于在链中传递数据
from langchain_core.runnables.base import RunnableLambda  # 将普通函数转换为可运行对象
from langchain_core.output_parsers import StrOutputParser  # 将输出解析为字符串
from .chroma_conn import ChromaDB  # 自定义的ChromaDB连接类

# 配置基础日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RagManager:
    """
    RAG(检索增强生成)系统管理器
    负责整合向量数据库检索和大语言模型生成能力
    """

    def __init__(self,
                 chroma_server_type="http",  # ChromaDB服务类型，默认HTTP模式
                 host="localhost",  # ChromaDB主机地址，默认本地
                 port=8000,  # ChromaDB端口号，默认8000
                 persist_path="chroma_db",  # 向量数据库持久化存储路径
                 llm=None,  # 语言模型实例，如OpenAI、ChatGLM等
                 embed=None):  # 文本嵌入模型，用于将文本转换为向量
        """
        初始化RAG系统
        参数:
            chroma_server_type: ChromaDB运行模式("http"或"local")
            host: ChromaDB服务器地址
            port: ChromaDB服务端口
            persist_path: 向量数据持久化存储目录
            llm: 预初始化的语言模型
            embed: 预初始化的嵌入模型
        """
        self.llm = llm  # 存储语言模型
        self.embed = embed  # 存储嵌入模型

        # 初始化ChromaDB连接
        chrom_db = ChromaDB(
            chroma_server_type=chroma_server_type,  # 服务类型
            host=host, port=port,  # 连接地址和端口
            persist_path=persist_path,  # 持久化路径
            embed=embed  # 嵌入模型
        )
        self.store = chrom_db.get_store()  # 获取向量存储对象

    def get_chain(self, retriever):
        """
        构建完整的RAG处理链
        参数:
            retriever: 文档检索器实例
        返回:
            配置好的RAG处理链
        """
        # 定义问答提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("human", """你是一个问答助手。请使用以下检索到的上下文来回答问题。
             如果不知道答案，就直接说不知道。
             回答最多三句话，保持简洁明了。
             问题: {question} 
             上下文: {context} 
             回答:""")
        ])

        # 将format_docs方法转换为可运行对象
        format_docs_runnable = RunnableLambda(self.format_docs)

        # 构建RAG处理链（使用LangChain的表达式语法）
        rag_chain = (
                {
                    # 左侧: 通过retriever检索文档 -> 用format_docs格式化
                    "context": retriever | format_docs_runnable,
                    # 右侧: 直接传递用户的问题
                    "question": RunnablePassthrough()
                }
                | prompt  # 将格式化后的输入传递给提示模板
                | self.llm  # 将提示输入语言模型
                | StrOutputParser()  # 将模型输出解析为字符串
        )

        return rag_chain

    def format_docs(self, docs):
        """
        格式化检索到的文档
        参数:
            docs: 检索到的文档列表
        返回:
            拼接后的文档内容字符串
        """
        # 记录检索到的文档数量
        logging.info(f"检索到资料文件个数：{len(docs)}")

        # 提取每个文档的元数据中的文件名
        retrieved_files = "\n".join([doc.metadata["source"] for doc in docs])
        logging.info(f"资料文件分别是:\n{retrieved_files}")

        # 拼接所有文档的内容，用两个换行符分隔
        retrieved_content = "\n\n".join(doc.page_content for doc in docs)
        logging.info(f"检索到的资料为:\n{retrieved_content}")

        return retrieved_content  # 返回格式化后的内容

    def get_retriever(self, k=4, mutuality=0.3):
        """
        获取带阈值的文档检索器
        参数:
            k: 返回的文档数量，默认4
            mutuality: 相似度阈值(0-1)，默认0.3
        返回:
            配置好的检索器实例
        """
        retriever = self.store.as_retriever(
            search_type="similarity_score_threshold",  # 使用带阈值的相似度搜索
            search_kwargs={
                "k": k,  # 返回文档数量
                "score_threshold": mutuality  # 相似度阈值
            })
        return retriever

    def get_result(self, question, k=4, mutuality=0.3):
        """
        执行RAG查询并返回结果
        参数:
            question: 用户问题字符串
            k: 返回的文档数量，默认4
            mutuality: 相似度阈值，默认0.3
        返回:
            语言模型生成的回答
        """
        # 获取配置好的检索器
        retriever = self.get_retriever(k, mutuality)

        # 构建完整的RAG处理链
        rag_chain = self.get_chain(retriever)

        # 执行查询并返回结果
        return rag_chain.invoke(input=question)