import chromadb
from chromadb import Settings
from langchain_chroma import Chroma
from typing import Optional, List, Any
from langchain_core.documents import Document


class ChromaDB:
    """ChromaDB 封装类，支持本地和HTTP两种连接方式"""

    def __init__(
            self,
            chromadb_server_type: str = 'local',
            port: int = 8000,
            host: str = 'localhost',
            persist_path: str = 'chroma_db',
            collection_name: str = 'langchain',
            embed: Optional[Any] = None
    ):
        """
        初始化ChromaDB存储

        参数:
            chromadb_server_type: 服务器类型，'local'或'http'
            port: HTTP服务器端口(仅当server_type='http'时使用)
            host: HTTP服务器主机(仅当server_type='http'时使用)
            persist_path: 本地持久化路径(仅当server_type='local'时使用)
            collection_name: 集合名称
            embed: 嵌入函数
        """
        self.embed = embed
        self.store = None

        if chromadb_server_type == 'local':
            self._init_local_store(persist_path, collection_name)
        elif chromadb_server_type == 'http':
            self._init_http_store(host, port, collection_name)
        else:
            raise ValueError(f"不支持的服务器类型: {chromadb_server_type}")

        if self.store is None:
            raise ValueError('Chroma存储初始化失败!')

    def _init_local_store(self, persist_path: str, collection_name: str):
        """初始化本地存储"""
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embed,
            persist_directory=persist_path,
        )

    def _init_http_store(self, host: str, port: int, collection_name: str):
        """初始化HTTP连接存储"""
        client = chromadb.HttpClient(host, port)
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embed,
            client=client,
        )

    def add_with_langchain(self, docs: List[Document]) -> None:
        """
        使用Langchain添加文档

        参数:
            docs: 要添加的文档列表
        """
        if not docs:
            raise ValueError("文档列表不能为空")
        self.store.add_documents(docs)

    def get_store(self) -> Chroma:
        """获取存储实例"""
        return self.store