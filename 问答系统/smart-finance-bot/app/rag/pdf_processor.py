import logging
import os
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from rag.chroma_conn import ChromaDB


class PDFProcessor:
    def __init__(self, directory: str, chroma_server_type: str, persist_path: str, embed: callable,
                 file_group_num: int = 20,  # 减少每组处理的文件数
                 batch_num: int = 4,  # 减少每批插入的文档数
                 max_workers: int = 4,  # 并发处理线程数
                 chunk_size: int = 500,
                 chunk_overlap: int = 100):

        self.directory = directory
        self.file_group_num = file_group_num
        self.batch_num = batch_num
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 配置日志
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # 初始化ChromaDB连接
        self.chroma_db = ChromaDB(
            chromadb_server_type=chroma_server_type,
            persist_path=persist_path,
            embed=embed
        )

    def _setup_logging(self):
        """配置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("pdf_processor.log"),  # 同时记录到文件
                logging.StreamHandler()
            ]
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, ConnectionResetError))
    )
    def _safe_add_documents(self, batch):
        """带重试机制的文档插入方法"""
        try:
            self.chroma_db.add_with_langchain(batch)
            return True
        except Exception as e:
            self.logger.warning(f"文档插入失败，将重试: {str(e)}")
            raise

    def load_pdf_files(self) -> List[str]:
        """加载目录下的所有PDF文件"""
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"目录不存在: {self.directory}")

        pdf_files = [
            os.path.join(self.directory, f)
            for f in os.listdir(self.directory)
            if f.lower().endswith('.pdf')
        ]
        self.logger.info(f"发现 {len(pdf_files)} 个PDF文件")
        return pdf_files

    def load_pdf_content(self, pdf_path: str) -> Optional[List[Document]]:
        """安全地加载PDF内容"""
        try:
            loader = PyMuPDFLoader(pdf_path)
            return loader.load()
        except Exception as e:
            self.logger.error(f"加载PDF失败 {pdf_path}: {str(e)}")
            return None

    def process_pdfs(self):
        """处理所有PDF文件的主方法"""
        pdf_files = self.load_pdf_files()
        if not pdf_files:
            self.logger.warning("没有找到PDF文件")
            return

        # 分批处理PDF文件
        for i in range(0, len(pdf_files), self.file_group_num):
            group = pdf_files[i:i + self.file_group_num]
            self.logger.info(f"正在处理第 {i // self.file_group_num + 1} 组 ({len(group)} 个文件)")

            try:
                self._process_group(group)
            except Exception as e:
                self.logger.error(f"处理组失败: {str(e)}")
                # 可以选择继续处理下一组
                continue

    def _process_group(self, file_group: List[str]):
        """处理一组PDF文件"""
        # 并发加载PDF内容
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.load_pdf_content, f) for f in file_group]
            documents = []

            for future in tqdm(as_completed(futures), total=len(futures), desc="加载PDF"):
                result = future.result()
                if result:
                    documents.extend(result)

        if not documents:
            self.logger.warning("当前组没有有效的PDF内容")
            return

        # 分割文本
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        self.logger.info(f"分割得到 {len(chunks)} 个文本块")

        # 分批插入ChromaDB
        self._insert_batches(chunks)

    def _insert_batches(self, chunks: List[Document]):
        """分批插入文档到ChromaDB"""
        total_batches = (len(chunks) + self.batch_num - 1) // self.batch_num
        progress = tqdm(total=total_batches, desc="插入文档")

        for i in range(0, len(chunks), self.batch_num):
            batch = chunks[i:i + self.batch_num]

            try:
                self._safe_add_documents(batch)
                progress.update(1)
            except Exception as e:
                self.logger.error(f"插入批次 {i // self.batch_num + 1} 失败: {str(e)}")
                # 可以选择保存失败的批次以便后续重试
                continue

        progress.close()