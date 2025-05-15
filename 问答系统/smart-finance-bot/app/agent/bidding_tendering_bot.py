import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent

from rag.rag import RAGManager
from utils.util import get_qwen_models
import settings as settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bidding_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BiddingBotEx:
    """招投标专业问答助手，集成招标文件解析和招投标数据库查询能力"""

    def __init__(
            self,
            llm=None,
            chat=None,
            embed=None,
            sql_db_uri: str = settings.SQLDATABASE_URI,
            rag_config: Optional[Dict] = None
    ):
        """
        初始化招投标助手

        参数:
            llm: 语言模型实例
            chat: 聊天模型实例
            embed: 嵌入模型实例
            sql_db_uri: 招投标数据库连接URI
            rag_config: RAG配置字典
        """
        # 初始化模型
        self.llm, self.chat, self.embed = get_qwen_models() if None in (llm, chat, embed) else (llm, chat, embed)

        # 初始化组件
        self.sql_db_uri = sql_db_uri
        self.rag_config = rag_config or {}
        self.agent_executor = None

        # 初始化核心组件
        self._initialize_components()
        logger.info("BiddingBotEx 招投标助手初始化完成")

    def _initialize_components(self):
        """初始化核心组件"""
        try:
            # 初始化招投标文档RAG
            self.rag = RAGManager(
                llm=self.llm,
                embed=self.embed,
                ** self.rag_config
            )

            # 初始化招投标数据库工具
            self.sql_tools = self._init_sql_tools()

            # 初始化招投标文档检索工具
            self.rag_tool = self._init_rag_tool()

            # 创建招投标专用Agent
            self.agent_executor = self._create_agent()

        except Exception as e:
            logger.error(f"初始化组件失败: {str(e)}")
            raise

    def _init_sql_tools(self) -> List:
        """初始化招投标SQL工具集"""
        try:
            logger.info(f"正在连接招投标数据库: {self.sql_db_uri}")
            db = SQLDatabase.from_uri(f'sqlite:///{self.sql_db_uri}')
            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            return toolkit.get_tools()
        except Exception as e:
            logger.error(f"初始化招投标数据库工具失败: {str(e)}")
            return []

    def _init_rag_tool(self):
        """初始化招投标文档检索工具"""
        try:
            retriever = self.rag.get_retriever()
            return create_retriever_tool(
                retriever=retriever,
                name="bidding_doc_search",
                description=(
                    "用于搜索招标文件、投标文件、技术规范等文档内容。"
                    "适用于招标公告解读、投标资格要求查询、评分标准分析等问题。"
                    "输入应为完整的招投标相关问题描述。"
                )
            )
        except Exception as e:
            logger.error(f"初始化招投标文档检索工具失败: {str(e)}")
            return None

    def _create_agent(self):
        """创建招投标专用Agent"""
        system_prompt = self._create_system_prompt()

        if not self.rag_tool or not self.sql_tools:
            logger.warning("部分工具初始化失败，Agent功能可能受限")

        tools = []
        if self.rag_tool:
            tools.append(self.rag_tool)
        if self.sql_tools:
            tools.extend(self.sql_tools)

        if not tools:
            raise ValueError("没有可用的工具，无法创建Agent")

        return create_react_agent(
            self.chat,
            tools=tools,
            state_modifier=system_prompt
        )

    def _create_system_prompt(self) -> str:
        """创建招投标专用系统提示模板"""
        return f"""
        当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        角色: 您是专业的招投标助手，擅长处理招标文件解析、投标问题解答和招投标数据分析。

        # 工具使用指南
        ## 招投标文档检索工具(bidding_doc_search)
        1. 适用于招标公告、投标文件、技术规范等文档内容检索
        2. 必须使用完整的招投标问题作为输入
        3. 回答应包含具体条款和依据

        ## 招投标数据库工具
        {self._get_sql_guidelines()}

        # 响应格式
        问题: <用户问题>
        思考: <分析步骤>
        行动: <工具名称>
        行动输入: <工具输入>
        观察: <工具输出>
        ...(可重复多次)
        最终答案: <最终回答>

        现在开始!
        """

    def _get_sql_guidelines(self) -> str:
        """生成招投标SQL工具使用指南"""
        return """
        ## 招投标数据库查询规则
        1. 先查看招标项目表(projects)、投标人表(bidders)等表结构
        2. 查询历史中标数据时限制为最近3年数据
        3. 金额类查询需注明币种(人民币/美元等)
        4. 资质要求查询需注明发标日期
        5. 只查询必要字段，避免SELECT *
        6. 禁止执行任何数据修改操作
        7. 字段含特殊字符时用双引号包裹
        8. 错误时重试最多3次
        """

    def query_bidding(self, question: str) -> str:
        """
        处理招投标相关问题
        参数:
            question: 招投标相关问题
        返回:
            专业回答内容
        """
        if not question or not isinstance(question, str):
            logger.warning(f"无效查询: {question}")
            return "请输入有效的招投标问题"

        try:
            logger.info(f"开始处理招投标问题: {question}")
            events = self.agent_executor.stream(
                {"messages": [("user", question)]},
                stream_mode="values",
            )

            result = ""
            for event in events:
                if "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    last_msg.pretty_print()
                    result = last_msg.content

            return result

        except Exception as e:
            logger.error(f"处理招投标问题失败: {question}, 错误: {str(e)}")
            return "处理招投标问题时发生错误，请稍后再试"

    def refresh_tools(self):
        """刷新招投标工具状态"""
        logger.info("正在刷新招投标工具...")
        self._initialize_components()