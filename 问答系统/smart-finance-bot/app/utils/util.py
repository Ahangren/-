
from dotenv import load_dotenv
import os

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)

# 构建到 conf/.qwen 的相对路径
conf_file_path_qwen = os.path.join(current_dir, '..', 'conf', '.qwen')

# 加载千问环境变量
load_dotenv(dotenv_path=conf_file_path_qwen)

def get_qwen_models():
    """
    加载千问系列大模型
    """
    # llm 大模型
    from langchain_community.llms.tongyi import Tongyi

    llm = Tongyi(model="qwen-max", temperature=0.1, top_p=0.7, max_tokens=1024)

    # chat 大模型
    from langchain_community.chat_models import ChatTongyi

    chat = ChatTongyi(model="qwen-max", temperature=0.01, top_p=0.2, max_tokens=1024)
    # embedding 大模型
    from langchain_community.embeddings import DashScopeEmbeddings

    embed = DashScopeEmbeddings(model="text-embedding-v3")

    return llm, chat, embed