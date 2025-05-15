# 测试导入PDF到向量库主流程
# import chromadb
#
# # 连接到持久化数据库
# client = chromadb.PersistentClient()
#
# # 列出所有集合并逐个删除
# collections = client.list_collections()
# for collection in collections:
#     client.delete_collection(collection.name)
# print("所有集合已删除！")

def test_import():
    from rag.pdf_processor import PDFProcessor
    from utils.util import get_qwen_models

    llm , chat , embed = get_qwen_models()
    # embed = get_huggingface_embeddings()

    directory = "../dataset/dataset/pdf"
    persist_path = "chroma_db"
    server_type = "local"

    # 创建 PDFProcessor 实例
    pdf_processor = PDFProcessor(directory=directory,
                                 chroma_server_type=server_type,
                                 persist_path=persist_path,
                                 embed=embed)

    # 处理 PDF 文件
    pdf_processor.process_pdfs()
# 测试RAG主流程
def test_rag():
    from rag.rag import RAGManager
    from utils.util import get_qwen_models

    llm, chat, embed = get_qwen_models()
    rag = RAGManager(host="localhost", port=8000, llm=llm, embed=embed)

    result = rag.get_result("湖南长远锂科股份有限公司变更设立时作为发起人的法人有哪些？")

    print(result)

def test_financebot_ex():
    from agent.bidding_tendering_bot import BiddingBotEx
    from utils.util import get_qwen_models
    llm, chat, embed = get_qwen_models()
    financebot = BiddingBotEx(llm=llm, chat=chat, embed=embed)

    example_query = "20210304日，一级行业为非银金融的股票的成交量合计是多少？取整。"

    financebot.query_bidding(example_query)

if __name__ == "__main__":
    # test_rag()
    # test_import()
    test_financebot_ex()
