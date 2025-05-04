import json
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
import os
from llama_index.llms.ollama import Ollama
from llama_index.finetuning.embeddings.common import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

# 修复路径（Windows）
BASE_DIR = r"./ragdata/"
TRAIN_FILES = [BASE_DIR + "中华人民共和国证券法(2019修订).pdf"]
VAL_FILES = [BASE_DIR + "中华人民共和国证券法(2019修订).pdf"]
TRAIN_CORPUS_FPATH = BASE_DIR + "train_corpus.json"
VAL_CORPUS_FPATH = BASE_DIR + "val_corpus.json"

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")
    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)
    if verbose:
        print(f"Parsed {len(nodes)} nodes")
    return nodes

def mk_dataset():
    # 确保 Ollama 模型已下载
    try:
        ollm = Ollama(model="qwen2:7b-instruct-q4_0", request_timeout=120.0)
    except Exception as e:
        print(f"Ollama 模型加载失败: {e}")
        print("请先运行: ollama pull qwen2:7b-instruct-q4_0")
        return

    # 加载数据
    train_nodes = load_corpus(TRAIN_FILES, verbose=True)
    val_nodes = load_corpus(VAL_FILES, verbose=True)

    # 生成 QA 数据集
    train_dataset = generate_qa_embedding_pairs(llm=ollm, nodes=train_nodes)
    val_dataset = generate_qa_embedding_pairs(llm=ollm, nodes=val_nodes)

    # 保存数据集
    train_dataset.save_json(TRAIN_CORPUS_FPATH)
    val_dataset.save_json(VAL_CORPUS_FPATH)
    print("数据集生成完成！")

from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset


def finetune_embedding_model():
    # 加载训练集和验证集
    train_dataset = EmbeddingQAFinetuneDataset.from_json(TRAIN_CORPUS_FPATH)
    val_dataset = EmbeddingQAFinetuneDataset.from_json(VAL_CORPUS_FPATH)
    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,  # 训练集
        model_id="BAAI/bge-small-zh-v1.5",  # 底模
        model_output_path="zhengquan",
        val_dataset=val_dataset,  # 验证集
    )
    finetune_engine.finetune()  # 直接微调
    embed_model = finetune_engine.get_finetuned_model()
    print(embed_model)

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm  # 显示进度条的工具
import pandas as pd  # 数据处理库
def evaluate(dataset, embed_model, top_k=5, verbose=False):
    # 获取数据集中的三个主要部分：
    corpus = dataset.corpus  # 所有法律条文内容（字典格式，id:条文内容）
    queries = dataset.queries  # 测试问题（字典格式，id:问题内容）
    relevant_docs = dataset.relevant_docs  # 标准答案（字典格式，问题id:[正确答案id]）
    # 将法律条文转换成TextNode对象列表
    # 每个TextNode包含id和文本内容，就像给每条法律发身份证
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]

    # 创建向量索引（相当于建立智能法律条文检索系统）
    # 使用embed_model将文本转换为向量（数学表示）
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)

    # 创建检索器，设置返回最相似的top_k个结果
    retriever = index.as_retriever(similarity_top_k=top_k)
    eval_results = []  # 存储评估结果的列表

    # 遍历所有测试问题（显示进度条）
    for query_id, query in tqdm(queries.items()):
        # 用当前问题检索最相关的法律条文
        retrieved_nodes = retriever.retrieve(query)

        # 获取检索结果的id列表
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]

        # 获取当前问题的标准答案id
        expected_id = relevant_docs[query_id][0]

        # 检查标准答案是否在检索结果中
        is_hit = expected_id in retrieved_ids

        # 记录评估结果
        eval_result = {
            "is_hit": is_hit,  # 是否命中
            "retrieved": retrieved_ids,  # 模型返回的结果
            "expected": expected_id,  # 标准答案
            "query": query_id,  # 问题id
        }
        eval_results.append(eval_result)
        df = pd.DataFrame(eval_results)
        hit_rate = df['is_hit'].mean()  # 计算命中率
        print(f"Top-{top_k}命中率：{hit_rate:.2%}")


if __name__ == "__main__":
    mk_dataset()
    finetune_embedding_model()

    # evaluate()