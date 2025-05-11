import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent  # 假设pdf_processor.py在rag/下
print(project_root)
sys.path.append(str(project_root))

def test_import():
    from rag.pdf_processor import PDFProcessor
    from utils.util import get_qwen_models
    llm,chat,embed=get_qwen_models()

    directory="./dataset/pdf"
    persist_path="chroma_db"
    server_type='local'

    pdf_processor=PDFProcessor(
        directory=directory,
        chroma_server_type=server_type,
        persist_path=persist_path,
        embed=embed
    )

    pdf_processor.process_pdfs()

if __name__ == '__main__':
    test_import()