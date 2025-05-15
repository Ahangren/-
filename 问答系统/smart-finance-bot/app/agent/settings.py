import os.path

SQLDATABASE_URI=os.path.join(os.getcwd(),r'dataset\dataset\博金杯比赛数据.db')

CHROMA_PORT=8000
CHROMA_HOST='local'
PERSIST_PATH='chroma_db'
COLLECTION_NAME='langchain'