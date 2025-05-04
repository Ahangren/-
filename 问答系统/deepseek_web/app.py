from flask import Flask, render_template, request, Response
import ollama
import logging

app = Flask(__name__)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat.log'),
        logging.StreamHandler()
    ]
)


def chat_ollama(user_message, stream):
    host = 'http://127.0.0.1:11434'
    cli = ollama.Client(host=host)
    response = cli.chat(
        model='deepseek-r1:7b',
        messages=[{
            'role': 'user', 'content': user_message}],
        stream=stream,
        # options={'temperature': 0.7}
    )
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """流式聊天接口"""

    def generate(user_message):
        try:
            app.logger.info(f"流式处理开始: {
            user_message[:50]}...")

            stream = chat_ollama(user_message, True)
            for chunk in stream:
                content = chunk['message']['content']
                if content.startswith('<think>'):
                    content = content.replace('<think>', '', 1)
                elif content.startswith('</think>'):
                    content = content.replace('</think>', '\n', 1)
                app.logger.debug(f"发送数据块: {
                content}")
                yield f"{
                content}"

            app.logger.info("流式处理完成")

        except Exception as e:
            app.logger.error(f"流式错误: {
            str(e)}")
            yield f"[ERROR] {
            str(e)}\n\n"

    return Response(generate(request.json['message']), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
# app.py
# import os, time, json
# from flask import Flask, request, jsonify, render_template, Response
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_ollama import OllamaEmbeddings, OllamaLLM
# from langchain_chroma import Chroma
#
# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#
# # 初始化模型
# # app.py 修改部分
# # 新增配置参数
# OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')  # 从环境变量读取
#
# # 修改模型初始化部分
# embeddings = OllamaEmbeddings(
#     model='mxbai-embed-large',
#     base_url=OLLAMA_HOST  # 添加base_url参数
# )
#
# llm = OllamaLLM(
#     model='deepseek-r1:7b',
#     temperature=0.3,
#     base_url=OLLAMA_HOST,  # 添加base_url参数
# )
# vector_store = None
#
#
# def init_vector_store(filepath=None):
#     global vector_store
#     # 空路径时初始化本地数据库
#     if not filepath:
#         if not vector_store:
#             if os.path.exists('chroma_db'):
#                 vector_store = Chroma(
#                     persist_directory='chroma_db',
#                     embedding_function=embeddings
#                 )
#                 print(f"成功加载本地知识库,文档数：{vector_store._collection.count()}")
#             else:
#                 raise ValueError("本地数据库不存在，请先添加文档")
#         return
#
#     # 非空路径时处理文档
#     try:
#         # 文档加载与分块
#         loader = PyPDFLoader(filepath) if filepath.endswith('.pdf') else TextLoader(filepath)
#         documents = loader.load()
#
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1024,
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_documents(documents)
#
#         # 数据库操作
#         if vector_store:
#             vector_store.add_documents(chunks)
#             print(f'文档已追加: {filepath}')
#         else:
#             vector_store = Chroma.from_documents(
#                 documents=chunks,
#                 embedding=embeddings,
#                 persist_directory='chroma_db'
#             )
#             print(f'新建知识库成功: {filepath}')
#
#         vector_store.persist()  # 确保持久化存储
#
#     except Exception as e:
#         print(f"文档处理失败: {str(e)}")
#         raise
#
#
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400
#
#     filepath = os.path.join(UPLOAD_FOLDER, f"{file.filename}")
#
#     try:
#         if not os.path.exists(filepath):
#             file.save(filepath)
#             init_vector_store(filepath)
#         return jsonify({'message': 'File processed successfully'})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
#
# def ask_question():
#     data = request.json
#     print('收到请求:', data)
#     if not data or 'question' not in data:
#         def error_stream():
#             yield json.dumps({'error': 'No question provided'}) + '\n'
#
#         return Response(error_stream(), mimetype='application/x-ndjson'), 400
#
#     try:
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
#             chain_type="stuff",
#         )
#
#         def generate_stream():
#             try:
#                 # 修改为流式调用方式
#                 response_stream = qa_chain.stream({'query': data['question']})
#                 for chunk in response_stream:
#                     content = chunk['result']
#                     # 逐块返回结果并立即刷新缓冲区
#                     print('发送应答:', content)
#                     yield content
#             except Exception as e:
#                 yield f"data: {json.dumps({'error': str(e)})}\n\n"
#
#         # 设置流式响应头
#         return Response(
#             generate_stream(),
#             mimetype='text/event-stream',
#             headers={
#                 'Cache-Control': 'no-cache',
#                 'Connection': 'keep-alive',
#                 'X-Accel-Buffering': 'no'
#             }
#         )
#     except Exception as e:
#         def error_stream():
#             yield json.dumps({'error': str(e)}) + '\n'
#
#         return Response(error_stream(), mimetype='application/x-ndjson'), 500
#
#
# if __name__ == '__main__':
#     init_vector_store()
#     app.run(host='0.0.0.0', port=5000, debug=True)
