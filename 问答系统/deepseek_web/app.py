from flask import Flask, render_template, request, Response
import ollama
import logging
from datetime import datetime

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