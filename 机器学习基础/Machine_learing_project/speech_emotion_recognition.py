# 调试代码：

# #1.导入需要的库
# import librosa # 音频处理库
# import soundfile # 音频文件读写
# import os   # 操作系统接口
# import glob  # 文件路径匹配
# import pickle  # 对象序列化
# import numpy as np
# from sklearn.model_selection import train_test_split,cross_val_score  # 数据划分和交叉验证
# from sklearn.neural_network import MLPClassifier  # 多层感知机分类器
# from sklearn.metrics import accuracy_score,classification_report  # 评估指标
# from sklearn.preprocessing import LabelEncoder  # 标签编码
# from joblib import Parallel,delayed  # 并行计算
# import time # 时间统计
# import warnings  # 警告处理
#
#
# # 忽略librosa的非关键警告
# warnings.filterwarnings('ignore',category=UserWarning)
# # 情感标签映射字典
# EMOTIONS = {
#     '01': "neutral",  # 中性
#     '02': "calm",     # 平静
#     '03': "happy",    # 高兴
#     '04': "sad",      # 悲伤
#     '05': 'angry',    # 愤怒
#     '06': "fearful",  # 恐惧
#     '07': 'disgust',  # 厌恶
#     '08': 'surprised' # 惊讶
# }
# # 要分析的情感类别
# OBSERVED_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
# # 音频数据路径（使用原始字符串避免转义）
#
# DATA_PATH = r".\data\Actor_*\*.wav"
# # 模型保存路径
# MODEL_SAVE_PATH='emotion_classifier.pkl'
#
#
# def extract_feature(file_path, mfcc=True, chroma=True, mel=True):
#     try:
#         # 使用librosa读取音频，兼容性更好
#         X, sample_rate = librosa.load(file_path, sr=None)
#
#         # 验证音频数据
#         if len(X) < 1024:
#             print("Audio too short")
#             return None
#
#         result = np.array([])
#
#         # 计算STFT（用于chroma）
#         if chroma:
#             stft = np.abs(librosa.stft(X, n_fft=2048, hop_length=512))
#
#         # 提取MFCC特征
#         if mfcc:
#             mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
#             mfccs = np.mean(mfccs.T, axis=0)
#             result = np.hstack((result, mfccs))
#
#         # 提取Chroma特征
#         if chroma:
#             chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
#             chroma = np.mean(chroma.T, axis=0)
#             result = np.hstack((result, chroma))
#
#         # 提取Mel频谱
#         if mel:
#             mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
#             mel = np.mean(mel.T, axis=0)
#             result = np.hstack((result, mel))
#
#         return result
#
#     except Exception as e:
#         print(f"Feature extraction error: {str(e)}")
#         return None
#
# def process_file(file_path):
#     try:
#         file_name = os.path.basename(file_path)
#         print(f"\nProcessing: {file_name}")
#
#         # 分割文件名并提取情感代码
#         parts = file_name.split('-')
#         if len(parts) < 3:
#             print("Invalid filename format")
#             return None
#
#         emotion_code = parts[2]  # 第三部分是情感代码
#         print(f"Emotion code: {emotion_code}")
#
#         # 确保emotion_code在映射表中
#         if emotion_code not in EMOTIONS:
#             print(f"Unknown emotion code: {emotion_code}")
#             return None
#
#         emotion = EMOTIONS[emotion_code]  # 直接获取，不再用get()避免None
#         print(f"Mapped emotion: {emotion}")
#
#         # 提取特征
#         feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
#         if feature is None:
#             print("Feature extraction failed")
#             return None
#
#         print("Successfully processed")
#         return (feature, emotion)
#
#     except Exception as e:
#         print(f"Error processing {file_name}: {str(e)}")
#         return None
#
#
# def load_data(test_size=0.2):
#     print("\nStarting data loading...")
#
#     # 获取文件列表
#     matched_files = glob.glob(DATA_PATH)
#     print(f"Found {len(matched_files)} files")
#
#     if not matched_files:
#         raise ValueError("No files found at specified path")
#
#     # 处理文件
#     features_and_labels = []
#     for file_path in matched_files[:100]:  # 先处理前100个测试
#         result = process_file(file_path)
#         if result is not None:
#             features_and_labels.append(result)
#
#     if not features_and_labels:
#         raise ValueError("""
#         No valid samples processed. Please check:
#         1. Audio files are valid (try playing them manually)
#         2. Filename format matches expected pattern
#         3. No errors in feature extraction
#         """)
#
#     # 准备数据
#     X = [fl[0] for fl in features_and_labels]
#     y = [fl[1] for fl in features_and_labels]
#
#     print(f"\nSuccessfully processed {len(X)} samples")
#     print(f"Feature vector length: {X[0].shape[0]}")
#     print("Sample labels:", y[:5])
#
#     # 编码标签
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)
#
#     return train_test_split(np.array(X), y_encoded, test_size=test_size, random_state=39), le
#
# def train_model(x_train,x_test,y_train,y_test):
#     """
#     训练和模型评估
#     :param x_train: 训练集特征
#     :param x_test: 测试集特征
#     :param y_train: 训练集标签
#     :param y_test: 测试集标签
#     :return: MLPClassifier：训练好的模型
#     """
#     print(" Training model ... ")
#     # 定义MLP分类器
#     model=MLPClassifier(
#         alpha=0.01,   # L2正则化
#         batch_size=256, # 批大小
#         epsilon=1e-08,  # 防止分母为零
#         hidden_layer_sizes=(300,), # 隐藏层大小
#         learning_rate='adaptive', # 自适应学习率
#         max_iter=500,  # 最大迭代次数
#         early_stopping=True,  # 启用早停
#         verbose=True  # 打印训练结果
#     )
#     # 模型训练
#     model.fit(x_train,y_train)
#     # 预测
#     y_pred=model.predict(x_test)
#     # 计算准确率
#     accuracy=accuracy_score(y_test,y_pred)
#     print("\nModel Evaluation:")
#     print(f"Accuracy: {accuracy:.2%}")
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred, target_names=OBSERVED_EMOTIONS))
#
#     # 进行5折交叉验证
#     cv_scores=cross_val_score(model,np.vstack((x_train,x_test)),np.concatenate((y_train,y_test)),cv=5)
#     print(f"\nCross-validation scores: {cv_scores}")
#     print(f"Mean CV accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
#
#     return model
#
# def save_model(model,label_encoder,file_path=MODEL_SAVE_PATH):
#     """
#     保存模型和标签编码器
#     :param model: 训练好的模型
#     :param label_encoder: 标签编码器
#     :param file_path: 保存路径
#     :return:
#     """
#     with open(file_path,'wb') as f:
#         # 使用pickle序列化模型和编码器
#         pickle.dump({'model':model,'label_encoder':label_encoder},f)
#     print(f"Model save to {file_path}")
#
# def load_saved_model(file_path=MODEL_SAVE_PATH):
#     """
#     加载保存的模型
#     :param file_path: 模型文件路径
#     :return: 元组：（模型，标签编码器）
#     """
#     with open(file_path,'rb') as f:
#         # 加载pickle文件
#         data=pickle.load(f)
#     return data['model'],data['label_encoder']
#
#
# if __name__ == "__main__":
#     # 测试单个文件
#     test_file = glob.glob(DATA_PATH)[0] if glob.glob(DATA_PATH) else None
#     if test_file:
#         print("\nTesting single file:", test_file)
#         print("Filename parts:", os.path.basename(test_file).split('-'))
#         print("Emotion mapping:", EMOTIONS)
#
#         # 测试特征提取
#         feature = extract_feature(test_file)
#         print("Extracted feature:", feature.shape if feature is not None else None)
#
#     # 加载数据
#     try:
#         (X_train, X_test, y_train, y_test), le = load_data(test_size=0.25)
#         print("\nData loading successful!")
#         print(f"Training samples: {len(X_train)}")
#         print(f"Test samples: {len(X_test)}")
#
#         # 训练模型...
#
#     except Exception as e:
#         print("\nError:", str(e))
#         print("\nDebugging tips:")
#         print("1. Verify DATA_PATH points to correct location")
#         print("2. Check audio files can be played manually")
#         print("3. Test extract_feature() on single file")
#         print("4. Check filename format matches EMOTIONS mapping")

# 完整代码
import librosa
import soundfile
import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 情感标签映射
EMOTIONS = {
    '01': "neutral",
    '02': "calm",
    '03': "happy",
    '04': "sad",
    '05': 'angry',
    '06': "fearful",
    '07': 'disgust',
    '08': 'surprised'
}

OBSERVED_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
DATA_PATH = r".\data\Actor_*\*.wav"
MODEL_SAVE_PATH = 'emotion_classifier.pkl'


def extract_feature(file_path, mfcc=True, chroma=True, mel=True):
    """提取音频特征"""
    try:
        X, sample_rate = librosa.load(file_path, sr=None)
        if len(X) < 1024:
            return None

        result = np.array([])

        if chroma:
            stft = np.abs(librosa.stft(X, n_fft=2048, hop_length=512))

        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            mfccs = np.mean(mfccs.T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            chroma = np.mean(chroma.T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            mel = np.mean(mel.T, axis=0)
            result = np.hstack((result, mel))

        return result
    except:
        return None


def process_file(file_path):
    """处理单个音频文件"""
    try:
        file_name = os.path.basename(file_path)
        parts = file_name.split('-')
        if len(parts) < 3:
            return None

        emotion_code = parts[2]
        if emotion_code not in EMOTIONS:
            return None

        emotion = EMOTIONS[emotion_code]
        if emotion not in OBSERVED_EMOTIONS:
            return None

        feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
        if feature is not None:
            return (feature, emotion)
        return None
    except:
        return None


def load_data(test_size=0.2):
    """加载并预处理数据"""
    matched_files = glob.glob(DATA_PATH)
    features_and_labels = []

    for file_path in matched_files:
        result = process_file(file_path)
        if result is not None:
            features_and_labels.append(result)

    X = [fl[0] for fl in features_and_labels]
    y = [fl[1] for fl in features_and_labels]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return train_test_split(np.array(X), y_encoded, test_size=test_size, random_state=39), le


def train_model(x_train, x_test, y_train, y_test):
    """训练和评估模型"""
    model = MLPClassifier(
        alpha=0.01,
        batch_size=256,
        epsilon=1e-08,
        hidden_layer_sizes=(300,),
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Evaluation:")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=OBSERVED_EMOTIONS))

    return model


def save_model(model, label_encoder, file_path=MODEL_SAVE_PATH):
    """保存模型"""
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': label_encoder}, f)


def load_saved_model(file_path=MODEL_SAVE_PATH):
    """加载模型"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['label_encoder']


if __name__ == "__main__":
    # 数据加载
    (X_train, X_test, y_train, y_test), le = load_data(test_size=0.25)

    # 模型训练
    model = train_model(X_train, X_test, y_train, y_test)

    # 模型保存
    save_model(model, le)
    print(f"Model saved to {MODEL_SAVE_PATH}")