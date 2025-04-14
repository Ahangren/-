import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sqlalchemy import True_

# 首次应用nltk需要：
nltk.download(['stopwords','wordnet'])

# 1数据加载与检查
def load_data(filepath):
    df=pd.read_csv(filepath)
    assert 'text' in df.columns, "数据必须包含text列"
    assert 'label' in df.columns, "数据必须包含label列"
    return df

df=load_data('./data/news.csv')
print(df.shape)
print(df.head())

# 文本预处理
class TextPreprocessor:
    "自定义文本预测理类"
    def __init__(self):
        self.stop_words=set(stopwords.words('english'))
        self.lemmatizer =WordNetLemmatizer()

    def clean_text(self,text):
        "文本清理和标准化"
        text=text.lower() # 转小写
        text=re.sub(r'[^\w\s]', '',text) # 移除标点
        text=re.sub(r'\d+','',text)  # 移除数字
        words=text.split()

        words=[self.lemmatizer.lemmatize(w) for w in words
               if w not in self.stop_words and len(w)>2]
        return ' '.join(words)

preprocessor=TextPreprocessor()
df['cleaned_text']=df['text'].apply(preprocessor.clean_text)

x_train,x_text,y_train,y_test=train_test_split(df['cleaned_text'],
                                               df['label'],
                                               test_size=0.2,
                                               random_state=39,
                                               stratify=df['label'])

# 构建处理管道
model=Pipeline([
    ('tfidf',TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=0.001,
        ngram_range=(1,2),
        sublinear_tf=True
    )),
    ('classifier',PassiveAggressiveClassifier(
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=39
    ))
])

model.fit(x_train,y_train)
y_pred=model.predict(x_text)

print('\n评估结果：')
print(f"准确率：{accuracy_score(y_test,y_pred):.2%}")
print("\n分类报告")
print(classification_report(y_test,y_pred,target_names=['FAKE','REAL']))

print("\n混淆矩阵")
print(confusion_matrix(y_test,y_pred,labels=['FAKE','REAL']))

# 模型保存
import joblib
joblib.dump(model,'news_classifier.pkl')




