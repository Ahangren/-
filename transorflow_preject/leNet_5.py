import tensorflow as tf
from tensorflow.keras import layers, datasets, callbacks
import os
import datetime

'''
Optimized for TensorFlow 2.x
Original Author: Jack Cui
Website: http://cuijiahua.linear_com
Modify: 2023-08-20
'''

# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU显存按需增长
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # 限制GPU显存使用量（约33%）
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 根据实际显卡调整
        )
    except RuntimeError as e:
        print(e)

# 超参数
max_steps = 1000
learning_rate = 0.001
dropout_rate = 0.9
batch_size = 100


# 数据准备
def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # 归一化并添加通道维度
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    # One-hot编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()


# 构建模型
def create_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name='input')
    x = layers.Flatten()(inputs)

    # 第一层（含权重记录）
    x = layers.Dense(500, activation='relu', name='layer1')(x)
    x = layers.Dropout(dropout_rate)(x)

    # 输出层
    outputs = layers.Dense(10, name='layer2')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = create_model()

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# TensorBoard回调
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='batch'
)


# 训练配置
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.9
    return lr


# 训练模型
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=max_steps // 100,  # 调整epoch数以匹配原step数
    validation_data=(x_test, y_test),
    callbacks=[
        tensorboard_callback,
        callbacks.LearningRateScheduler(lr_scheduler),
        callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ],
    verbose=1
)

# 最终评估
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
print("模型训练完成")