import cv2
import easygui
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk


class CartoonifyApp:
    def __init__(self, root):
        # root:程序主页面
        self.root = root
        # 程序标题
        self.root.title('Cartoonify Your Image!')
        # 页面大小
        self.root.geometry('500x500')
        # 页面配置：背景为白色
        self.root.configure(background='white')
        #
        self.style = ttk.Style()
        self.style.configure('TButton', font=('calibri', 12, 'bold'), padding=10)
        # 创建窗口工具
        self.create_widgets()

    def create_widgets(self):
        # Main frame 创建主容器框架
        main_frame = tk.Frame(self.root, bg='white')
        # 设置框架布局：expand：允许扩展填充空间，fill：填充xy方向，padx和pady：设置边框边距
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Title label：创建标题标签
        title_label = tk.Label(main_frame,  # 放在主容器框架内
                               text="Cartoonify Your Image", # 显示文本
                               font=('calibri', 20, 'bold'),  # 字体设置
                               bg='white',  # 白色背景
                               fg='#364156')  # 文字颜色
        # 标签布局：放在距离上方0像素，下方20像素的位置
        title_label.pack(pady=(0, 20))

        # Upload button  创建上传按钮
        upload_btn = ttk.Button(main_frame, # 放在主框架中
                                text="Select Image",  # 显示文本
                                command=self.upload_image)  # 点击后的回调函数
        # 布局在距离上下边框10像素位置
        upload_btn.pack(pady=10)

        # Save button (initially hidden) 创建保存按钮，没有上传文件之前禁用
        self.save_btn = ttk.Button(main_frame,  # 放在主框架内
                                   text="Save Cartoon Image", # 显示文本
                                   command=self.save_image, # 回调函数
                                   state='disabled') # 初始不可点击
        # 布局
        self.save_btn.pack(pady=10)

        # Status label 创建标签显示状态
        self.status_label = tk.Label(main_frame,
                                     text="", # 初始无文本
                                     font=('calibri', 10), # 设置字体
                                     bg='white',  # 白色背景
                                     fg='#364156') # 字体颜色
        # 布局标签
        self.status_label.pack(pady=10)

        # Store the cartoon image for saving
        self.cartoon_image = None  # 用于存储处理后的卡通图像
        self.original_path = None  # 用于存储原始图片路径

    def upload_image(self):
        try:
            image_path = easygui.fileopenbox()
            if image_path:
                self.original_path = image_path
                self.cartoonify(image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {str(e)}")

    def cartoonify(self, image_path):
        try:
            # Read and process image 读取或者验证图像
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError("Could not read the image file")
            # 将原始图像转换成RGB形式
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Resize once  改变图片大小
            resized = cv2.resize(original_image, (1400, 1920))

            # Convert to grayscale  # 将图片转换成灰度图像
            gray_image = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

            # Apply median blur  中值滤波去噪，原理：用像素周围的平均值来代替当前值，有效保留边缘同时去除噪声
            smooth_gray = cv2.medianBlur(gray_image, 5)


            # Edge detection 自适应阈值边缘检测
            edges = cv2.adaptiveThreshold(smooth_gray, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C, # 使用局部均值作为阈值
                                          cv2.THRESH_BINARY, 9, 9) # 二值化类型，区块大小和常数c

            # Color smoothing  双边滤波颜色平滑
            color_image = cv2.bilateralFilter(resized, 9, 300, 300)
            """
            参数说明：
            d=9 - 滤波直径
            sigmaColor=300 - 颜色空间标准差(越大颜色混合越明显)
            sigmaSpace=300 - 坐标空间标准差(越大远处像素影响越大)
            特点：在平滑颜色的同时保留边缘锐度
            """
            # Combine edges with color 合成卡通效果
            cartoon_image = cv2.bitwise_and(color_image, color_image, mask=edges)
            """
            位操作原理：
            - 使用edges作为掩码，只保留颜色图像中边缘对应的区域
            - 实现轮廓强化+颜色平滑的卡通效果
            """
            # Store for saving  保存处理结果
            self.cartoon_image = cartoon_image

            # Display results
            self.display_results(resized, gray_image, smooth_gray, edges, color_image, cartoon_image)

            # Enable save button
            self.save_btn.config(state='normal')  # 将保存按钮的状态变更为可点击
            self.status_label.config(text="Image processed successfully!")  # 更新状态栏
            """
            graph TD
                A[原始图像] --> B[RGB转换]
                B --> C[尺寸标准化]
                C --> D[灰度化]
                D --> E[中值滤波]
                E --> F[边缘检测]
                C --> G[双边滤波]
                F & G --> H[效果合成]
                H --> I[结果输出]
            """

        except Exception as e:
            # 如果图片异常
            messagebox.showerror("Processing Error", f"Failed to process image: {str(e)}")
            self.status_label.config(text="Error processing image")

    def display_results(self, *images):
        # 定义子图列表标签： [0]原始RGB -> [1]灰度 -> [2]平滑灰度 -> [3]边缘 -> [4]平滑颜色 -> [5]最终卡通效果
        titles = ['Original', 'Grayscale', 'Smoothed', 'Edges', 'Color', 'Cartoon']
        # 创建画布
        plt.figure(figsize=(12, 8))
        # 东涛生成子图网格
        for i, (image, title) in enumerate(zip(images, titles)):
            # 创建2*3大小的网格，索引从1开始
            plt.subplot(2, 3, i + 1)
            # 判断图像类型，shape==2代表单通道灰度图像
            if len(image.shape) == 2:  # Grayscale
                # 显示设置灰度范围
                plt.imshow(image, cmap='gray')
            else:  # Color  三通道彩色图像
                plt.imshow(image)  # 自动归一化处理
            plt.title(title,fontsize=10,pad=5) # pad调整标题与图像的间距
            plt.axis('off')  # 关闭坐标显示

        plt.tight_layout()
        plt.show()
        """
        graph LR
        A[输入图像列表] --> B{通道数判断}
        B -->|单通道| C[灰度显示]
        B -->|三通道| D[彩色显示]
        C & D --> E[添加标题]
        E --> F[关闭坐标轴]
        F --> G[自动布局调整]
        G --> H[最终显示]
        """

    def save_image(self):
        # 检测要保存的图片和保存的路径是否为空
        if self.cartoon_image is not None and self.original_path is not None:
            try:
                # Create save path 获取原图的获取路径
                dir_path = os.path.dirname(self.original_path)
                # 获取文件名字和后缀名
                filename, ext = os.path.splitext(os.path.basename(self.original_path))
                # 生成新的文件名
                new_filename = f"cartoonified_{filename}{ext}"
                # 构建完整的保存名字
                save_path = os.path.join(dir_path, new_filename)

                # Convert back to BGR for saving 转换颜色空间，因为plt使用的是RGB图像，而Opencv保存图像需要BGR格式
                save_image = cv2.cvtColor(self.cartoon_image, cv2.COLOR_RGB2BGR)
                # 图像保存
                cv2.imwrite(save_path, save_image)
                # 反馈保存状态和路径
                messagebox.showinfo("Success", f"Image saved as:\n{new_filename}\nin:\n{dir_path}")
                # 更新状态栏
                self.status_label.config(text=f"Saved as {new_filename}")

            except Exception as e:
                # 异常处理：最多的异常是路径包含非法字符，opencv保存的时候路径中不能有中文
                messagebox.showerror("Save Error", f"Failed to save image: {str(e)}")
                self.status_label.config(text="Error saving image")


if __name__ == "__main__":
    root = tk.Tk()
    app = CartoonifyApp(root)
    root.mainloop()