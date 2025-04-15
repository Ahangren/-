# import argparse
# import pandas as pd
# import cv2
# import numpy as np
# from typing import Tuple, Dict
#
#
# class ColorDetector:
#     def __init__(self, color_db_path: str = 'colors.csv'):
#         """
#         初始化颜色检测器
#         :param color_db_path: 颜色数据库CSV文件路径
#         """
#         # 初始化全局状态
#         self.clicked = False
#         self.bgr_values = (0, 0, 0)
#         self.position = (0, 0)
#
#         # 加载颜色数据库
#         self.color_db = self._load_color_db(color_db_path)
#
#         # 创建显示窗口
#         cv2.namedWindow('Color Detector')
#         cv2.setMouseCallback('Color Detector', self._mouse_callback)
#
#     def _load_color_db(self, path: str) -> pd.DataFrame:
#         """加载颜色数据库并验证格式"""
#         columns = ['color', 'color_name', 'hex', 'R', 'G', 'B']
#         try:
#             df = pd.read_csv(path, names=columns, header=None)
#             # 验证必要列是否存在
#             if not all(col in df.columns for col in ['R', 'G', 'B', 'color_name']):
#                 raise ValueError("颜色数据库缺少必要列")
#             return df
#         except Exception as e:
#             raise RuntimeError(f"加载颜色数据库失败: {str(e)}")
#
#     def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
#         """鼠标回调函数"""
#         if event == cv2.EVENT_LBUTTONDBLCLK:  # 监听左键双击事件
#             self.clicked = True
#             self.position = (x, y)
#             # 获取点击位置的BGR值（注意OpenCV是BGR顺序）
#             self.bgr_values = tuple(map(int, self.img[y, x]))
#
#     def _get_closest_color(self, bgr: Tuple[int, int, int]) -> Dict[str, str]:
#         """
#         查找最接近的颜色名称
#         :param bgr: (B, G, R) 格式的颜色值
#         :return: 包含颜色信息的字典
#         """
#         b, g, r = bgr
#         # 计算与所有颜色的曼哈顿距离（向量化操作提升性能）
#         distances = (
#                 (self.color_db['R'] - r).abs() +
#                 (self.color_db['G'] - g).abs() +
#                 (self.color_db['B'] - b).abs()
#         )
#         closest_idx = distances.idxmin()
#
#         return {
#             'name': self.color_db.loc[closest_idx, 'color_name'],
#             'hex': self.color_db.loc[closest_idx, 'hex'],
#             'rgb': (r, g, b),
#             'bgr': (b, g, r)
#         }
#
#     def _draw_info_panel(self) -> None:
#         """绘制颜色信息面板"""
#         color_info = self._get_closest_color(self.bgr_values)
#         b, g, r = self.bgr_values
#
#         # 计算亮度决定文字颜色（浅色背景用深色文字）
#         brightness = 0.299 * r + 0.587 * g + 0.114 * b
#         text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)
#
#         # 绘制半透明背景
#         overlay = self.img.copy()
#         cv2.rectangle(overlay, (10, 10), (600, 90), (b, g, r), -1)
#         cv2.addWeighted(overlay, 0.7, self.img, 0.3, 0, self.img)
#
#         # 添加文字信息
#         info_lines = [
#             f"Color: {color_info['name']}",
#             f"HEX: #{color_info['hex']}",
#             f"RGB: ({r}, {g}, {b})",
#             f"Position: {self.position}"
#         ]
#
#         for i, line in enumerate(info_lines):
#             cv2.putText(
#                 self.img, line, (20, 40 + i * 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                 text_color, 1, cv2.LINE_AA
#             )
#
#     def run(self, image_path: str) -> None:
#         """
#         运行颜色检测器
#         :param image_path: 要检测的图片路径
#         """
#         try:
#             # 读取图片
#             self.img = cv2.imread(image_path)
#             if self.img is None:
#                 raise ValueError("无法加载图片，请检查路径")
#
#             # 主循环
#             while True:
#                 cv2.imshow('Color Detector', self.img)
#
#                 if self.clicked:
#                     self._draw_info_panel()
#                     self.clicked = False
#
#                 # 按键处理
#                 key = cv2.waitKey(20)
#                 if key == 27:  # ESC退出
#                     break
#                 elif key == ord('s'):  # 保存当前截图
#                     timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
#                     cv2.imwrite(f'color_detection_{timestamp}.png', self.img)
#                     print(f"截图已保存为 color_detection_{timestamp}.png")
#
#         finally:
#             cv2.destroyAllWindows()
#
#
# def main():
#     # 设置命令行参数
#     parser = argparse.ArgumentParser(
#         description="高级颜色检测工具",
#         epilog="使用说明: python color_detector.py -i your_image.jpg"
#     )
#     parser.add_argument(
#         '-i', '--image',
#         required=True,
#         help="要分析的图片路径"
#     )
#     parser.add_argument(
#         '-c', '--color_db',
#         default='colors.csv',
#         help="颜色数据库路径 (默认: colors.csv)"
#     )
#
#     args = parser.parse_args()
#
#     try:
#         detector = ColorDetector(args.color_db)
#         detector.run(args.image)
#     except Exception as e:
#         print(f"错误: {str(e)}")
#         exit(1)
#
#
# if __name__ == "__main__":
#     main()


import argparse
import pandas as pd
import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime
import os


class ColorDetector:
    def __init__(self, color_db_path: str = 'colors.csv'):
        """
        初始化颜色检测器
        :param color_db_path: 颜色数据库CSV文件路径
        """
        # 初始化状态
        self.clicked = False   # 初始化鼠标点击标识
        self.bgr_values = (0, 0, 0)  # 初始化rgb值
        self.position = (0, 0)  # 初始化鼠标点击位置
        self.img: Optional[np.ndarray] = None  # 初始化图片

        # 加载颜色数据库（带缓存）
        self.color_db = self._load_color_db(color_db_path)
        self._precompute_colors()

    def _load_color_db(self, path: str) -> pd.DataFrame:
        """加载并验证颜色数据库"""
        try:
            df = pd.read_csv(path, names=['color', 'color_name', 'hex', 'R', 'G', 'B'], header=None)

            # 数据验证
            if df.empty:
                raise ValueError("颜色数据库为空")
            if not all(col in df.columns for col in ['R', 'G', 'B', 'color_name']):
                raise ValueError("颜色数据库缺少必要列")

            # 转换颜色值为整数
            df[['R', 'G', 'B']] = df[['R', 'G', 'B']].astype(int)
            return df

        except Exception as e:
            raise RuntimeError(f"加载颜色数据库失败: {str(e)}")

    def _precompute_colors(self) -> None:
        """预计算颜色数据加速查找"""
        self.color_array = self.color_db[['R', 'G', 'B']].values

    def _mouse_callback(self, event: int, x: int, y: int, *_) -> None:
        """鼠标回调函数（优化性能）"""
        if event == cv2.EVENT_LBUTTONDBLCLK and self.img is not None:  # 是否双击鼠标，点击图片是否存在
            self.clicked = True  # 将鼠标点击设为True
            self.position = (x, y)  # 获取当前点击图片的xy坐标值
            self.bgr_values = tuple(map(int, self.img[y, x]))  # 获取当前的RGB值

    def _get_closest_color(self, bgr: Tuple[int, int, int]) -> Dict[str, str]:
        """
        查找最接近的颜色（使用向量化计算优化性能）
        :param bgr: (B, G, R) 格式的颜色值，opencv中使用的是BGR，其他的使用的是RGB
        :return: 包含颜色信息的字典
        """
        b, g, r = bgr
        # 使用numpy向量化计算曼哈顿距离
        distances = np.sum(np.abs(self.color_array - [r, g, b]), axis=1)  # 计算加载出来的数据与当前点击位置的RGB的距离
        closest_idx = np.argmin(distances)  # 返回距离最小的颜色下标

        return {
            'name': self.color_db.loc[closest_idx, 'color_name'],  # 获取当前颜色的名字
            'hex': self.color_db.loc[closest_idx, 'hex'],  # 获取当前颜色的hex
            'rgb': (r, g, b),
            'bgr': (b, g, r)
        }

    def _draw_info_panel(self) -> None:
        """信息面板绘制"""
        if self.img is None:
            return

        color_info = self._get_closest_color(self.bgr_values)
        b, g, r = self.bgr_values

        # 智能文字颜色选择
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)

        # 创建信息面板
        panel = np.zeros((100, 600, 3), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (600, 100), (b, g, r), -1)

        # 添加文字信息
        info_lines = [
            f"Color: {color_info['name']}",
            f"HEX: #{color_info['hex']}",
            f"RGB: ({r}, {g}, {b})  BGR: ({b}, {g}, {r})",
            f"Position: {self.position}"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                panel, line, (10, 30 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                text_color, 1, cv2.LINE_AA
            )

        # 合并到原图
        self.img = cv2.vconcat([panel, self.img])

    def run(self, image_path: str) -> None:
        """
        运行颜色检测器
        :param image_path: 要检测的图片路径
        """
        try:
            # 验证图片路径
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")

            # 读取图片（支持中文路径）
            self.img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.img is None:
                raise ValueError("无法解码图片，可能是不支持的格式")

            # 创建窗口并设置回调
            cv2.namedWindow('Color Detector', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Color Detector', self._mouse_callback)

            while True:
                display_img = self.img.copy()
                cv2.imshow('Color Detector', display_img)

                key = cv2.waitKey(20)
                if key == 27:  # ESC退出
                    break
                elif key == ord('s'):  # 保存截图
                    self._save_screenshot()
                elif key == ord('h'):  # 显示帮助
                    print("操作指南: 双击选取颜色 | ESC退出 | S保存截图 | H显示帮助")

        except Exception as e:
            print(f"运行时错误: {str(e)}")
        finally:
            cv2.destroyAllWindows()

    def _save_screenshot(self) -> None:
        """保存截图功能"""
        if self.img is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'color_detection_{timestamp}.png'
            try:
                cv2.imwrite(filename, self.img)
                print(f"截图已保存为 {filename}")
            except Exception as e:
                print(f"保存截图失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="高级颜色检测工具 v2.0",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""使用示例:
  python color_detector.py -i test.jpg
  python color_detector.py -i test.jpg -c custom_colors.csv

快捷键:
  ESC   退出程序
  S     保存当前截图
  H     显示帮助信息"""
    )

    parser.add_argument(
        '-i', '--image',
        required=True,
        help="要分析的图片路径（支持中文路径）"
    )
    parser.add_argument(
        '-c', '--color_db',
        default='colors.csv',
        help="自定义颜色数据库路径 (默认: colors.csv)"
    )

    args = parser.parse_args()

    try:
        print("启动颜色检测器...")
        detector = ColorDetector(args.color_db)
        print("操作提示: 双击图片选取颜色，按S保存截图，ESC退出")
        detector.run(args.image)
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()