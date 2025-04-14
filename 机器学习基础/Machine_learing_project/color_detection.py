import argparse
import pandas as pd
import cv2
import numpy as np
from typing import Tuple, Dict


class ColorDetector:
    def __init__(self, color_db_path: str = 'colors.csv'):
        """
        初始化颜色检测器
        :param color_db_path: 颜色数据库CSV文件路径
        """
        # 初始化全局状态
        self.clicked = False
        self.bgr_values = (0, 0, 0)
        self.position = (0, 0)

        # 加载颜色数据库
        self.color_db = self._load_color_db(color_db_path)

        # 创建显示窗口
        cv2.namedWindow('Color Detector')
        cv2.setMouseCallback('Color Detector', self._mouse_callback)

    def _load_color_db(self, path: str) -> pd.DataFrame:
        """加载颜色数据库并验证格式"""
        columns = ['color', 'color_name', 'hex', 'R', 'G', 'B']
        try:
            df = pd.read_csv(path, names=columns, header=None)
            # 验证必要列是否存在
            if not all(col in df.columns for col in ['R', 'G', 'B', 'color_name']):
                raise ValueError("颜色数据库缺少必要列")
            return df
        except Exception as e:
            raise RuntimeError(f"加载颜色数据库失败: {str(e)}")

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDBLCLK:  # 监听左键双击事件
            self.clicked = True
            self.position = (x, y)
            # 获取点击位置的BGR值（注意OpenCV是BGR顺序）
            self.bgr_values = tuple(map(int, self.img[y, x]))

    def _get_closest_color(self, bgr: Tuple[int, int, int]) -> Dict[str, str]:
        """
        查找最接近的颜色名称
        :param bgr: (B, G, R) 格式的颜色值
        :return: 包含颜色信息的字典
        """
        b, g, r = bgr
        # 计算与所有颜色的曼哈顿距离（向量化操作提升性能）
        distances = (
                (self.color_db['R'] - r).abs() +
                (self.color_db['G'] - g).abs() +
                (self.color_db['B'] - b).abs()
        )
        closest_idx = distances.idxmin()

        return {
            'name': self.color_db.loc[closest_idx, 'color_name'],
            'hex': self.color_db.loc[closest_idx, 'hex'],
            'rgb': (r, g, b),
            'bgr': (b, g, r)
        }

    def _draw_info_panel(self) -> None:
        """绘制颜色信息面板"""
        color_info = self._get_closest_color(self.bgr_values)
        b, g, r = self.bgr_values

        # 计算亮度决定文字颜色（浅色背景用深色文字）
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)

        # 绘制半透明背景
        overlay = self.img.copy()
        cv2.rectangle(overlay, (10, 10), (600, 90), (b, g, r), -1)
        cv2.addWeighted(overlay, 0.7, self.img, 0.3, 0, self.img)

        # 添加文字信息
        info_lines = [
            f"Color: {color_info['name']}",
            f"HEX: #{color_info['hex']}",
            f"RGB: ({r}, {g}, {b})",
            f"Position: {self.position}"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                self.img, line, (20, 40 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                text_color, 1, cv2.LINE_AA
            )

    def run(self, image_path: str) -> None:
        """
        运行颜色检测器
        :param image_path: 要检测的图片路径
        """
        try:
            # 读取图片
            self.img = cv2.imread(image_path)
            if self.img is None:
                raise ValueError("无法加载图片，请检查路径")

            # 主循环
            while True:
                cv2.imshow('Color Detector', self.img)

                if self.clicked:
                    self._draw_info_panel()
                    self.clicked = False

                # 按键处理
                key = cv2.waitKey(20)
                if key == 27:  # ESC退出
                    break
                elif key == ord('s'):  # 保存当前截图
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(f'color_detection_{timestamp}.png', self.img)
                    print(f"截图已保存为 color_detection_{timestamp}.png")

        finally:
            cv2.destroyAllWindows()


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(
        description="高级颜色检测工具",
        epilog="使用说明: python color_detector.py -i your_image.jpg"
    )
    parser.add_argument(
        '-i', '--image',
        required=True,
        help="要分析的图片路径"
    )
    parser.add_argument(
        '-c', '--color_db',
        default='colors.csv',
        help="颜色数据库路径 (默认: colors.csv)"
    )

    args = parser.parse_args()

    try:
        detector = ColorDetector(args.color_db)
        detector.run(args.image)
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()