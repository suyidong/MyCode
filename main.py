from models.CNN import CnnNet
from models.mnist_dataset import MnistData
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox


# 预测类
class Pred:
    def __init__(self):
        # 动态修复模块路径
        if getattr(sys, 'frozen', False):
            # 打包模式：将模块路径添加到搜索路径
            sys.path.append(os.path.join(sys._MEIPASS, 'models'))

        # 显式导入模块（关键修复）
        from models.CNN import CnnNet
        import torch.serialization
        torch.serialization.add_safe_globals([CnnNet])

        # 模型路径处理
        base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
        model_path = os.path.join(base_path, 'checkpoints', 'best_model.pth')

        # 加载模型（先用非安全模式测试）
        self.model = torch.load(
            model_path,
            map_location='cpu',
            weights_only=False  # 测试通过后改为True
        )
        self.device = torch.device('cpu')
        self.labels = list(range(10))  # 添加缺失的属性

    # 预测
    def predict(self, img_path):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = transform(img)
        img = img.view(1, 3, 28, 28).to(self.device)
        output = self.model(img)
        output = torch.softmax(output, dim=1)
        # 每个预测值的概率
        probability = output.cpu().detach().numpy()[0]
        # 找出最大概率值的索引
        output = torch.argmax(output, dim=1)
        index = output.cpu().numpy()[0]
        # 预测结果
        pred = self.labels[index]
        return pred, probability[index]


# GUI类
class DigitRecognizerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("手写数字识别")

        self.pred = Pred()

        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=10, pady=10)

        self.btn_select = tk.Button(self.frame, text="选择图片", command=self.select_image)
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.label_result = tk.Label(self.frame, text="预测结果: ")
        self.label_result.pack(side=tk.LEFT, padx=5)

    def select_image(self):
        img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if img_path:
            pred, probability = self.pred.predict(img_path)
            self.label_result.config(text=f"预测结果: {pred}, 概率: {probability:.2f}")
            messagebox.showinfo("预测结果", f"数字: {pred}, 概率: {probability:.2f}")


if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
root.mainloop()
