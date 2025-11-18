# 导入GUI相关库（用于构建图形界面）
import tkinter as tk
from tkinter import filedialog, messagebox, ttk  # 文件选择、消息框、高级组件

# 导入PyTorch核心库（深度学习框架）
import torch
import torch.nn as nn  # 神经网络层模块
import torch.optim as optim  # 优化器模块

# 导入数据加载相关工具
from torch.utils.data import DataLoader  # 批量数据加载器
from torchvision import datasets, models, transforms  # 数据集、预训练模型、图像变换

# 导入图像处理库（用于图片显示和预处理）
from PIL import Image, ImageTk  # 图片打开、格式转换

# 导入其他辅助库
import os  # 文件路径操作
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图（概率分布图）
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 嵌入Tkinter的绘图组件
from tqdm import tqdm  # 训练进度条
import time  # 时间戳记录
import matplotlib.font_manager as fm  # 字体管理（解决中文显示问题）

# -------------------------- 修复中文字体问题 --------------------------
# 定义中文字体设置函数（Windows系统默认自带字体，无需额外安装）
def setup_chinese_font():
    try:
        # 定义Windows系统常见的中文字体列表（按兼容性排序）
        windows_fonts = [
            'Microsoft YaHei',      # 微软雅黑（Windows 7+默认自带）
            'SimHei',               # 黑体（兼容所有Windows版本）
            'SimSun',               # 宋体
            'FangSong',             # 仿宋
            'KaiTi',                # 楷体
            'Microsoft JhengHei'     # 微软正黑体（繁体兼容）
        ]
        
        # 遍历字体列表，选择系统已安装的第一个可用中文字体
        for font_name in windows_fonts:
            # 检查当前字体是否在系统字体列表中
            if font_name in [f.name for f in fm.fontManager.ttflist]:
                plt.rcParams['font.sans-serif'] = [font_name]  # 设置matplotlib中文字体
                break
        else:
            # 如果没有找到上述中文字体，使用默认英文 fallback 字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        # 解决matplotlib中负号（-）显示为方块的问题
        plt.rcParams['axes.unicode_minus'] = False
        # 打印当前使用的字体（方便调试）
        print(f"已设置中文字体：{plt.rcParams['font.sans-serif'][0]}")
    except Exception as e:
        # 异常处理：字体设置失败时，强制使用默认字体避免程序崩溃
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"字体设置失败，使用默认字体：{e}")

# 调用函数初始化中文字体（程序启动时执行）
setup_chinese_font()

# -------------------------- 核心配置参数 --------------------------
# 花卉类别列表（必须与数据集文件夹名完全一致）
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
# 数据集路径（你的本地数据集位置，已按需求预设）
DATASET_PATH = "C:\\Users\\Air\\Desktop\\模型训练\\花卉识别\\flowers_5"
# 训练好的模型保存路径（训练完成后自动生成）
MODEL_PATH = "flower_model.pth"

# -------------------------- 主应用类（封装所有功能） --------------------------
class FlowerRecognitionApp:
    # 类初始化（程序启动时执行）
    def __init__(self, root):
        self.root = root  # 接收主窗口实例
        self.root.title("花卉识别系统")  # 设置窗口标题
        self.root.geometry("1000x700")  # 设置窗口初始大小（宽x高）
        
        # 初始化核心变量
        self.image_path = None  # 存储导入的图片路径
        self.model = None  # 存储神经网络模型实例
        # 自动检测设备：CUDA可用则用GPU，否则用CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = None  # 训练集数据加载器
        self.test_loader = None  # 测试集数据加载器
        
        # 创建所有UI组件（按钮、显示区域等）
        self.create_widgets()
        
        # 检查数据集是否存在且完整
        self.check_dataset()
        
        # 尝试加载已训练好的模型（如果存在）
        self.load_model()

    # 创建UI界面组件
    def create_widgets(self):
        # 主框架（包裹所有组件，方便布局）
        main_frame = ttk.Frame(self.root, padding=10)  # padding：内边距10像素
        main_frame.pack(fill=tk.BOTH, expand=True)  # 填充整个窗口，支持拉伸
        
        # 左侧控制面板（训练、导入、识别按钮 + 状态显示）
        control_frame = ttk.LabelFrame(main_frame, text="控制中心", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))  # 靠左排列，填充垂直方向
        
        # 训练模型按钮：点击触发start_training方法
        self.train_btn = ttk.Button(
            control_frame, text="训练模型", command=self.start_training
        )
        self.train_btn.pack(fill=tk.X, pady=5)  # 填充水平方向，上下间距5像素
        
        # 导入图片按钮：点击触发import_image方法
        self.import_btn = ttk.Button(
            control_frame, text="导入图片", command=self.import_image
        )
        self.import_btn.pack(fill=tk.X, pady=5)
        
        # 识别花卉按钮：默认禁用（导入图片且模型加载后启用）
        self.classify_btn = ttk.Button(
            control_frame, text="识别花卉", command=self.classify_image, state=tk.DISABLED
        )
        self.classify_btn.pack(fill=tk.X, pady=5)
        
        # 状态信息显示框
        self.status_label = ttk.LabelFrame(control_frame, text="状态信息")
        self.status_label.pack(fill=tk.BOTH, expand=True, pady=10)  # 填充剩余空间
        
        # 状态文本框（显示训练/识别日志）
        self.status_text = tk.Text(self.status_label, height=15, width=30, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_text.insert(tk.END, "程序已启动，等待操作...\n")  # 初始提示
        self.status_text.config(state=tk.DISABLED)  # 设置为只读模式
        
        # 右侧显示区域（图片预览 + 识别结果）
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # 靠右排列，填充整个右侧
        
        # 图片预览区
        self.image_frame = ttk.LabelFrame(right_frame, text="图片预览")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))  # 上下间距10像素
        
        # 识别结果显示区
        self.result_frame = ttk.LabelFrame(right_frame, text="识别结果")
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 结果文本标签（显示识别类别和置信度）
        self.result_text = ttk.Label(self.result_frame, text="请导入图片并点击识别", font=("Arial", 12))
        self.result_text.pack(pady=10)  # 上下间距10像素
        
        # 概率分布图表（matplotlib嵌入Tkinter）
        self.fig, self.ax = plt.subplots(figsize=(5, 3))  # 创建图表实例（宽5英寸，高3英寸）
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.result_frame)  # 转换为Tkinter组件
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # 填充结果区

    # 日志输出函数（在状态文本框显示信息）
    def log(self, message):
        """在状态区域显示带时间戳的日志"""
        self.status_text.config(state=tk.NORMAL)  # 临时启用可编辑模式
        # 插入时间戳+消息（时间格式：时:分:秒）
        self.status_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.status_text.see(tk.END)  # 自动滚动到最后一行
        self.status_text.config(state=tk.DISABLED)  # 恢复只读模式
        self.root.update_idletasks()  # 强制刷新界面（避免日志卡顿）

    # 检查数据集是否存在且完整
    def check_dataset(self):
        """验证数据集路径和类别文件夹完整性"""
        # 检查数据集根路径是否存在
        if not os.path.exists(DATASET_PATH):
            self.log(f"错误：数据集路径不存在 - {DATASET_PATH}")
            messagebox.showerror("路径错误", f"未找到数据集，请确认路径：\n{DATASET_PATH}")
            return False
        
        # 检查5个类别文件夹是否都存在
        for cls in CLASSES:
            cls_path = os.path.join(DATASET_PATH, cls)  # 拼接类别文件夹路径
            if not os.path.exists(cls_path):
                self.log(f"错误：缺少类别文件夹 - {cls}")
                messagebox.showerror("数据错误", f"缺少类别文件夹：{cls}")
                return False
        
        # 数据集检查通过
        self.log(f"数据集检查成功，路径：{DATASET_PATH}")
        return True

    # 准备训练集和测试集数据加载器
    def prepare_data(self):
        """数据预处理、增强和加载（返回训练/测试集加载器）"""
        # 训练集数据增强（提升模型泛化能力）
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪为224x224（模拟不同拍摄角度）
            transforms.RandomHorizontalFlip(),  # 随机水平翻转（左右镜像）
            transforms.RandomRotation(15),  # 随机旋转±15度
            transforms.ToTensor(),  # 转换为PyTorch张量（0-1浮点型）
            # 标准化（使用ImageNet数据集的均值和标准差，适配预训练模型）
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 测试集数据预处理（仅标准化，不增强）
        test_transform = transforms.Compose([
            transforms.Resize(256),  # 缩放到256x256
            transforms.CenterCrop(224),  # 中心裁剪为224x224（保证输入尺寸一致）
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加载完整数据集（按文件夹自动分类标签）
        full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=train_transform)
        # 按8:2分割训练集和测试集
        train_size = int(0.8 * len(full_dataset))  # 80%用于训练
        test_size = len(full_dataset) - train_size  # 20%用于测试
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]  # 随机分割
        )
        
        # 给测试集单独设置预处理（覆盖训练集的增强变换）
        test_dataset.dataset.transform = test_transform
        
        # 创建数据加载器（批量加载数据，提升训练效率）
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=32,  # 每批32张图片
            shuffle=True,  # 训练集打乱顺序（避免过拟合）
            num_workers=0  # Windows系统设为0，避免多线程报错
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=32,
            shuffle=False,  # 测试集不打乱（便于评估）
            num_workers=0
        )
        
        # 日志输出数据量信息
        self.log(f"数据准备完成：训练集{train_size}张，测试集{test_size}张")
        return True

    # 构建神经网络模型（基于ResNet18迁移学习）
    def build_model(self):
        """构建并初始化ResNet18迁移学习模型"""
        # 加载预训练的ResNet18模型（适配不同torchvision版本）
        try:
            # 新版本torchvision（0.13+）使用weights参数
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except:
            # 旧版本torchvision使用pretrained参数
            self.model = models.resnet18(pretrained=True)
        
        # 冻结大部分网络层（只训练最后10层，节省计算资源）
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False  # 冻结参数，不计算梯度
        
        # 修改最后一层全连接层（适配5个花卉类别）
        num_ftrs = self.model.fc.in_features  # 获取最后一层输入特征数
        self.model.fc = nn.Linear(num_ftrs, len(CLASSES))  # 替换为5输出的全连接层
        
        # 将模型移到GPU/CPU设备
        self.model = self.model.to(self.device)
        self.log(f"模型构建完成，使用设备：{self.device}")
        return self.model

    # 模型训练主函数
    def train_model(self, epochs=10):
        """训练模型（epochs：训练轮数，默认10轮）"""
        # 检查数据加载器是否已创建，未创建则自动准备
        if not self.train_loader or not self.test_loader:
            if not self.prepare_data():
                return
        
        # 检查模型是否已构建，未构建则自动创建
        if not self.model:
            self.build_model()
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失（多分类任务常用）
        # Adam优化器：只优化未冻结的参数（requires_grad=True）
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=0.001  # 学习率（控制参数更新步长）
        )
        
        # 输出训练开始信息
        self.log(f"开始训练，共{epochs}轮")
        self.log(f"使用设备：{self.device}")
        
        # 训练循环（每轮遍历一次完整训练集）
        for epoch in range(epochs):
            self.model.train()  # 开启训练模式（启用Dropout等训练特有的层）
            running_loss = 0.0  # 累计训练损失
            correct = 0  # 正确预测数
            total = 0  # 总样本数
            
            # 创建训练进度条（tqdm可视化训练进度）
            pbar = tqdm(
                enumerate(self.train_loader), 
                total=len(self.train_loader), 
                desc=f"Epoch {epoch+1}/{epochs}"
            )
            
            # 遍历训练集每一批次
            for i, (inputs, labels) in pbar:
                # 将数据移到GPU/CPU设备（与模型同设备）
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 清空梯度（避免上一批次梯度累积）
                optimizer.zero_grad()
                
                # 前向传播：模型预测
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)  # 获取预测类别（概率最大的索引）
                loss = criterion(outputs, labels)  # 计算损失值
                
                # 反向传播+参数更新
                loss.backward()  # 计算梯度（反向传播）
                optimizer.step()  # 优化器更新模型参数
                
                # 统计训练指标
                running_loss += loss.item()  # 累积损失
                total += labels.size(0)  # 累积样本数
                correct += (predicted == labels).sum().item()  # 累积正确数
                
                # 更新进度条显示（实时显示损失和准确率）
                pbar.set_postfix({
                    "Loss": running_loss/(i+1),  # 平均损失
                    "Accuracy": f"{100*correct/total:.2f}%"  # 训练准确率
                })
            
            # 每轮训练结束后，计算训练集准确率和测试集性能
            train_acc = 100 * correct / total  # 训练集准确率
            test_acc, test_loss = self.evaluate_model()  # 测试集评估
            
                        # 日志输出本轮训练结果
            self.log(f"Epoch {epoch+1}/{epochs} - 训练损失: {running_loss/len(self.train_loader):.4f}, "
                    f"训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%")
        
        # 训练完成后保存模型权重（仅保存参数，不保存完整模型结构）
        torch.save(self.model.state_dict(), MODEL_PATH)
        self.log(f"模型训练完成，已保存至 {MODEL_PATH}")
        return True

    # 模型评估函数（计算测试集准确率和损失）
    def evaluate_model(self):
        """在测试集上评估模型性能，返回准确率和平均损失"""
        self.model.eval()  # 开启评估模式（禁用Dropout、固定BatchNorm等）
        test_loss = 0.0  # 测试集累计损失
        correct = 0  # 测试集正确预测数
        total = 0  # 测试集总样本数
        criterion = nn.CrossEntropyLoss()  # 与训练一致的损失函数
        
        # 禁用梯度计算（评估阶段不需要反向传播，节省内存和计算）
        with torch.no_grad():
            # 遍历测试集每一批次
            for inputs, labels in self.test_loader:
                # 数据移到对应设备
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 前向传播预测
                outputs = self.model(inputs)
                # 计算测试损失
                loss = criterion(outputs, labels)
                # 获取预测类别（概率最大的索引）
                _, predicted = torch.max(outputs.data, 1)
                
                # 累积统计指标
                test_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算测试集准确率和平均损失
        test_acc = 100 * correct / total
        avg_loss = test_loss / len(self.test_loader)
        return test_acc, avg_loss

    # 加载已训练的模型权重
    def load_model(self):
        """从保存路径加载模型权重，恢复训练好的模型"""
        if os.path.exists(MODEL_PATH):  # 检查模型文件是否存在
            try:
                # 先构建模型结构（与训练时一致）
                self.build_model()
                # 加载模型权重，自动映射到当前设备（GPU/CPU）
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                self.model.eval()  # 设置为评估模式（重要！避免训练模式影响预测）
                self.log(f"已加载模型：{MODEL_PATH}")
                return True
            except Exception as e:
                # 加载失败时提示错误
                self.log(f"模型加载失败：{str(e)}")
                messagebox.showerror("模型错误", f"加载模型失败：{str(e)}")
                self.model = None
        else:
            # 未找到模型文件，提示用户先训练
            self.log("未找到已训练模型，请先训练模型")
        return False

    # 启动训练（带训练轮数输入对话框）
    def start_training(self):
        """弹出对话框让用户设置训练轮数，确认后启动训练"""
        # 先检查数据集是否完整
        if not self.check_dataset():
            return
        
        # 创建训练轮数输入窗口（子窗口）
        epochs_window = tk.Toplevel(self.root)
        epochs_window.title("设置训练轮数")  # 子窗口标题
        epochs_window.geometry("300x150")  # 子窗口大小
        epochs_window.transient(self.root)  # 依附主窗口
        epochs_window.grab_set()  # 锁定焦点（只能操作子窗口）
        
        # 输入提示标签
        ttk.Label(epochs_window, text="请输入训练轮数（建议10-30）：").pack(pady=10)
        # 轮数输入框
        epochs_entry = ttk.Entry(epochs_window, width=10)
        epochs_entry.pack(pady=5)
        epochs_entry.insert(0, "15")  # 默认值15轮
        
        # 确认训练的回调函数
        def confirm_training():
            try:
                epochs = int(epochs_entry.get())  # 读取输入的轮数
                if epochs < 1:  # 轮数必须为正整数
                    raise ValueError("轮数必须为正数")
                epochs_window.destroy()  # 关闭子窗口
                self.train_btn.config(state=tk.DISABLED)  # 禁用训练按钮（避免重复训练）
                self.log("开始准备训练数据...")
                self.root.update()  # 刷新界面
                self.train_model(epochs=epochs)  # 启动训练
                self.train_btn.config(state=tk.NORMAL)  # 训练完成后启用按钮
            except ValueError as e:
                # 输入错误提示
                messagebox.showerror("输入错误", str(e))
        
        # 确认按钮
        ttk.Button(epochs_window, text="确认", command=confirm_training).pack(pady=10)

    # 导入图片并显示
    def import_image(self):
        """打开文件选择对话框，导入花卉图片并在界面预览"""
        # 打开文件选择框（仅允许选择图片格式）
        file_path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        
        if file_path:  # 如果用户选择了图片
            self.image_path = file_path  # 保存图片路径
            # 模型加载成功后才启用识别按钮
            self.classify_btn.config(state=tk.NORMAL if self.model else tk.DISABLED)
            
            # 显示图片到预览区
            try:
                # 清空预览区之前的内容
                for widget in self.image_frame.winfo_children():
                    widget.destroy()
                
                # 打开图片并调整大小（避免图片过大撑破界面）
                image = Image.open(file_path)
                image.thumbnail((600, 400))  # 限制最大尺寸为600x400
                photo = ImageTk.PhotoImage(image)  # 转换为Tkinter支持的图片格式
                
                # 显示图片
                image_label = ttk.Label(self.image_frame, image=photo)
                image_label.image = photo  # 保持图片引用（避免被垃圾回收）
                image_label.pack(fill=tk.BOTH, expand=True)  # 填充预览区
                
                # 日志和结果区提示
                self.log(f"已导入图片：{os.path.basename(file_path)}")  # 显示图片文件名
                self.result_text.config(text="请点击识别按钮")  # 提示用户识别
                
                # 清空之前的概率图表
                self.ax.clear()
                self.canvas.draw()
                
            except Exception as e:
                # 图片加载失败提示
                self.log(f"图片加载失败：{str(e)}")
                messagebox.showerror("图片错误", f"加载图片失败：{str(e)}")

    # 识别花卉种类
    def classify_image(self):
        """使用训练好的模型识别导入的图片，返回类别和置信度"""
        # 检查是否已导入图片和加载模型
        if not self.image_path or not self.model:
            return
        
        try:
            # 图像预处理（与训练集预处理一致，保证输入格式相同）
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # 加载图片并预处理
            image = Image.open(self.image_path).convert('RGB')  # 打开图片并转为RGB格式
            image_tensor = transform(image).unsqueeze(0)  # 预处理后添加批次维度（模型要求输入为批次）
            image_tensor = image_tensor.to(self.device)  # 移到对应设备
            
            # 模型预测
            with torch.no_grad():  # 禁用梯度计算
                self.model.eval()  # 确保模型在评估模式
                outputs = self.model(image_tensor)  # 前向传播获取输出
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # 转换为概率（0-1）
                _, predicted = torch.max(outputs, 1)  # 获取概率最大的类别索引
                predicted_class = CLASSES[predicted.item()]  # 映射到花卉类别名称
                confidence = probabilities[predicted.item()].item() * 100  # 计算置信度（转为百分比）
            
            # 在界面显示识别结果
            self.result_text.config(
                text=f"识别结果：{predicted_class}\n置信度：{confidence:.2f}%"
            )
            
            # 绘制各类别概率柱状图
            self.ax.clear()  # 清空之前的图表
            y_pos = np.arange(len(CLASSES))  # 类别索引
            prob_values = probabilities.cpu().numpy() * 100  # 概率值（转为百分比，CPU上计算）
            self.ax.barh(y_pos, prob_values, align='center', color='skyblue')  # 水平柱状图
            self.ax.set_yticks(y_pos)  # 设置y轴刻度
            self.ax.set_yticklabels(CLASSES)  # y轴标签为类别名称
            self.ax.invert_yaxis()  # 反转y轴（让概率最高的在上方）
            self.ax.set_xlabel('概率 (%)')  # x轴标签
            self.ax.set_title('各类别识别概率')  # 图表标题
            
            # 在柱状图上标注具体概率值
            for i, v in enumerate(prob_values):
                self.ax.text(v + 1, i, f'{v:.1f}%', va='center')  # 标注在柱子右侧
            
            self.canvas.draw()  # 刷新图表
            self.log(f"识别完成：{predicted_class} (置信度：{confidence:.2f}%)")  # 日志记录结果
            
        except Exception as e:
            # 识别失败提示
            self.log(f"识别失败：{str(e)}")
            messagebox.showerror("识别错误", f"识别过程出错：{str(e)}")

# 程序入口（当脚本直接运行时执行）
if __name__ == "__main__":
    root = tk.Tk()  # 创建主窗口实例
    app = FlowerRecognitionApp(root)  # 初始化应用类
    root.mainloop()  # 启动主窗口消息循环（保持窗口显示）