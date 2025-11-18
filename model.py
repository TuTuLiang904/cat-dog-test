# model_pretrained.py
# 导入必要的库
from torch.utils.data import DataLoader  # 数据加载器
import torchvision  # 包含常用数据集和图像处理工具
import torch  # PyTorch核心库
from torch import nn  # 神经网络模块

# ----------------------
# 1) 模型定义（与你保持一致）
# ----------------------
class TUTU(nn.Module):  # 定义一个名为TUTU的神经网络类，继承自nn.Module
    def __init__(self):  # 初始化方法
        super(TUTU, self).__init__()  # 调用父类的初始化方法
        
        # 定义模型的层结构，使用nn.Sequential按顺序组合
        self.model = nn.Sequential(
            # 第一层卷积：输入通道3，输出通道32，卷积核大小5，填充2
            nn.Conv2d(3, 32, 5, padding=2),
            # 最大池化：池化核大小2
            nn.MaxPool2d(2),
            
            # 第二层卷积：输入通道32，输出通道32，卷积核大小5，步长1，填充2
            nn.Conv2d(32, 32, 5, 1, padding=2),
            # 最大池化：池化核大小2
            nn.MaxPool2d(2),
            
            # 第三层卷积：输入通道32，输出通道64，卷积核大小5，步长1，填充2
            nn.Conv2d(32, 64, 5, 1, 2),
            # 最大池化：池化核大小2
            nn.MaxPool2d(2),
            
            # 展平层：将多维张量转换为一维
            nn.Flatten(),
            
            # 全连接层1：输入特征数64*4*4，输出特征数64
            nn.Linear(64 * 4 * 4, 64),
            # 全连接层2：输入特征数64，输出特征数10（对应CIFAR10的10个类别）
            nn.Linear(64, 10)
        )

    def forward(self, x):  # 前向传播方法
        return self.model(x)  # 直接调用定义的模型结构

# ----------------------
# 2) 训练与评估逻辑
# ----------------------
def main(
    data_root=r"C:\\Users\\Air\\Desktop\\Code\\dataset",  # 数据集根目录
    batch_size: int = 64,  # 批次大小
    lr: float = 5e-2,  # 学习率
    epochs: int = 10,  # 训练轮数
    print_every: int = 100,  # 每隔多少个iter打印一次loss
    num_workers: int = 0  # 数据加载的线程数，Windows下建议先用0
):
    # 设备选择：优先使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 数据预处理：将图像转换为张量
    transform = torchvision.transforms.ToTensor()
    
    # 加载CIFAR10训练数据集
    train_data = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=transform, download=True
    )
    
    # 加载CIFAR10测试数据集
    test_data = torchvision.datasets.CIFAR10(
        root=data_root, train=False, transform=transform, download=True
    )

    # 获取数据集长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print(f"训练数据集的长度为：{train_data_size}")
    print(f"测试数据集的长度为：{test_data_size}")

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # 初始化模型并移至指定设备
    model = TUTU().to(device)
    
    # 定义损失函数：交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    
    # 定义优化器：随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # （可选）快速前向检查，确保shape正确
    with torch.no_grad():  # 不计算梯度，提高效率
        dummy = torch.ones((batch_size, 3, 32, 32), device=device)  # 创建一个批次的全1张量
        out = model(dummy)  # 前向传播
        print(f"[Sanity Check] dummy forward ok, output shape: {out.shape}")

    # 训练步数计数器
    total_train_step = 0
    total_test_step = 0

    # ----------------------
    # 开始训练
    # ----------------------
    for epoch in range(epochs):  # 遍历训练轮数
        print(f"\n第{epoch + 1}轮训练开始")
        model.train()  # 设置模型为训练模式
        
        try:
            for i, (imgs, targets) in enumerate(train_loader):  # 遍历训练数据
                # 将数据移至指定设备
                imgs = imgs.to(device)
                targets = targets.to(device)

                # 前向传播
                outputs = model(imgs)
                # 计算损失
                loss = loss_fn(outputs, targets)

                # 反向传播
                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数

                total_train_step += 1
                # 打印训练信息
                if (i + 1) % print_every == 0 or (i + 1) == len(train_loader):
                    print(f"训练步数 {total_train_step:6d} | "
                          f"Epoch {epoch + 1}/{epochs} | "
                          f"Iter {i + 1}/{len(train_loader)} | "
                          f"loss = {loss.item():4f}")
        except Exception as e:
            # 异常处理
            import traceback
            print("[Error] 训练阶段出现异常：", repr(e))
            traceback.print_exc()
            return

        # ----------------------
        # 简单测试评估
        # ----------------------
        model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        test_loss_sum = 00
        
        with torch.no_grad():  # 不计算梯度
            for imgs, targets in test_loader:  # 遍历测试数据
                # 将数据移至指定设备
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = new_func(model, imgs)
                # 计算损失
                loss = loss_fn(outputs, targets)
                test_loss_sum += loss.item() * imgs.size(0)

                # 计算准确率
                preds = outputs.argmax(dim=1)  # 获取预测类别
                correct += (preds == targets).sum().item()  # 统计正确预测数
                total += targets.size(0)  # 统计总数
                total_test_step += 1

        # 计算平均测试损失和准确率
        avg_test_loss = test_loss_sum / max(1, total)
        acc = correct / max(1, total)
        print(f"[Eval] 第{epoch + 1}轮 | 测试集平均loss: {avg_test_loss:.4f} | 准确率: {acc*100:.2f}%")

    print("\n[Done] 训练完成！")

def new_func(model, imgs):
    outputs = model(imgs)
    return outputs

# ----------------------
# 3) 入口
# ----------------------
if __name__ == "__main__":
    try:
        main()  # 调用主函数
    except Exception as e:
        # 顶层异常处理
        import traceback
        print("[Fatal] 程序异常退出：", repr(e))
        traceback.print_exc()