import torchvision
from torch.utils.tensorboard import SummaryWriter
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()  # 组合一个转换操作：将图像转为Tensor
])
# 加载训练集
train_set = torchvision.datasets.CIFAR10(
    root="C:\\Users\\Air\\Desktop\\Code\\dataset",  # 数据集保存路径
    train=True,  # True表示加载训练集，False表示加载测试集
    transform=dataset_transform,  # 应用前面定义的转换（转为Tensor）
    download=True  # 若root路径下没有数据集，则自动从官网下载
)

# 加载测试集
test_set = torchvision.datasets.CIFAR10(
    root="C:\\Users\\Air\\Desktop\\Code\\dataset",  # 同训练集路径（可共用数据集文件夹）
    train=False,  # 加载测试集
    transform=dataset_transform,  # 同样应用转换
    download=True
)
writer=SummaryWriter("p10")
for i in range(20):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)
writer.close()