import torch
from FCANet import fcanet50  # 确保 FCANet.py 与这个文件同目录，或修改为相对导入路径

def main():
    # 构建模型
    model = fcanet50(num_classes=17)
    model.eval()

    # 构造输入：batch size 1, 3通道，224x224图像
    input_tensor = torch.randn(1, 32, 64, 48)

    # 前向传播并打印输出形状
    with torch.no_grad():
        output = model(input_tensor)
        print("✅ Model forward successful!")
        print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
