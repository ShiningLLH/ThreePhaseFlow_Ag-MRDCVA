import torch
import torch.nn as nn

class Fea_encoder(nn.Module):       # 数据-特征编码器
    def __init__(self):
        super(Fea_encoder, self).__init__()
        self.Net1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=4,stride=1,padding='same'), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=4, stride=1, padding='same'), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),  # 铺平
            nn.Linear(in_features=32, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=7)
        )

    def forward(self, input):
        input = input.reshape(-1, 1, 7)   # 把二维变为三维数据
        x = self.Net1(input)
        return x

class Output_encoder(nn.Module):       # 特征-属性编码器, 一定程度上进行监督, 不是最终的特征-属性映射
    def __init__(self):
        super(Output_encoder, self).__init__()
        self.Net = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding='same'), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding='same'), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),  # 铺平
            nn.Linear(in_features=32, out_features=16, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=11),      # 11个属性
        )

    def forward(self, input):
        input = input.reshape(-1, 1, 7)
        x = self.Net(input)
        return x

# 一维卷积
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.Fea_encoder = Fea_encoder()        # 实例化数据-特征编码器
        self.Output_encoder = Output_encoder()  # 实例化特征-属性编码器

    def forward_once(self, x):
        output = self.Fea_encoder(x)
        output = torch.flatten(output, 1)   # 将向量展平
        return output

    def forward(self, input1):
        output1 = self.forward_once(input1)
        output_attri = self.Output_encoder(output1)
        return output1, output_attri

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 150)
        self.fc4 = nn.Linear(150, 50)
        self.fc5 = nn.Linear(50, output_size)  # 最后一个全连接层，输入为50维，输出为7维

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # 最后一层不需要激活函数，因为输出层可以根据需要选择合适的激活函数
        return x
