import torchvision.models as models
import torch.nn as nn

# 1. 把现成的 ResNet18 模型拿过来
model = models.resnet18(pretrained=True)

# 2. 我们先用 print 偷看一下它的内部结构（你会发现最后一行是 fc 层）
print(model)
# 输出: Linear(in_features=512, out_features=1000, bias=True)
# 意思是：输入512维，输出1000类

# 3. 开始“换头”
# 我们建一个新的层，输入还是 512，但输出变成 5（对应 5 种花）
model.fc = nn.Linear(in_features=512, out_features=5)

# 4. 再次打印，头已经换了
print(model.fc)
# 输出: Linear(in_features=512, out_features=5, bias=True)
