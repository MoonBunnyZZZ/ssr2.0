import hiddenlayer as hl
import torch
import torchvision.models

model = torchvision.models.resnet18()

transforms = [
    # Fold Conv, BN, RELU layers into one
    hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    # Fold Conv, BN layers together
    hl.transforms.Fold("Conv > BatchNorm", "ConvBn"),
    # Fold bottleneck blocks
    hl.transforms.Fold("""
        ((ConvBnRelu > ConvBn) | ConvBn) > Add > Relu
        """, "BasicBlock", "Basic Block"),
    # Fold residual blocks
    hl.transforms.Fold("""ConvBnRelu > ConvBn > Add > Relu""",
                       "ResBlock", "Residual Block"),
    # Fold repeated blocks
    hl.transforms.FoldDuplicates(),
]

# print(model)
# Display graph using the transforms above
graph = hl.build_graph(model, torch.zeros([1, 3, 112, 112]), transforms=transforms)
# graph.save('./graph.jpg')

