
import torch
import torch.nn as nn
"""
(kernel_size, output_channels, stride, padding)
"M" MaxPool2D(kernel_size=2, stride=2)
[...kwargs..., num_count]
"""

YoloV1_architecture = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]


class cnn_block(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int):
        super(cnn_block, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.cnn_layer(x)


def create_seq_from_arch(in_channels, architecture:list):
    layer = []
    for x in architecture:
        if(isinstance(x, tuple)):
            layer += [
                cnn_block(
                    in_channels=in_channels,
                    out_channels=x[1],
                    kernel_size=x[0],
                    stride=x[2],
                    padding=x[3]
                )
            ]
            in_channels = x[1]
        elif(isinstance(x, str)):
            layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif(isinstance(x, list)):
            for _ in range(x[-1]):
                for i in range(len(x)-1):
                    layer.append(
                        cnn_block(
                            in_channels=in_channels,
                            out_channels=x[i][1],
                            kernel_size=x[i][0],
                            stride=x[i][2],
                            padding=x[i][3]
                        )
                    )
                    in_channels=x[i][1]
    return nn.Sequential(*layer)


def create_front_layer(split_size, num_boxes, num_classes):
    S, B, C = split_size, num_boxes, num_classes
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024*S*S, 1024),
        nn.Dropout(0.5),
        nn.LeakyReLU(0.1),
        nn.Linear(1024, S*S*(C+5*B))
    )


class YoloV1_model(nn.Module):
    def __init__(self, split_size:int =7, num_boxes:int=2, num_classes:int=1, in_channels:int=3):
        super(YoloV1_model, self).__init__()
        self.architecture = YoloV1_architecture
        self.in_channels = in_channels
        self.darknet = create_seq_from_arch(in_channels=in_channels, architecture=self.architecture)
        self.fcs = create_front_layer(split_size, num_boxes, num_classes)
    def forward(self, x):
        return self.fcs(self.darknet(x))


