import torch.nn as nn




def conv_block(in_channels, out_channels, pool, pad):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=pad),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(pool),
    )


class ConvNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ConvNet, self).__init__()
        self.conv_1 = nn.Sequential(conv_block(x_dim, hid_dim, 2, 1), )
        self.conv_2 = nn.Sequential(conv_block(hid_dim, hid_dim, 2, 1), )
        # self.conv_3 = nn.Sequential(conv_block(hid_dim, hid_dim, 4, 1), )
        self.conv_4 = nn.Sequential(conv_block(hid_dim, hid_dim, 4, 1), nn.Dropout(p=0.4))
        self.conv_out = nn.Sequential(nn.Conv2d(64, 32, 4), nn.BatchNorm2d(32), nn.ReLU())
        # ~ self.conv_out = nn.Sequential(nn.Conv2d(64, 32, 3,padding=1), nn.BatchNorm2d(32), nn.ReLU())


    def forward(self, x):
        # print('input shape:' + str(x.shape))
        x = self.conv_1(x)
        # print('conv_1:' + str(x.shape))
        x = self.conv_2(x)
        # print('conv_2:' + str(x.shape))
        # x = self.conv_3(x)
        # print('conv_3:' + str(x.shape))
        x = self.conv_4(x)
        # print('conv_4:' + str(x.shape))
        x = self.conv_out(x)
        # print('conv_out:' + str(x.shape))

        return x.view(x.size(0), -1)

#
#
# def conv_block(in_channels, out_channels, pool, pad):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=pad),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(),
#         nn.MaxPool2d(pool),
#     )
#
# class ConvNet(nn.Module):
#     def __init__(self, x_dim=3, hid_dim=64, z_dim=56):
#         super(ConvNet, self).__init__()
#         self.conv_1 = nn.Sequential(conv_block(3, 4, 3, 3), )
#         self.conv_2 = nn.Sequential(conv_block(4, 4, 51, 51), )
#         self.conv_3 = nn.Sequential(conv_block(4, 4, 2, 2), )
#         self.conv_4 = nn.Sequential(conv_block(4, 4, 2, 2), nn.Dropout(p=0.4))
#         self.conv_out = nn.Sequential(nn.Conv2d(4,4,2,2), nn.ReLU())
#                                       # nn.BatchNorm2d(32), nn.ReLU()) #this one
#
#     def forward(self, x):
#         # print()
#         # print('input shape:' + str(x.shape))
#         x = self.conv_1(x)
#         # print('conv_1:' + str(x.shape))
#         x = self.conv_2(x)
#         # print('conv_2:' + str(x.shape))
#         x = self.conv_3(x)
#         # print('conv_3:' + str(x.shape))
#         x = self.conv_4(x)
#         # print('conv_4:' + str(x.shape))
#         x = self.conv_out(x)
#         # print('conv_out:' + str(x.shape))
#         # print()
#
#         return x.view(x.size(0),-1)
#
# # #
# def conv_block(in_channels, out_channels, pool):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(),
#         nn.MaxPool2d(pool),
#
#     )
# class ConvNet(nn.Module):
#
#     def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
#         super(ConvNet, self).__init__()
#         self.encoder = nn.Sequential(
#             conv_block(x_dim, hid_dim,2),
#             conv_block(hid_dim, hid_dim,2),
#             conv_block(hid_dim, hid_dim,),
#             conv_block(hid_dim, hid_dim, 4),
#             nn.Dropout(p=0.4),
#             nn.Sequential(
#                 nn.Conv2d(hid_dim, z_dim, 1, padding=0),
#                 nn.BatchNorm2d(z_dim),
#                 nn.ReLU())
#         )
#         self.out_channels = 1600
#
#     def forward(self, x):
#         x = self.encoder(x)
        # print(x.shape)
#         return x.view(x.size(0), -1)
