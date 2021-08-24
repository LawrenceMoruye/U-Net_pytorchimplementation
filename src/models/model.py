import torch 
import torch.nn as nn 


def double_convolution(in_channels,out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv


def crop_image_tensor(original_tensor,target_tensor):
    """
    format of tensor in pytorch batch_size,channels,Height,Width
    original_tensor:tensor to crop
    target_tensor:target tensor which shoukd be smaller than original tensor
    """
    original_tensor_size = original_tensor.size()[-1]
    target_tensor_size = target_tensor.size()[-1]
    change = original_tensor_size - target_tensor_size
    change = change // 2
    # print(original_tensor_size)
    # print(change)
    return original_tensor[:,:,change:original_tensor_size - change,change:original_tensor_size - change]



class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2_x_2 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.down_conv_block_1 = double_convolution(1,64)
        self.down_conv_block_2 = double_convolution(64,128)
        self.down_conv_block_3 = double_convolution(128,256)
        self.down_conv_block_4 = double_convolution(256,512)
        self.down_conv_block_5 = double_convolution(512,1024)

        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2,stride=2)
        self.up_conv_block_1 = double_convolution (1024,512)

        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2,stride=2)
        self.up_conv_block_2 = double_convolution (512,256)

        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2,stride=2)
        self.up_conv_block_3 = double_convolution (256,128)

        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2,stride=2)
        self.up_conv_block_4 = double_convolution (128,64)

        self.output =nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self,image):
        """
        bs,C,H,W
        Contacting Path

        """

        x1 =self. down_conv_block_1(image) ##concat1
        # print(x1.size())
        x2 = self.max_pool_2_x_2(x1)
        x3 = self. down_conv_block_2(x2)##concat2
        x4 = self.max_pool_2_x_2(x3)
        x5 = self. down_conv_block_3(x4)##concat3
        x6 = self.max_pool_2_x_2(x5)
        x7 = self. down_conv_block_4(x6)##concat4
        x8 = self.max_pool_2_x_2(x7)
        x9 = self. down_conv_block_5(x8)
        # print(x9.size())

        x = self.up_transpose_1(x9)
        y = crop_image_tensor(x7,x)
        x = self.up_conv_block_1(torch.cat([x,y],axis=1))

        x = self.up_transpose_2(x)
        y = crop_image_tensor(x5,x)
        x = self.up_conv_block_2(torch.cat([x,y],axis=1))

        x = self.up_transpose_3(x)
        y = crop_image_tensor(x3,x)
        x = self.up_conv_block_3(torch.cat([x,y],axis=1))

        x = self.up_transpose_4(x)
        y = crop_image_tensor(x1,x)
        x = self.up_conv_block_4(torch.cat([x,y],axis=1))

        #print("size of cropped x7",x.size())

        output =self.output(x)
        # print(output)
        # print(output.size())
        return output

        


if __name__== "__main__":
    image =torch.rand((1,1,572,572))
    model = UNet()
    model(image)

