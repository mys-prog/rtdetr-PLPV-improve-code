import torch
import torch.nn as nn
from einops import rearrange
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, DWConvTranspose2d, CBAM
from .transformer import TransformerBlock



__all__ = (
    "GatedUp",
    "GatedDwon",
    "CSP_PAC",
    "ParallelAtrousConv",
    "CrossAttentionBlock",
)


class GatedUp(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(inc, inc, 1),
            nn.Hardsigmoid()
        )

        self.conv = Conv(inc, inc, k=1)
        self.up_branch1 = DWConvTranspose2d(inc, inc // 2, 2, 2)
        self.up_branch2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc, inc // 2, k=1)
        )

    def forward(self, x):
        channel_gate = self.gate(self.globalpool(x))
        x_up = torch.cat([self.up_branch1(x), self.up_branch2(x)], dim=1) * channel_gate
        output = self.conv(x_up)
        return output


class GatedDwon(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(inc, inc, 1),
            nn.Hardsigmoid()
        )

        self.conv = Conv(inc, inc, k=1)
        self.down_branch1 = nn.Sequential(
            CBAM(inc),
            RepConv(inc, inc//2, k=3, s=2)
        )
        self.down_branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(inc, inc // 2, k=1)
        )

    def forward(self, x):
        channel_gate = self.gate(self.globalpool(x))
        x_up = torch.cat([self.down_branch1(x), self.down_branch2(x)], dim=1) * channel_gate
        output = self.conv(x_up)
        return output


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        dim1, dim2 = dim
        if dim1 != dim2:
            self.conv1x1 = Conv(dim1, dim2)
        else:
            self.conv1x1 = nn.Identity()

        self.q = nn.Conv2d(dim2, dim2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1, groups=dim2, bias=bias)
        self.kv = nn.Conv2d(dim2, dim2*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim2*2, dim2*2, kernel_size=3, stride=1, padding=1, groups=dim2*2, bias=bias)
        self.project_out = nn.Conv2d(dim2, dim2, kernel_size=1, bias=bias)

    def forward(self, inputs):
        x, y = inputs
        b, c, h, w = y.shape

        x = self.conv1x1(x)

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class ParallelAtrousConv(nn.Module):
    def __init__(self, inc, ratio=[1, 2, 3]) -> None:
        super().__init__()

        self.conv1 = Conv(inc, inc, k=3, d=ratio[0])
        self.conv2 = Conv(inc, inc // 2, k=3, d=ratio[1])
        self.conv3 = Conv(inc, inc // 2, k=3, d=ratio[2])
        self.conv4 = Conv(inc * 2, inc, k=1)

    def forward(self, x):
        return self.conv4(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))


class CSP_PAC(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = ParallelAtrousConv(c_)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

