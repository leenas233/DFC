import torch
import torch.nn as nn
import torch.nn.functional as F


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f K' % (num_params / 1e3))


def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


def pixel_shuffle_inv(x, scale_factor):
    num, ch, height, width = x.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    x = x.reshape(
        (num, ch, new_height, scale_factor, new_width, scale_factor))
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    x = x.permute((0, 1, 3, 5, 2, 4))
    x = x.reshape((num, new_ch, new_height, new_width))
    return x


############### Basic Convolutional Layers ###############
class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ActConv(nn.Module):
    """ Conv. with activation. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.act = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out


############### MuLUT Blocks ###############
class MuLUTUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    def __init__(self, mode, nf, upscale=1, out_c=1, dense=True):
        super(MuLUTUnit, self).__init__()
        self.act = nn.ReLU()
        self.upscale = upscale

        if mode == '2x2':
            self.conv1 = Conv(1, nf, 2)
        elif mode == '2x2d':
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == '2x2d3':
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '1x4':
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            self.conv6 = Conv(nf * 5, 1 * upscale * upscale, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            self.conv6 = Conv(nf, upscale * upscale, 1)
        if self.upscale > 1:
            self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = torch.tanh(self.conv6(x))
        x = self.conv6(x)
        if self.upscale > 1:
            x = self.pixel_shuffle(x)
        return x

class SRUnit(nn.Module):
    def __init__(self):
        super(SRUnit, self).__init__()
        pass

class SRUNet(nn.Module):
    """ Wrapper of a generalized (spatial-wise) MuLUT block.
        By specifying the unfolding patch size and pixel indices,
        arbitrary sampling pattern can be implemented.
    """

    def __init__(self, mode, nf=64, out_c=None, dense=True, stride=1):
        super(SRUNet, self).__init__()
        self.mode = mode

        # if 'x1' in mode:
        #     assert out_c is None
        if mode == 'Sx1':
            self.model = SRUNetUnit('2x2', nf, out_c=1, dense=dense)
            self.K = 2
        elif mode == 'SxN':
            self.model = SRUNetUnit('2x2', nf, out_c=out_c, dense=dense)
            self.K = 2
        elif mode == 'Dx1':
            self.model = SRUNetUnit('2x2d', nf, out_c=1, dense=dense)
            self.K = 3
        elif mode == 'DxN':
            self.model = SRUNetUnit('2x2d', nf, out_c=out_c, dense=dense)
            self.K = 3
        elif mode == 'Yx1':
            self.model = SRUNetUnit('1x4', nf, out_c=1, dense=dense)
            self.K = 3
        elif mode == 'YxN':
            self.model = SRUNetUnit('1x4', nf, out_c=out_c, dense=dense)
            self.K = 3
        elif mode == 'Ex1':
            self.model = SRUNetUnit('2x2d3', nf, out_c=1, dense=dense)
            self.K = 4
        elif mode == 'ExN':
            self.model = SRUNetUnit('2x2d3', nf, out_c=out_c, dense=dense)
            self.K = 4
        elif mode in ['Ox1', 'Hx1']:
            self.model = SRUNetUnit('1x4', nf, out_c=1, dense=dense)
            self.K = 4
        elif mode == ['OxN', 'HxN']:
            self.model = SRUNetUnit('1x4', nf, out_c=out_c, dense=dense)
            self.K = 4
        else:
            raise AttributeError
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K, stride=self.stride)  # B,C*K*K,L
        x = x.reshape(B, C, self.K * self.K,
                      ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1), self.K,
                      self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,1,K,K

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        elif 'H' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                           x[:, :, 2, 3], x[:, :, 3, 2]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        elif 'O' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                           x[:, :, 1, 3], x[:, :, 3, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)  # B*C*L,1,1,1 or B*C*L,1,upscale,upscale
        x = x.reshape(B, C, ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1), -1)  # B,C,L,out_c
        x = x.permute((0, 1, 3, 2))  # B,C,out_c,L
        x = x.reshape(B, -1, ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1))  # B,C*out_c,L
        x = F.fold(x, ((H - self.K) // self.stride + 1, (W - self.K) // self.stride + 1), (1, 1), stride=(1, 1))

        return x


class SRUNetUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    def __init__(self, mode, nf, out_c=1, dense=True):
        super(SRUNetUnit, self).__init__()
        self.act = nn.ReLU()

        if mode == '2x2':
            self.conv1 = Conv(1, nf, 2)
        elif mode == '2x2d':
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == '2x2d3':
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '1x4':
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            self.conv6 = Conv(nf * 5, out_c, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            self.conv6 = Conv(nf, out_c, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class SRLUTUnit(nn.Module):
    def __init__(self, in_c, out_c, first_kernel_size, nf=64, dense=True):
        super(SRLUTUnit, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_c, nf, first_kernel_size)

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            self.conv6 = Conv(nf * 5, out_c, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            self.conv6 = Conv(nf, out_c, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = torch.tanh(self.conv6(x))
        x = self.conv6(x)
        return x


class IMDB(nn.Module):
    def __init__(self):
        super(IMDB, self).__init__()
        # self.spatial_conv1 = Conv(1, 4, 2)
        self.spatial_conv1 = SRLUTUnit(2, 8, 2, 32)

        self.channel_conv2 = Conv(6, 1, 1)
        # self.spatial_conv2 = Conv(1, 4, 2)
        self.spatial_conv2 = SRLUTUnit(1, 8, 2, 32)

        self.channel_conv3 = Conv(6, 1, 1)
        # self.spatial_conv3 = Conv(1, 4, 2)
        self.spatial_conv3 = SRLUTUnit(1, 8, 2, 32)

        self.channel_conv4 = Conv(6, 1, 1)
        # self.spatial_conv4 = Conv(1, 1, 2)  # the last one outputs 1 channel
        self.spatial_conv4 = SRLUTUnit(1, 2, 2, 32)

        self.channel_conv_tail = Conv(8, 2, 1)
        self.multi_distillation = [(None, self.spatial_conv1), (self.channel_conv2, self.spatial_conv2),
                                   (self.channel_conv3, self.spatial_conv3), (self.channel_conv4, self.spatial_conv4)]

    def forward(self, x):
        x_origin = x
        refined_list = []
        for i, modules in enumerate(self.multi_distillation):
            pred = 0
            if i != 0:
                x = modules[0](x)
                x = torch.tanh(x)
                x = round_func(torch.clamp(x, 0, 1) * 16.0) / 16.0

            for r in [0, 1, 2, 3]:
                pred += torch.tanh(
                    torch.rot90(modules[1](F.pad(torch.rot90(x, r, [2, 3]), (0, 1, 0, 1), mode='replicate')),
                                (4 - r) % 4, [2, 3]))
            avg_factor, bias, norm = 4, 127, 255.0
            x = round_func(torch.clamp((pred / avg_factor), 0, 1) * 16.0) / 16.0
            refined_list.append(x[:, 0:2, :, :])
            x = x[:, 2:, :, :]
        x = torch.tanh(self.channel_conv_tail(torch.cat(refined_list, dim=1)))
        x = round_func(torch.clamp(x + x_origin, 0, 1) * 16.0) / 16.0
        return x


class RC_Module(nn.Module):
    def __init__(self, in_channels, out_channels, out_dim, mlp_field=7):
        super(RC_Module, self).__init__()
        self.mlp_field = mlp_field
        for i in range(mlp_field * mlp_field):
            setattr(self, 'linear{}'.format(i + 1), nn.Linear(in_channels, out_channels))
            setattr(self, 'out{}'.format(i + 1), nn.Linear(out_channels, out_dim))

    def forward(self, x):
        x_kv = {}
        # print('x', x.size())
        for i in range(self.mlp_field):
            for j in range(self.mlp_field):
                num = i * self.mlp_field + j + 1
                module1 = getattr(self, 'linear{}'.format(num))
                x_kv[str(num)] = module1(x[:, i, j, :]).unsqueeze(1) # (N,1,out_channels)
        x_list = []
        for i in range(self.mlp_field * self.mlp_field):
            module = getattr(self, 'out{}'.format(i + 1))
            x_list.append(module(x_kv[str(i + 1)])) # (N,1,out_dim)

        out = torch.cat(x_list, dim=1)

        out = out.mean(1) # (N, out_channels)

        out = out.unsqueeze(-1).unsqueeze(-1)  # (N, out_channels, 1, 1)

        out = torch.tanh(out)
        out = round_func(out * 127)
        bias, norm = 127, 255.0
        out = round_func(torch.clamp(out + bias, 0, 255)) / norm
        # print(out.min(),out.max())
        return out

class RC_MuLUTConvUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """
    def __init__(self, mode, nf, out_c=1, dense=True, stage=1):
        super(RC_MuLUTConvUnit, self).__init__()
        self.act = nn.ReLU()

        self.conv_naive = Conv(1, nf, 2)
        self.mode = mode
        self.stage = stage

        if mode == '2x2':
            if self.stage == 1:
                self.conv1 = RC_Module(1, nf, 1, mlp_field=5)
            else:
                self.conv1 = RC_Module(1, nf, 1, mlp_field=5)
            self.s_conv = Conv(1, nf, 1)
            # self.conv1 = Conv_test(1, nf)
        elif mode == '2x2d':
            # self.conv1 = Conv(1, nf, 2, dilation=2)
            if self.stage == 1:
                self.conv1 = RC_Module(1, nf, 1, mlp_field=7)
            else:
                self.conv1 = RC_Module(1, nf, 1, mlp_field=7)
            self.d_conv = Conv(1, nf, 1)
        elif mode == '2x2d3':
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '2x2d4':
            self.conv1 = Conv(1, nf, 2, dilation=4)
        elif mode == '1x4':
            # self.conv1 = Conv(1, nf, (1, 4))
            if self.stage == 1:
                self.conv1 = RC_Module(1, nf, 1, mlp_field=3)
            else:
                self.conv1 = RC_Module(1, nf, 1, mlp_field=3)
            self.y_conv = Conv(1, nf, 1)
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            self.conv6 = Conv(nf * 5, out_c, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            self.conv6 = Conv(nf, out_c, 1)

    def forward(self, x, r_H, r_W, x_dense, x_3x3, x_7x7, transfer = False):
        B, C, H, W = x_dense.shape
        # print(x_dense.shape)
        x_dense = x_dense.reshape(-1, 1, H, W)
        if self.mode == '2x2':
            if not transfer:
                x = x
                x = self.conv1(x)
                # x = torch.tanh(self.conv1(x))
                # print(x.shape)
                if self.stage == 1:
                    x = self.s_conv(x)
                else:
                    x = x.reshape(-1, 1, H, W)
                    # x += x_dense
                    x = F.pad(x, [0, 1, 0, 1], mode='replicate')
                    x = F.unfold(x, 2)
                    x = x.view(B, C, 2 * 2, H * W)
                    x = x.permute((0, 1, 3, 2))
                    x = x.reshape(B * C * H * W, 2, 2)
                    x = x.unsqueeze(1)
                    x = self.act(self.conv_naive(x))
            else:
                # x = torch.tanh(x)
                if self.stage == 1:
                    x  =self.s_conv(x)
                else:
                    x = self.act(self.conv_naive(x))

        elif self.mode == '2x2d':
            if not transfer:
                x = x_7x7
                x = self.conv1(x)
                # x = torch.tanh(self.conv1(x))
                if self.stage == 1:
                    x = self.d_conv(x)
                else:
                    x = x.reshape(-1, 1, H, W)
                    # x += x_dense
                    x = F.pad(x, [0, 1, 0, 1], mode='replicate')
                    x = F.unfold(x, 2)
                    x = x.view(B, C, 2 * 2, H * W)
                    x = x.permute((0, 1, 3, 2))
                    x = x.reshape(B * C * H * W, 2, 2)
                    x = x.unsqueeze(1)
                    x = self.act(self.conv_naive(x))
            else:
                # x = torch.tanh(x)
                if self.stage==1:
                    x = self.d_conv(x)
                else:
                    x = self.act(self.conv_naive(x))
        elif self.mode == '1x4':
            if not transfer:
                x = x_3x3
                x = self.conv1(x)
                # x = torch.tanh(self.conv1(x))
                if self.stage == 1:
                    x = self.y_conv(x)
                else:
                    x = x.reshape(-1, 1, H, W)
                    # x += x_dense
                    x = F.pad(x, [0, 1, 0, 1], mode='replicate')
                    x = F.unfold(x, 2)
                    x = x.view(B, C, 2 * 2, H * W)
                    x = x.permute((0, 1, 3, 2))
                    x = x.reshape(B * C * H * W, 2, 2)
                    x = x.unsqueeze(1)
                    x = self.act(self.conv_naive(x))
            else:
                # x = torch.tanh(x)
                if self.stage==1:
                    x = self.y_conv(x)
                else:
                    x = self.act(self.conv_naive(x))

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x

class RC_MuLUTConv(nn.Module):
    """ Wrapper of a generalized (spatial-wise) MuLUT block.
        By specifying the unfolding patch size and pixel indices,
        arbitrary sampling pattern can be implemented.
    """

    def __init__(self, mode, nf=64, out_c=None, dense=True,stride=1):
        super(RC_MuLUTConv, self).__init__()
        self.mode = mode

        if mode == 'Sx1':
            self.model = RC_MuLUTConvUnit('2x2', nf, out_c=out_c, dense=dense, stage=1)
            self.K = 5
        elif mode == 'SxN':
            self.model = RC_MuLUTConvUnit('2x2', nf, out_c=out_c, dense=dense, stage=2)
            self.K = 5
        elif mode == 'Dx1':
            self.model = RC_MuLUTConvUnit('2x2d', nf, out_c=out_c, dense=dense, stage=1)
            self.K = 5
        elif mode == 'DxN':
            self.model = RC_MuLUTConvUnit('2x2d', nf, out_c=out_c, dense=dense, stage=2)
            self.K = 5
        elif mode == 'Yx1':
            self.model = RC_MuLUTConvUnit('1x4', nf, out_c=out_c, dense=dense, stage=1)
            self.K = 5
        elif mode == 'YxN':
            self.model = RC_MuLUTConvUnit('1x4', nf, out_c=out_c, dense=dense, stage=2)
            self.K = 5
        else:
            raise AttributeError
        self.stride = stride

    def forward(self, x,transfer=False):
        if 'H' in self.mode:
            channel = x.size(1)
            x = x.reshape(-1, 1, x.size(2), x.size(3))
            x = self.model(x)
            x = x.reshape(-1, channel, x.size(2), x.size(3))
        elif self.mode == 'Connect':
            x = self.model(x)
        else:
            if not transfer:
                B, C, H, W = x.shape
                x_dense = x[:, :, :-4, :-4]
                x_7x7 = F.pad(x, [2, 0, 2, 0], mode='replicate')
                B7, C7, H7, W7 = x_7x7.shape
                x_7x7 = F.unfold(x_7x7, 7)
                x_3x3 = x[:, :, :-2, :-2]
                B3, C3, H3, W3 = x_3x3.shape
                x_3x3 = F.unfold(x_3x3, 3)

                x_3x3 = x_3x3.view(B3, C3, 9, (H3 - 2) * (W3 - 2))
                x_3x3 = x_3x3.permute((0, 1, 3, 2))
                x_3x3 = x_3x3.reshape(B3 * C3 * (H3 - 2) * (W3 - 2), 3, 3)
                x_3x3 = x_3x3.unsqueeze(-1)

                x_7x7 = x_7x7.view(B7, C7, 49, (H7 - 6) * (W7 - 6))
                x_7x7 = x_7x7.permute((0, 1, 3, 2))
                x_7x7 = x_7x7.reshape(B7 * C7 * (H7 - 6) * (W7 - 6), 7, 7)
                x_7x7 = x_7x7.unsqueeze(-1)

                B, C, H, W = x.shape
                x = F.unfold(x, self.K, stride=self.stride)  # B,C*K*K,L
                x = x.reshape(B, C, self.K * self.K,
                              ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1))  # B,C,K*K,L
                x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
                x = x.reshape(B * C * ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1), self.K,
                              self.K)  # B*C*L,K,K
                x = x.unsqueeze(-1)
                # print(x.shape)

                x = self.model(x, 0, 0, x_dense, x_3x3, x_7x7)  # B*C*L,K,K
                # print(x.shape)
                x = x.reshape(B, C, ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1),
                              -1)  # B,C,L,out_c
                x = x.permute((0, 1, 3, 2))  # B,C,out_c,L
                x = x.reshape(B, -1,
                              ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1))  # B,C*out_c,L
                x = F.fold(x, ((H - self.K) // self.stride + 1, (W - self.K) // self.stride + 1), (1, 1), stride=(1, 1))
            else:
                x = self.model(x,0,0,x,x,x,transfer)

        return x


class MuLUTConvUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    def __init__(self, mode, nf, out_c=1, dense=True):
        super(MuLUTConvUnit, self).__init__()
        self.act = nn.ReLU()

        if mode == '2x2':
            self.conv1 = Conv(1, nf, 2)
        elif mode == '2x2d':
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == '2x2d3':
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '1x4':
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            self.conv6 = Conv(nf * 5, out_c, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            self.conv6 = Conv(nf, out_c, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class MuLUTConv(nn.Module):
    """ Wrapper of a generalized (spatial-wise) MuLUT block.
        By specifying the unfolding patch size and pixel indices,
        arbitrary sampling pattern can be implemented.
    """

    def __init__(self, mode, nf=64, out_c=None, dense=True, stride=1):
        super(MuLUTConv, self).__init__()
        self.mode = mode

        # if 'x1' in mode:
        #     assert out_c is None
        if mode == 'Sx1':
            self.model = MuLUTConvUnit('2x2', nf, out_c=1, dense=dense)
            self.K = 2
        elif mode == 'SxN':
            self.model = MuLUTConvUnit('2x2', nf, out_c=out_c, dense=dense)
            self.K = 2
        elif mode == 'Dx1':
            self.model = MuLUTConvUnit('2x2d', nf, out_c=1, dense=dense)
            self.K = 3
        elif mode == 'DxN':
            self.model = MuLUTConvUnit('2x2d', nf, out_c=out_c, dense=dense)
            self.K = 3
        elif mode == 'Yx1':
            self.model = MuLUTConvUnit('1x4', nf, out_c=1, dense=dense)
            self.K = 3
        elif mode == 'YxN':
            self.model = MuLUTConvUnit('1x4', nf, out_c=out_c, dense=dense)
            self.K = 3
        elif mode == 'Ex1':
            self.model = MuLUTConvUnit('2x2d3', nf, out_c=1, dense=dense)
            self.K = 4
        elif mode == 'ExN':
            self.model = MuLUTConvUnit('2x2d3', nf, out_c=out_c, dense=dense)
            self.K = 4
        elif mode in ['Ox1', 'Hx1']:
            self.model = MuLUTConvUnit('1x4', nf, out_c=1, dense=dense)
            self.K = 4
        elif mode == ['OxN', 'HxN']:
            self.model = MuLUTConvUnit('1x4', nf, out_c=out_c, dense=dense)
            self.K = 4
        else:
            raise AttributeError
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K, stride=self.stride)  # B,C*K*K,L
        x = x.reshape(B, C, self.K * self.K,
                      ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1), self.K,
                      self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,1,K,K

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        elif 'H' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                           x[:, :, 2, 3], x[:, :, 3, 2]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        elif 'O' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                           x[:, :, 1, 3], x[:, :, 3, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)  # B*C*L,1,1,1 or B*C*L,1,upscale,upscale
        x = x.reshape(B, C, ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1), -1)  # B,C,L,out_c
        x = x.permute((0, 1, 3, 2))  # B,C,out_c,L
        x = x.reshape(B, -1, ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1))  # B,C*out_c,L
        x = F.fold(x, ((H - self.K) // self.stride + 1, (W - self.K) // self.stride + 1), (1, 1), stride=(1, 1))

        return x

class MuLUTcUnit(nn.Module):
    """ Channel-wise MuLUT block [RGB(3D) to RGB(3D)]. """

    def __init__(self, in_c, out_c, mode, nf):
        super(MuLUTcUnit, self).__init__()
        self.act = nn.ReLU()

        if mode == '1x1':
            self.conv1 = Conv(in_c, nf, 1)
        else:
            raise AttributeError

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        self.conv6 = Conv(nf * 5, out_c, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class TMUnit(nn.Module):
    """Tone Mapping Unit RGB(3D) to RGB(3D)"""
    def __init__(self, mode, nf):
        super(TMUnit, self).__init__()
        self.act = nn.ReLU()

        if mode == '1x1':
            self.conv1 = Conv(3, nf, 1)
        else:
            raise AttributeError

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        self.conv6 = Conv(nf * 5, 3, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

############### Image Super-Resolution ###############
class SRNet(nn.Module):
    """ Wrapper of a generalized (spatial-wise) MuLUT block. 
        By specifying the unfolding patch size and pixel indices,
        arbitrary sampling pattern can be implemented.
    """

    def __init__(self, mode, nf=64, upscale=None, dense=True):
        super(SRNet, self).__init__()
        self.mode = mode

        if 'x1' in mode:
            assert upscale is None
        if mode == 'Sx1':
            self.model = MuLUTUnit('2x2', nf, upscale=1, dense=dense)
            self.K = 2
            self.S = 1
        elif mode == 'SxN':
            self.model = MuLUTUnit('2x2', nf, upscale=upscale, dense=dense)
            self.K = 2
            self.S = upscale
        elif mode == 'Dx1':
            self.model = MuLUTUnit('2x2d', nf, upscale=1, dense=dense)
            self.K = 3
            self.S = 1
        elif mode == 'DxN':
            self.model = MuLUTUnit('2x2d', nf, upscale=upscale, dense=dense)
            self.K = 3
            self.S = upscale
        elif mode == 'Yx1':
            self.model = MuLUTUnit('1x4', nf, upscale=1, dense=dense)
            self.K = 3
            self.S = 1
        elif mode == 'YxN':
            self.model = MuLUTUnit('1x4', nf, upscale=upscale, dense=dense)
            self.K = 3
            self.S = upscale
        elif mode == 'Ex1':
            self.model = MuLUTUnit('2x2d3', nf, upscale=1, dense=dense)
            self.K = 4
            self.S = 1
        elif mode == 'ExN':
            self.model = MuLUTUnit('2x2d3', nf, upscale=upscale, dense=dense)
            self.K = 4
            self.S = upscale
        elif mode in ['Ox1', 'Hx1']:
            self.model = MuLUTUnit('1x4', nf, upscale=1, dense=dense)
            self.K = 4
            self.S = 1
        elif mode == ['OxN', 'HxN']:
            self.model = MuLUTUnit('1x4', nf, upscale=upscale, dense=dense)
            self.K = 4
            self.S = upscale
        else:
            raise AttributeError
        self.P = self.K - 1

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        elif 'H' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                           x[:, :, 2, 3], x[:, :, 3, 2]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        elif 'O' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                           x[:, :, 1, 3], x[:, :, 3, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)  # B*C*L,K,K
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)
        return x


############### Grayscale Denoising, Deblocking, Color Image Denosing ###############
class DNNet(nn.Module):
    """ Wrapper of basic MuLUT block without upsampling. """

    def __init__(self, mode, nf=64, dense=True):
        super(DNNet, self).__init__()
        self.mode = mode

        self.S = 1
        if mode == 'Sx1':
            self.model = MuLUTUnit('2x2', nf, dense=dense)
            self.K = 2
        elif mode == 'Dx1':
            self.model = MuLUTUnit('2x2d', nf, dense=dense)
            self.K = 3
        elif mode == 'Yx1':
            self.model = MuLUTUnit('1x4', nf, dense=dense)
            self.K = 3
        else:
            raise AttributeError
        self.P = self.K - 1

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)  # B*C*L,K,K
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)
        return x


############### Image Demosaicking ###############
class DMNet(nn.Module):
    """ Wrapper of the first stage of MuLUT network for demosaicking. 4D(RGGB) bayer patter to (4*3)RGB"""

    def __init__(self, mode, nf=64, dense=False):
        super(DMNet, self).__init__()
        self.mode = mode

        if mode == 'SxN':
            self.model = MuLUTUnit('2x2', nf, upscale=2, out_c=3, dense=dense)
            self.K = 2
            self.C = 3
        else:
            raise AttributeError
        self.P = 0  # no need to add padding self.K - 1
        self.S = 2  # upscale=2, stride=2

    def forward(self, x):
        B, C, H, W = x.shape
        # bayer pattern, stride = 2
        x = F.unfold(x, self.K, stride=2)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H // 2) * (W // 2))  # stride = 2
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H // 2) * (W // 2),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        # print("in", torch.round(x[0, 0]*255))

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)  # B*C*L,out_C,S,S
        # self.C along with feat scale
        x = x.reshape(B, C, (H // 2) * (W // 2), -1)  # B,C,L,out_C*S*S
        x = x.permute((0, 1, 3, 2))  # B,C,outC*S*S,L
        x = x.reshape(B, -1, (H // 2) * (W // 2))  # B,C*out_C*S*S,L
        x = F.fold(x, ((H // 2) * self.S, (W // 2) * self.S),
                   self.S, stride=self.S)
        return x


if __name__ == '__main__':

    # x = torch.range(0, 63).view((1, 16, 4))
    # print(x)
    # x = F.fold(x, (4,4),2 ,stride=2 )
    # print(x)
    # print(x.shape)
    model = RC_MuLUTConv(mode='SxN', nf=64, out_c=1, dense=True)
    x = torch.rand((4, 3, 24, 24))
    y = model(x)
    print(y.shape)

    # x = torch.range(0, 35).view((1, 1, 6, 6))
    # B, C, H, W = x.shape
    # K = 3
    # S = 4
    # stride = 1
    #
    # pixel_shuffle = nn.PixelShuffle(S)
    # model = Conv(in_channels=1, out_channels=S ** 2, kernel_size=K)
    # y = model(x)
    # print(y.shape)
    #
    # x = F.unfold(x, K, stride=stride)  # B,C*K*K,L
    # x = x.reshape(B, C, K * K, ((H - K) // stride + 1) * ((W - K) // stride + 1))  # B,C,K*K,L
    # x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
    # x = x.reshape(B * C * ((H - K) // stride + 1) * ((W - K) // stride + 1), K, K)  # B*C*L,K,K
    # x = x.unsqueeze(1)  # B*C*L,1,K,K
    #
    # x = model(x)  # B*C*L,out_c,1,1
    # x = x.reshape(B, C, ((H - K) // stride + 1) * ((W - K) // stride + 1), -1)  # B,C,L,out_c
    # x = x.permute((0, 1, 3, 2))  # B,C,out_c,L
    # x = x.reshape(B, -1, ((H - K) // stride + 1) * ((W - K) // stride + 1))  # B,C*out_c,L
    # x = F.fold(x, ((H - K) // stride + 1, (W - K) // stride + 1), (1, 1), stride=(1, 1))
    # print(x.shape)
    #
    # print(y - x)
    # print(pixel_shuffle(y) - pixel_shuffle(x))
