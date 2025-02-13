import torch
from models.spp_layer import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn

# Initially from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
# Updated at https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/convlstm/convlstm.py
class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # does 4 represent years
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        #         return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
        #                 Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())
        return (
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width).data,
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width).data,
        )


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_size=(21, 21),
        input_dim=5,
        hidden_dim=(16, 32),
        kernel_size=((3, 3),),
        num_layers=2,
        bias=True,
        return_all_layers=False,
    ):

        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        if not len(kernel_size) == num_layers:
            kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        if not len(hidden_dim) == num_layers:
            hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        self.height, self.width = input_size
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_size=self.input_size,
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        Returns
        -------
        last_state_list, layer_output
        """

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        #         layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(2)  # Number of years worth of dynamic tensors
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, :, t, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=2)
            cur_layer_input = layer_output

            #             returns all [layer_1(h_1,h_2,...h_t),layer_2(h_1,h_2,...h_t),layer_3(h_1,h_2,...h_t)...]
            #             dont need it if not tracking individual loss
            #             layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            #             layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        #          return layer_output_list, last_state_list
        return last_state_list[0]

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param[0]] * num_layers
        return param


# Adapted from https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/sequenceencoder.py
class LSTMSequentialEncoder(torch.nn.Module):
    def __init__(
        self,
        height=21,
        width=21,
        input_dim=(2, 5),
        hidden_dim=(16, 16, 64, 8),
        kernel_size=((3, 3), (1, 3, 3), (3, 3), (3, 3)),
        levels=(13,),
        dropout=0.2,
        bias=True,
    ):
        super(LSTMSequentialEncoder, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
        )

        self.inconv = nn.Sequential(
            torch.nn.Conv3d(input_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(hidden_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(hidden_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
        )

        cell_input_size = height - 3 * (kernel_size[1][-1] - 1)
        self.cell = ConvLSTMCell(
            input_size=(cell_input_size, cell_input_size),
            input_dim=hidden_dim[1],
            hidden_dim=hidden_dim[2],
            kernel_size=kernel_size[2],
            bias=bias,
        )

        self.final = nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[2] + hidden_dim[0], hidden_dim[3], kernel_size[3]
            ),
            torch.nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[3]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[3] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):
        # Split into static (z) and dynamic tensors (x) to be fed into different branches
        z, x = data
        # 2D convolutions over the static tensor
        z = self.conv.forward(z)
        #
        x = self.inconv.forward(x)
        # bands, channels, time, height, width
        b, c, t, h, w = x.shape
        hidden = torch.zeros((b, self.hidden_dim[2], h, w))
        state = torch.zeros((b, self.hidden_dim[2], h, w))
        for iter in range(t):
            hidden, state = self.cell.forward(x[:, :, iter, :, :], (hidden, state))
        x = hidden
        # Join dynamic and static branches
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        return x.flatten()


class DeepLSTMSequentialEncoder(torch.nn.Module):
    """
    DeepLSTMSequentialEncoder with the option to add multiple ConvLSTM layers
    """

    def __init__(
        self,
        height=21,
        width=21,
        input_dim=(2, 5),
        hidden_dim=(16, 16, (16, 16), 8),
        kernel_size=((3, 3), (1, 3, 3), ((3, 3),), (3, 3)),
        num_layers=2,
        levels=(13,),
        dropout=0.2,
        bias=True,
        return_all_layers=False,
    ):
        super(DeepLSTMSequentialEncoder, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[0]),
        )

        self.inconv = nn.Sequential(
            torch.nn.Conv3d(input_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(hidden_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(hidden_dim[1], hidden_dim[1], kernel_size[1]),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim[1]),
        )

        cell_input_size = height - 3 * (kernel_size[1][1] - 1)

        self.cell = ConvLSTM(
            input_size=(cell_input_size, cell_input_size),
            input_dim=hidden_dim[1],
            hidden_dim=hidden_dim[2],
            kernel_size=kernel_size[2],
            num_layers=num_layers,
            bias=bias,
            return_all_layers=return_all_layers,
        )

        self.final = nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[2][-1] + hidden_dim[0], hidden_dim[3], kernel_size[3]
            ),
            torch.nn.ReLU(),
            nn.BatchNorm2d(hidden_dim[3]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[3] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        z = self.conv.forward(z)
        x = self.inconv.forward(x)
        hidden, state = self.cell.forward(x)
        x = hidden
        # Join dynamic and static branches
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)
        return x.flatten()


class Conv_3D(torch.nn.Module):
    """
    Making deforestation predictions with 3D convolutions (space + time)
    """

    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
        # start_year=14,
        # end_year=17,
    ):
        super(Conv_3D, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv_2D = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            torch.nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
        )

        self.conv_3D = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=input_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=kernel_size[1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
            # This second 3d conv layer is troublesome
            # Kernel size needs to be tweaked by year
            torch.nn.Conv3d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=(
                    kernel_size[3][0],# + (end_year - start_year - 2),
                    kernel_size[3][1],
                    kernel_size[3][2],
                ),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
        )

        self.pool3d = nn.AdaptiveAvgPool3d((1, None, None))

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[0] + hidden_dim[1], hidden_dim[2], kernel_size[2]
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        x = x.squeeze(dim=2)
        # print("x shape post squeeze:", x.shape)
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        return x.flatten()

# Kernel size needs to be different depending on how many years of data are being handled
# This model is for an even number of training years (e.g. start_date = 14, end_date = 17)
class Conv_3Deven(torch.nn.Module):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
    ):
        super(Conv_3Deven, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv_2D = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0], ), # padding=(1,1)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            torch.nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0], ), # padding=(1,1)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
        )

        self.conv_3D = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=input_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=kernel_size[1],
                # padding=(1, 0, 0), # to preserve depth for few-year models
                # padding=(0,1,1), # including preserves (W,H), else drops by 2, but conv2D does as well; for kernel of size 3 (5 will drop 2)
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[1],
                # DEPENDING ON NUMBER OF YEARS, NEED TO SWITCH BETWEEN KERNEL SIZE #
                kernel_size=(
                    kernel_size[1][0],# + 1, # + 1 for even years; + 0 for odd years (TODO adaptive pooling makes this irrelevant now?)
                    kernel_size[1][1],
                    kernel_size[1][2],
                ),
                # padding=(1, 0, 0), # to preserve depth for few-year models
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
        )

        self.pool3d = nn.AdaptiveAvgPool3d((1, None, None))

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[0] + hidden_dim[1], hidden_dim[2], kernel_size[2]
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        # print("z shape start:", z.shape)
        # print("x shape start:", x.shape)
        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        x = self.pool3d(x)  # Reduce depth dimension to 1
        # print("z shape post conv2d:", z.shape)
        # print("x shape post conv3d:", x.shape)
        x = x.squeeze(dim=2)
        # print("x shape post squeeze:", x.shape)
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        return x.flatten()


# Kernel size needs to be different depending on how many years of data are being handled
# This model is for an odd number of training years (e.g. start_date = 14, end_date = 16)
class Conv_3Dodd(torch.nn.Module):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
    ):
        super(Conv_3Dodd, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv_2D = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0], 
                            # padding=(1,1) # used to preserve size (W,H), else drops by 4 (must match conv3D below, or use adaptivePool2D)
                            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            torch.nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]
                            #, padding=(1,1)
                            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
        )

        self.conv_3D = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=input_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=kernel_size[1],
                padding=(1, 0, 0), # to preserve depth for few-year models
                # padding=(0,1,1), # including preserves (W,H), else drops by 2, but conv2D does as well; for kernel of size 3 (5 will drop 2)
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[1],
                # DEPENDING ON NUMBER OF YEARS, NEED TO SWITCH BETWEEN KERNEL SIZE #
                # This one for odd num of years#
                # kernel_size=kernel_size[1],
                # This one for even num of years#
                kernel_size=(
                    kernel_size[3][0],# + 2, # Plus number of years from 4??? TODO # boosted by one for even number of years
                    kernel_size[3][1],
                    kernel_size[3][2],
                ),
                padding=(1, 0, 0), # to preserve depth for few-year models
                # padding=(0,1,1), # including preserves (W,H), else drops by 2, but conv2D does as well; for kernel of size 3 (5 will drop 2)
                # ),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
        )

        self.pool3d = nn.AdaptiveAvgPool3d((1, None, None))

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[0] + hidden_dim[1], hidden_dim[2], kernel_size[2]
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        # print("z shape start:", z.shape)
        # print("x shape start:", x.shape)
        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        # print("x shape post conv3d:", x.shape)
        x = self.pool3d(x)  # Reduce depth dimension to 1
        # print("x shape post pool:", x.shape)
        # print("z shape post conv2d:", z.shape)
        x = x.squeeze(dim=2)
        # print("x shape post squeeze:", x.shape)
        x = torch.cat((x, z), dim=1)  # Problem with dimensions here
        # print("x shape post cat:", x.shape)
        x = self.final.forward(x)
        # print("x shape post final:", x.shape)
        x = spp_layer(x, self.levels)
        # print("x shape post spp:", x.shape)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        return x.flatten()

class Conv_3DUNet(nn.Module):
    def __init__(self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
        base_filters=64,
    ):
        super(Conv_3DUNet, self).__init__()
        in_channels_2D, in_channels_3D = input_dim

        # 2D Encoder
        self.encoder_2D = nn.ModuleList([
            self._conv_block(in_channels_2D, base_filters),
            self._conv_block(base_filters, base_filters * 2),
            self._conv_block(base_filters * 2, base_filters * 4)
        ])
        
        # 3D Encoder
        self.encoder_3D = nn.ModuleList([
            self._conv_block_3D(in_channels_3D, base_filters // 2),
            self._conv_block_3D(base_filters // 2, base_filters),
            self._conv_block_3D(base_filters, base_filters * 2)
        ])
        self.enc_pool_3D = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        
        # Decoder
        self.decoder = nn.ModuleList([
            self._conv_block(base_filters * 6, base_filters * 3),
            self._conv_block(base_filters * 3, base_filters)
        ])
        
        skip_connection_channels = 192
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(base_filters * 6 + skip_connection_channels, base_filters * 2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(base_filters * 3, base_filters, kernel_size=2, stride=2)
        ])
        
        self.pool_2D = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_3D = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.sigmoid = nn.Sigmoid()

        # Global pooling and final layers
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)  # Collapse spatial dimensions to [batch, channels, 1, 1]
        self.fc = nn.Linear(base_filters, 1)  # Map features to a single value
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_channels, out_channels):
        """Creates a 2D convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def _conv_block_3D(self, in_channels, out_channels):
        """Creates a 3D convolutional block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, data, sigmoid=None):
        x_2D, x_3D = data
        # 2D Encoder
        enc_2D_features = []
        for enc in self.encoder_2D:
            x_2D = enc(x_2D)
            enc_2D_features.append(x_2D)
            x_2D = self.pool_2D(x_2D)
        
        # 3D Encoder
        enc_3D_features = []
        for enc in self.encoder_3D:
            x_3D = enc(x_3D)
            enc_3D_features.append(x_3D)
            x_3D = self.enc_pool_3D(x_3D)
        
        # Flatten 3D features into 2D
        x_3D = enc_3D_features[-1].view(
            enc_3D_features[-1].size(0), -1, enc_3D_features[-1].size(3), enc_3D_features[-1].size(4)
        )
        x_3D = self.pool_3D(x_3D)
        x_3D = torch.nn.functional.interpolate(x_3D, size=(x_2D.shape[2], x_2D.shape[3]), mode="bilinear", align_corners=True)

        # Feature Fusion
        bottleneck = torch.cat((x_2D, x_3D), dim=1)
        
        # Decoder
        for i, dec in enumerate(self.decoder):

            bottleneck = self.upconvs[i](bottleneck)
            
            enc_skip = torch.nn.functional.interpolate(
                enc_2D_features[-(i + 1)], size=(bottleneck.shape[2], bottleneck.shape[3]), mode="bilinear", align_corners=True
            )

            bottleneck = torch.cat((bottleneck, enc_skip), dim=1)  # Skip connection
            bottleneck = dec(bottleneck)

        # Global Pooling and dimension reduction
        pooled = self.global_pool(bottleneck)
        pooled = pooled.squeeze(-1).squeeze(-1)
        out = self.fc(pooled).squeeze(-1)
        return self.sigmoid(out)


# Updated to change how labels are handled - 2 labels instead of one
class Conv_3DoddT(torch.nn.Module):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
    ):
        super(Conv_3DoddT, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv_2D = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            torch.nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
        )

        self.conv_3D = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=input_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=kernel_size[1],
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[1],
                # DEPENDING ON NUMBER OF YEARS, NEED TO SWITCH BETWEEN KERNEL SIZE #
                # This one for odd num of years#
                kernel_size=kernel_size[1],
            ),
            # This one for even num of years#
            #                                        kernel_size = (kernel_size[1][0]+1,kernel_size[1][1],kernel_size[1][2])),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[0] + hidden_dim[1], hidden_dim[2], kernel_size[2]
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 2),
        )  # changed to 2

    #        self.sig = torch.nn.Sigmoid()

    #        self.sfmx = torch.nn.Softmax(dim=1)

    def forward(self, data, sigmoid=True):

        z, x = data

        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        x = x.squeeze(dim=2)
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)

        #        if sigmoid:
        #            x = self.sig(x)
        #        x = self.sfmx(x) # need this?

        return x



class Conv_3DoddOnly(torch.nn.Module):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
    ):
        super(Conv_3Dodd, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv_2D = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0], 
                            # padding=(1,1) # used to preserve size (W,H), else drops by 4 (must match conv3D below, or use adaptivePool2D)
                            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            torch.nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]
                            #, padding=(1,1)
                            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
        )

        self.conv_3D = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=input_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=kernel_size[1],
                # padding=(0,1,1), # including preserves (W,H), else drops by 2, but conv2D does as well; for kernel of size 3 (5 will drop 2)
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[1],
                # DEPENDING ON NUMBER OF YEARS, NEED TO SWITCH BETWEEN KERNEL SIZE #
                # This one for odd num of years#
                # kernel_size=kernel_size[1],
                # This one for even num of years#
                kernel_size=(
                    kernel_size[3][0],# + 2, # Plus number of years from 4??? TODO # boosted by one for even number of years
                    kernel_size[3][1],
                    kernel_size[3][2],
                ),
                # padding=(0,1,1), # including preserves (W,H), else drops by 2, but conv2D does as well; for kernel of size 3 (5 will drop 2)
                # ),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
        )

        self.pool3d = nn.AdaptiveAvgPool3d((1, None, None))

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[0] + hidden_dim[1], hidden_dim[2], kernel_size[2]
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        x = self.pool3d(x)
        x = x.squeeze(dim=2)
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        return x.flatten()

class Conv_3DevenOnly(torch.nn.Module):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
    ):
        super(Conv_3Dodd, self).__init__()

        self.levels = levels
        self.hidden_dim = hidden_dim

        self.conv_2D = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim[0], hidden_dim[0], kernel_size=kernel_size[0], 
                            # padding=(1,1) # used to preserve size (W,H), else drops by 4 (must match conv3D below, or use adaptivePool2D)
                            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
            torch.nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=kernel_size[0]
                            #, padding=(1,1)
                            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[0]),
        )

        self.conv_3D = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=input_dim[1],
                out_channels=hidden_dim[1],
                kernel_size=kernel_size[1],
                # padding=(0,1,1), # including preserves (W,H), else drops by 2, but conv2D does as well; for kernel of size 3 (5 will drop 2)
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
            torch.nn.Conv3d(
                in_channels=hidden_dim[1],
                out_channels=hidden_dim[1],
                # DEPENDING ON NUMBER OF YEARS, NEED TO SWITCH BETWEEN KERNEL SIZE #
                # This one for odd num of years
                # kernel_size=kernel_size[1],
                # This one for even num of years
                kernel_size=(
                    kernel_size[3][0],# + 2, # Plus number of years from 4??? TODO # boosted by one for even number of years
                    kernel_size[3][1],
                    kernel_size[3][2],
                ),
                # padding=(0,1,1), # including preserves (W,H), else drops by 2, but conv2D does as well; for kernel of size 3 (5 will drop 2)
                # ),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(hidden_dim[1]),
        )

        self.pool3d = nn.AdaptiveAvgPool3d((1, None, None))

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim[0] + hidden_dim[1], hidden_dim[2], kernel_size[2]
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
            torch.nn.Conv2d(hidden_dim[2], hidden_dim[2], kernel_size[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(hidden_dim[2]),
        )

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        self.ln = torch.nn.Sequential(
            torch.nn.Linear(ln_in, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(self, data, sigmoid=True):

        z, x = data
        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        x = self.pool3d(x)  # Reduce depth dimension to 1
        x = x.squeeze(dim=2)
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)
        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        return x.flatten()



class Autoencoder2D(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Autoencoder2D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by 1/2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Downsample by 1/4
        )

        self.latent = nn.Sequential(
            nn.Conv2d(64, latent_dim, kernel_size=1),  # Transform 128 -> latent_dim
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print("2D pre-encoding", x.size())
        encoded = self.encoder(x)
        # print("2D encoded", encoded.size())
        latent = self.latent(encoded)
        # print("2D latent", latent.size())
        decoded = self.decoder(latent)
        # print("2D decoded before interpolation:", decoded.size())
        decoded = torch.nn.functional.interpolate(decoded, size=x.size()[2:], mode='bilinear', align_corners=False)
        return latent, decoded


class Autoencoder3D(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Autoencoder3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2), # Downsample to 1/2
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2), # Downsample to 1/4
        )

        # Latent transformation layer to adjust channels
        self.latent = nn.Sequential(
            nn.Conv3d(64, latent_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print("3D pre-encoding", x.size())
        encoded = self.encoder(x)
        # print("3D encoded", encoded.size())
        latent = self.latent(encoded)
        # print("3D latent", latent.size())
        decoded = self.decoder(latent)
        # print("3D decoded before interpolation:", decoded.size())
        # Get decoded to line up exactly with initial data, despite pooling, etc.
        decoded = torch.nn.functional.interpolate(decoded, size=x.size()[2:], mode='trilinear', align_corners=False)

        return latent, decoded
    
class Conv_3DoddWithAutoencoders(Conv_3Dodd):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
        latent_dim=16,
    ):
        super(Conv_3DoddWithAutoencoders, self).__init__(
            input_dim, hidden_dim, kernel_size, levels, dropout
        )
        self.autoencoder_2D = Autoencoder2D(input_dim[0], latent_dim)
        self.autoencoder_3D = Autoencoder3D(input_dim[1], latent_dim)

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        latent_size = latent_dim * 64 # latent_dim * 8 * 8 in latent space
        auto_ln_in = latent_size * 2 + ln_in
        self.auto_ln = torch.nn.Sequential(
            torch.nn.Linear(auto_ln_in, (auto_ln_in - latent_size)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d((auto_ln_in - latent_size)),
            torch.nn.Dropout(dropout),
            torch.nn.Linear((auto_ln_in - latent_size), ln_in),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(ln_in),
            torch.nn.Dropout(dropout),
        )  

    def forward(self, data, sigmoid=True):
        z, x = data

        # Pass through autoencoders
        latent_2D, recon_2D = self.autoencoder_2D(z)
        latent_3D, recon_3D = self.autoencoder_3D(x)
        latent_3D = torch.nn.functional.adaptive_avg_pool3d(latent_3D, (1, latent_3D.size(3), latent_3D.size(4))).squeeze(2)
        latent_2D = latent_2D.flatten(start_dim=1)
        latent_3D = latent_3D.flatten(start_dim=1)

        # Combine latent features
        fused_latent = torch.cat((latent_2D, latent_3D), dim=1)

        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        x = self.pool3d(x)
        x = x.squeeze(dim=2) # dropping depth dimension
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)

        # print("x post-spp layer", x.size())
        x = torch.cat((x, fused_latent), dim=1)  # Combine fused_latent with processed features
        # print("x catted with fused_latent", x.size())
        x = self.auto_ln(x)
        # print("x post-auto_ln", x.size())

        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        # Output the main prediction and reconstructions for loss calculation
        return x.flatten(), (recon_2D, recon_3D)


class Conv_3DevenWithAutoencoders(Conv_3Deven):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
        latent_dim=16,
    ):
        super(Conv_3DevenWithAutoencoders, self).__init__(
            input_dim, hidden_dim, kernel_size, levels, dropout
        )
        self.autoencoder_2D = Autoencoder2D(input_dim[0], latent_dim)
        self.autoencoder_3D = Autoencoder3D(input_dim[1], latent_dim)

        ln_in = 0
        for i in levels:
            ln_in += hidden_dim[2] * i * i

        latent_size = latent_dim * 64 # latent_dim * 8 * 8 in latent space
        auto_ln_in = latent_size * 2 + ln_in
        self.auto_ln = torch.nn.Sequential(
            torch.nn.Linear(auto_ln_in, (auto_ln_in - latent_size)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d((auto_ln_in - latent_size)),
            torch.nn.Dropout(dropout),
            torch.nn.Linear((auto_ln_in - latent_size), ln_in),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(ln_in),
            torch.nn.Dropout(dropout),
        )  

    def forward(self, data, sigmoid=True):
        z, x = data

        # Pass through autoencoders
        latent_2D, recon_2D = self.autoencoder_2D(z)
        latent_3D, recon_3D = self.autoencoder_3D(x)
        latent_3D = torch.nn.functional.adaptive_avg_pool3d(latent_3D, (1, latent_3D.size(3), latent_3D.size(4))).squeeze(2)
        latent_2D = latent_2D.flatten(start_dim=1)
        latent_3D = latent_3D.flatten(start_dim=1)

        # Combine latent features
        fused_latent = torch.cat((latent_2D, latent_3D), dim=1)

        z = self.conv_2D.forward(z)
        x = self.conv_3D.forward(x)
        x = self.pool3d(x)
        x = x.squeeze(dim=2) # dropping depth dimension
        x = torch.cat((x, z), dim=1)
        x = self.final.forward(x)
        x = spp_layer(x, self.levels)

        # print("x post-spp layer", x.size())
        x = torch.cat((x, fused_latent), dim=1)  # Combine fused_latent with processed features
        # print("x catted with fused_latent", x.size())
        x = self.auto_ln(x)
        # print("x post-auto_ln", x.size())

        x = self.ln(x)
        if sigmoid:
            x = self.sig(x)

        # Output the main prediction and reconstructions for loss calculation
        return x.flatten(), (recon_2D, recon_3D)


class AttentionBlock2D(nn.Module):
    """Attention mechanism to dynamically weigh 2D feature maps."""
    def __init__(self, in_channels):
        super(AttentionBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = torch.nn.functional.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class AttentionBlock3D(nn.Module):
    """Attention mechanism to dynamically weigh 3D feature maps."""
    def __init__(self, in_channels):
        super(AttentionBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = torch.nn.functional.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class Conv_3DBasicAttention(nn.Module):
    def __init__(
        self,
        input_dim=(2, 8),
        hidden_dim=(16, 32, 32),
        kernel_size=((5, 5), (2, 5, 5), (5, 5)),
        levels=(13,),
        dropout=0.2,
        latent_dim=16,
        time_steps=4,
    ):
        super(Conv_3DBasicAttention, self).__init__()
        input_channels_2d, input_channels_3d = input_dim
        
        # Temporal-Spatial 3D CNN branch with residual connections
        self.cnn3d = nn.Sequential(
            nn.Conv3d(input_channels_3d, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        
        # FPN for 2D Spatial Features
        self.fpn2d = nn.Sequential(
            nn.Conv2d(input_channels_2d, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Attention block to reweight features
        self.attention2D = AttentionBlock2D(64)
        self.attention3D = AttentionBlock3D(64)
        
        self.global_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.global_pool2d = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers for fusion
        self.fc_fusion = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, data, sigmoid=None):
        x2d, x3d = data

        x3d = self.cnn3d(x3d)
        x3d = self.attention3D(x3d)
        x3d = self.global_pool3d(x3d)
        x3d = x3d.flatten(start_dim=1)

        x2d = self.fpn2d(x2d)
        x2d = self.attention2D(x2d)
        x2d = self.global_pool2d(x2d)
        x2d = x2d.flatten(start_dim=1)

        # Feature Fusion
        fused_features = torch.cat((x3d, x2d), dim=1)
        output = self.fc_fusion(fused_features)
        output = output.squeeze(-1)
        return output

