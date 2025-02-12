import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CausalFlowModel(nn.Module):

    def __init__(self,
                 state_dim,
                 control_dim,
                 output_dim,
                 control_rnn_size,
                 control_rnn_depth,
                 encoder_size,
                 encoder_depth,
                 decoder_size,
                 decoder_depth,
                 use_batch_norm=False):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        # self.output_dim = output_dim
        self.modes = 16
        self.output_dim = self.modes
        self.trunk_modes = 16
        self.control_rnn_size = control_rnn_size
        self.trunk_enabled = False

        x_dnn_osz = control_rnn_depth * control_rnn_size
        if self.trunk_enabled:
            out_size_flow = self.modes+self.trunk_modes
            self.trunk = Trunk(in_size=1,
                                out_size=self.trunk_modes,
                                hidden_size=encoder_depth *
                                (encoder_size * x_dnn_osz, ), # hidden size is equal encoder depth (encoder_size*x_dnn_osz)
                            # if encoder depth = 2 -> (encoder_size*x_dnn_osz, encoder_size*x_dnn_osz)
                                use_batch_norm=use_batch_norm)
        else:
            self.trunk = None
            out_size_flow = self.modes

        # project the inputs to the flow model 
        self.projection = True
        if self.projection:
            in_size_flow = self.modes
            in_size_rnn = self.modes
        else:
            in_size_flow = state_dim
            in_size_rnn = control_dim


        self.u_rnn = torch.nn.LSTM(
            input_size=1 + in_size_rnn,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=control_rnn_depth,
            dropout=0,
        )

        self.x_dnn = FFNet(in_size=in_size_flow,
                        out_size=x_dnn_osz,
                        hidden_size=encoder_depth *
                        (encoder_size * x_dnn_osz, ), # hidden size is equal encoder depth (encoder_size*x_dnn_osz)
                        # if encoder depth = 2 -> (encoder_size*x_dnn_osz, encoder_size*x_dnn_osz)
                        use_batch_norm=use_batch_norm)

        u_dnn_isz = control_rnn_size
        self.u_dnn = FFNet(in_size=u_dnn_isz,
                        out_size=out_size_flow,
                        hidden_size=decoder_depth *
                        (decoder_size * u_dnn_isz, ),
                        use_batch_norm=use_batch_norm)
            
        ### Trunk net used to encode locations and find phi(x), takes as as input a location x
        
        # self.bias = nn.Parameter(torch.zeros(1, output_dim))  # Shape: [1, num_locations]
        self.bias = nn.Parameter(torch.tensor(0.0))


    def forward(self, x, rnn_input, deltas,X_loc,POD):
        if self.projection:
            # project initial state
            x = torch.einsum("bni,bn->bi",POD[:,:,:self.modes],x)

            # project input of the RNN
            unpadded_seq, lengths = pad_packed_sequence(rnn_input, batch_first=True)
            POD_0 = POD[:,0,:self.modes] # modes corresponding to x0
            U_without_deltas = unpadded_seq[:,:,0] # select the inputs to project
            U_deltas = unpadded_seq[:,:,1] # the delta values
            U_projected = torch.einsum("bi,bj->bij",U_without_deltas,POD_0) # project the inputs
            U_projected = torch.cat([U_projected,U_deltas.unsqueeze(-1)],dim=-1) # combine projected inputs and deltas
            rnn_input = pack_padded_sequence(U_projected, lengths, batch_first=True, enforce_sorted=True)

        h0 = self.x_dnn(x)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)


        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                           batch_first=True)

        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0[-1]

        encoded_controls = (1 - deltas) * h_shift + deltas * h
        X_func = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :])
        
        # no trunk net
        if self.trunk is None:
            output = torch.einsum("bi,bni->bn", X_func,POD[:,:,:self.modes] )

        # trunk net
        else:
            X_loc = self.trunk(X_loc)
            X_loc = X_loc.unsqueeze(0).expand(POD.shape[0],-1,-1)
            output = torch.einsum("bi,bni->bn", X_func, torch.concat((POD[:,:, :self.modes], X_loc), 2))
            output += self.bias

        
        return output

class CausalFlowModelV2(nn.Module):

    def __init__(self,
                 state_dim,
                 control_dim,
                 output_dim,
                 control_rnn_size,
                 control_rnn_depth,
                 encoder_size,
                 encoder_depth,
                 decoder_size,
                 decoder_depth,
                 use_batch_norm=False):
        super(CausalFlowModelV2, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim

        self.control_rnn_size = control_rnn_size

        self.u_rnn = torch.nn.LSTM(
            input_size=1 + control_dim,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=control_rnn_depth,
            dropout=0,
        )

        ### ENCODER ###
        x_dnn_osz = control_rnn_depth * control_rnn_size
        
        # Convolutional encoder
        self.encoder = CONV_Encoder(in_size=state_dim,
                               out_size=x_dnn_osz)
        
        ### DECODER ###
        u_dnn_isz = control_rnn_size
        # Convolutional decoder 
        self.decoder = CONV_Decoder(in_size=u_dnn_isz,
                                    out_size=output_dim)

    def forward(self, x, rnn_input, deltas):
        h0 = self.encoder(x)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                           batch_first=True)

        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0[-1]

        encoded_controls = (1 - deltas) * h_shift + deltas * h
       
        output = self.decoder(encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :])
        return output
    
class Trunk(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 activation=nn.ReLU,
                 use_batch_norm=False):
        super(Trunk, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, 40))
        self.layers.append(activation())
        self.layers.append(nn.Linear(40, 40))
        self.layers.append(activation())
        self.layers.append(nn.Linear(40, out_size))

        # if use_batch_norm:
        #     self.layers.append(nn.BatchNorm1d(hidden_size[0]))

        # self.layers.append(activation())

        # for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
        #     self.layers.append(nn.Linear(isz, osz))

        #     if use_batch_norm:
        #         self.layers.append(nn.BatchNorm1d(osz))



    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
    
class FFNet(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 activation=nn.Tanh,
                 use_batch_norm=False):
        super(FFNet, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))

        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_size[0]))

        self.layers.append(activation())

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))

            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))

            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

class CONV_Encoder(nn.Module):
    def __init__(self,
                 in_size,
                 out_size):         
        super(CONV_Encoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        # activation and pool layer
        self.activation = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)
        
        # convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(in_channels=16,out_channels=8,kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(in_channels=8,out_channels=4,kernel_size=3,padding=1)
        
        # linear layer to get right dimensionality of hidden states
        self.fc = nn.Linear(in_features=100,out_features=out_size)

    def forward(self, input):
        # reshape input from [1,batch_size,channel] -> [batch_size,1,channel] ??
        input = input.view(-1,1,self.in_size)
        input = self.activation(self.conv1(input))
        input = self.pool(input)
        input = self.activation(self.conv2(input))
        input = self.pool(input)
        input = self.activation(self.conv3(input))
        
        ## omit this one cause more downsizing isnt needed
        # input = self.pool(input)

        # flatten  
        input = input.view(input.size(0),1,-1)
        input = self.fc(input)
        
        # reshape to (batch_size,out_size)
        input = input.view(-1,self.out_size)
        return input

class CONV_Decoder(nn.Module):
    def __init__(self,
                 in_size,
                 out_size):         
        super(CONV_Decoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        # activation and pool layer
        self.activation = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)
        
        # convolutional layers
        # self.TransposeConv1 = nn.ConvTranspose1d(in_channels=4,out_channels=4,kernel_size=2,stride=2,padding=0,output_padding=0)
        self.TransposeConv2 = nn.ConvTranspose1d(in_channels=4,out_channels=8,kernel_size=2,stride=2,padding=0)
        self.TransposeConv3 = nn.ConvTranspose1d(in_channels=8,out_channels=16,kernel_size=2,stride=2,padding=0)
        self.TransposeConv4 = nn.ConvTranspose1d(in_channels=16,out_channels=1,kernel_size=3,stride=1,padding=1)
        
        # linear layer to get right dimensionality of hidden states
        self.fc = nn.Linear(in_features=in_size,out_features=100)
        self.fc2 = nn.Linear(in_features=100,out_features=out_size)

    def forward(self, input):
        # reshape input from [1,batch_size,channel] -> [batch_size,1,channel] 
        input = input.view(-1,1,self.in_size)

        # Linear Layer receiving Ld
        input = self.activation(self.fc(input))

        # reshape to CNN input
        # input = input.view(-1,4,12)
        input = input.view(-1,4,25)

        # ConvTranspose layers
        # input = self.activation(self.TransposeConv1(input))
        
        input = self.activation(self.TransposeConv2(input))
        input = self.activation(self.TransposeConv3(input))
        input = self.activation(self.TransposeConv4(input))

        # reshape for linear layer
        # input = input.view(input.size(0),1,-1)
        # print(input.shape)
        
        # final layer is linear layer
        input = self.fc2(input)

        # reshape for correct output
        input = input.view(-1,self.out_size)

        return input


