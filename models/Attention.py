import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

## Luong et al ##
class Attention(nn.Module):
    def __init__(self, dim, transform=0):
        super(Attention, self).__init__()
        
        if transform != 0:
            self.transform = True
            self.linear_in = nn.Linear(dim, transform)
            self.linear_out = nn.Linear(transform*2, transform)
        else:
            self.transform = False
            self.linear_out = nn.Linear(dim*2, dim)

        # self.U = nn.Linear(dim,dim)

    def forward(self, output, context):
        # output:  decoder hidden state
        # context: encoder outputs

        
        if self.transform: output = self.linear_in(output)

        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # context = self.U(context)
        attn = torch.bmm(output, context.transpose(1, 2))
        attn = F.softmax(attn.view(-1, input_size),dim=1).view(batch_size, -1, input_size)
        mix = torch.bmm(attn, context)

        combined = torch.cat((mix, output), dim=2)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


## Dot ##
class Attention1(nn.Module):
    def __init__(self, dim):
        super(Attention1, self).__init__()
        

    def forward(self, decoder_hidden, encoder_outputs):

        batch_size = decoder_hidden.size(0)

        attn = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))
        attn = F.softmax(attn.view(-1, encoder_outputs.size(1)),dim=1).view(batch_size, -1, encoder_outputs.size(1))
        context = torch.bmm(attn, encoder_outputs)

        return context, attn

## General ##
class Attention2(nn.Module):
    def __init__(self, dim):
        super(Attention2, self).__init__()
        self.U = nn.Linear(dim,dim)

    def forward(self, decoder_hidden, encoder_outputs):

        batch_size = decoder_hidden.size(0)

        encoder_outputs = self.U(encoder_outputs)
        attn = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))
        attn = F.softmax(attn.view(-1, encoder_outputs.size(1)),dim=1).view(batch_size, -1, encoder_outputs.size(1))
        context = torch.bmm(attn, encoder_outputs)

        return context, attn

## Concatenate ##
class Attention3(nn.Module):
    def __init__(self, dim):
        super(Attention3, self).__init__()
        self.W = nn.Linear(dim,dim)
        self.U = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,1)


    def forward(self, decoder_hidden, encoder_outputs):

        batch_size = decoder_hidden.size(0)
        encoder_length = encoder_outputs.size(1)

        attn = self.v(F.tanh(self.W(decoder_hidden) + self.U(encoder_outputs)))

        attn = F.softmax(attn.view(-1, encoder_outputs.size(1)),dim=1).view(batch_size, -1, encoder_outputs.size(1))
        context = torch.bmm(attn, encoder_outputs)

        return context, attn






