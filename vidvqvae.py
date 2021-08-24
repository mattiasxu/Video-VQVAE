import torch
from torch import nn
from torch.nn import functional as F

# Implementation based on https://github.com/rosinality/vq-vae-2-pytorch

class Quantizer(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv3d(in_channel, channel, kernel_size=(4, 8, 8), stride=(2,4,4), padding=(1,2,2))
            ]

        elif stride == 2:
            blocks = [
                nn.Conv3d(in_channel, channel, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1,1,1))
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel
    ):
        super().__init__()

        blocks = [nn.Conv3d(in_channel, channel, 3, padding=1)]
        
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        
        blocks.append(nn.ReLU(inplace=True))
        
        blocks.extend(
            [
                nn.ConvTranspose3d(channel, out_channel, kernel_size=(4,8,8), stride=(2,4,4), padding=(1,2,2))
            ]
        ) 
                
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, input):
        return self.blocks(input)

if __name__ == "__main__":
    rand_input = torch.rand((8, 3, 16, 256, 256))
    bot_test = Encoder(3, 4, 8, 8, stride=4)
    top_test = Encoder(4, 4, 8, 8, stride=2)

    bot_out = bot_test(rand_input)
    decoder = Decoder(4, 3, 128, 5, 10)
    print(bot_out.shape)
    print(decoder(bot_out).shape)