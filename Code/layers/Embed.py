import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)

class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, height=None, width=None):
        super(PositionalEncoding2D, self).__init__()
        if embed_dim % 4 != 0:
            raise ValueError("Embedding dimension must be divisible by 4")
        self.embed_dim = embed_dim
        self.height = height
        self.width = width

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, H, W, D)
        Returns:
            Tensor of same shape with positional encodings added
        """
        B, H, W, D = x.shape
        if self.height is not None and self.height != H:
            raise ValueError(f"Expected height={self.height}, but got {H}")
        if self.width is not None and self.width != W:
            raise ValueError(f"Expected width={self.width}, but got {W}")
        if D != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, but got {D}")

        device = x.device
        d_model = D
        d_half = d_model // 2

        div_term = torch.exp(torch.arange(0., d_half, 2, device=device) * -(math.log(10000.0) / d_half))

        pos_w = torch.arange(0., W, device=device).unsqueeze(1)  # [W, 1]
        pos_h = torch.arange(0., H, device=device).unsqueeze(1)  # [H, 1]

        pe_w = torch.zeros(d_half, H, W, device=device)
        pe_h = torch.zeros(d_half, H, W, device=device)

        pe_w[0::2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
        pe_w[1::2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)

        pe_h[0::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
        pe_h[1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)

        pe = torch.cat([pe_h, pe_w], dim=0)  # [D, H, W]
        pe = pe.permute(1, 2, 0)  # [H, W, D]
        pe = pe.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, D]

        return x + pe


def magnitude_max_pooling_1d(input_tensor, pool_size, stride):
    # Get the dimensions of the input tensor
    B, N, L = input_tensor.size()
    
    # Calculate the output length
    out_length = (L - pool_size) // stride + 1
    
    # Initialize the output tensor
    output_tensor = torch.zeros((B, N, out_length))
    
    # Unfold the input tensor to create sliding windows
    windows = input_tensor.unfold(2, pool_size, stride)
    
    # Reshape the windows to a 4D tensor
    windows = windows.contiguous().view(B, N, out_length, pool_size)
    
    # Compute the magnitudes of the values in each window
    magnitudes = torch.abs(windows)
    
    # Find the indices of the maximum magnitudes in each window
    max_indices = torch.argmax(magnitudes, dim=-1, keepdim=True)
    
    # Gather the values corresponding to the maximum magnitudes
    max_values = windows.gather(dim=-1, index=max_indices).squeeze(-1)
    
    return max_values



class DataEmbedding_FeaturePatching(nn.Module):
    def __init__(self, seq_len,  patch_size,  embed_dim = 512, embed_type='fixed', freq='10min', dropout=0.1):
        super(DataEmbedding_FeaturePatching, self).__init__()
        self.seq_len = seq_len 
        self.patch_size = patch_size
        self.n_of_patches = (seq_len - patch_size)//(patch_size//2) + 1
        self.inner_dim = patch_size * 10
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv1d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv1d(1, 3, kernel_size=9)
        self.conv3 = nn.Conv1d(1, 3, kernel_size=15)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.fc1 = nn.Linear(self.inner_dim, embed_dim*4)
        self.fc2 = nn.Linear(embed_dim*4, embed_dim)
        self.pe  = PositionalEncoding2D(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.sigm = nn.GELU()

    def forward(self, x, x_mark):
        B, L, N = x.shape
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]

        if x_mark is not None:
            N += x_mark.shape[2]
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)

        x = x.reshape(-1, 1, L)
        x_1 = F.pad(x, (4, 0), mode = 'replicate')
        x_1 = self.conv1(x_1)
        x_2 = F.pad(x, (8, 0), mode = 'replicate')
        x_2 = self.conv2(x_2)
        x_3 = F.pad(x, (14, 0), mode = 'replicate')
        x_3 = self.conv3(x_3)
        x_1 = F.pad(x_1, (2, 0), mode = 'constant', value = 0)
        x_2 = F.pad(x_2, (4, 0), mode = 'constant', value = 0)
        x_3 = F.pad(x_3, (6, 0), mode = 'constant', value = 0)

        x_1 = magnitude_max_pooling_1d(x_1, 3, 1)
        x_2 = magnitude_max_pooling_1d(x_2, 5, 1)
        x_3 = magnitude_max_pooling_1d(x_3, 7, 1)

        

        x_1 = x_1.reshape(B, N, 3, L)
        x_2 = x_2.reshape(B, N, 3, L)
        x_3 = x_3.reshape(B, N, 3, L)
        x = x.reshape(B, N, 1, L)
        

        x_1 = x_1.unfold(3, self.patch_size, self.patch_size//2)
        x_2 = x_2.unfold(3, self.patch_size, self.patch_size//2)
        x_3 = x_3.unfold(3, self.patch_size, self.patch_size//2)
        x = x.unfold(3, self.patch_size, self.patch_size//2)


        x_1 = x_1.permute(0, 1, 3, 2, 4)
        x_2 = x_2.permute(0, 1, 3, 2, 4)
        x_3 = x_3.permute(0, 1, 3, 2, 4)
        x = x.permute(0, 1, 3, 2, 4)


        x = torch.cat([x, x_1, x_2, x_3], dim = 3)


        x = x.reshape(B, N, self.n_of_patches, -1)
        x = self.gelu1(self.fc1(x))
        x = self.fc2(x)
        x = self.pe(x) + x #apply 2D positional encodings

        x = x.reshape(B, -1, self.embed_dim)

        return self.dropout(x)