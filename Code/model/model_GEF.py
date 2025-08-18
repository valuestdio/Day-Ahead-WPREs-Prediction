import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding_FeaturePatching
from layers.mark import generate_x_mark
from layers.Invertible import RevIN

class BP(nn.Module):
    def __init__(self, args):
        super(BP, self).__init__()
        self.P = args.P          # 输入时间步长
        self.Q = args.Q          # 输出时间步长

        # 直接线性映射：输入维度P -> 输出维度Q
        self.fc = nn.Linear(self.P, self.Q)

    def forward(self, X, SE=None, TE=None, WE=None):
        # X_VMD shape: (batch, P, 134)

        # 把时间维度(P)和特征维度拼在一起: (batch, 134, P)
        X = X.permute(0, 2, 1)

        X = self.fc(X)  
        # reshape成 (batch, Q, 134)
        X = X.permute(0, 2, 1)
        return X

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.P = args.P          # 输入时间步长
        self.Q = args.Q          # 输出时间步长
        self.hidden_dim = args.d_ff if hasattr(args, 'd_ff') else 128  # 隐藏层维度
        self.dropout_rate = args.dropout if hasattr(args, 'dropout') else 0.1

        self.input_size = self.P
        # 简单三层前馈网络
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.Q)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = F.gelu

    def forward(self, X, SE=None, TE=None, WE=None):
        # X_VMD shape: (batch, P, 134)
        batch_size = X.size(0)

        # 把时间维度(P)和特征维度拼在一起: (batch, 134, P)
        X = X.permute(0, 2, 1)

        # 全连接层要求输入是 (batch, features)，所以 flatten：-> (batch * 134, P)
        X = X.reshape(batch_size * 10, self.P)

        # 经过FC层：每个bus位置有一个BP模型
        X = F.relu(self.dropout(self.fc1(X)))
        X = F.relu(self.dropout(self.fc2(X)))
        X = self.fc3(X)  # -> (batch*134, Q)

        # reshape成 (batch, Q, 134)
        X = X.view(batch_size, 10, self.Q).permute(0, 2, 1)
        return X

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.P = args.P       # 输入时间步长
        self.Q = args.Q       # 输出时间步长

        self.hidden_dim = args.d_ff if hasattr(args, 'd_ff') else 128
        self.dropout_rate = args.dropout if hasattr(args, 'dropout') else 0.1

        kernel_size = 3
        padding = 1

        self.conv1 = nn.Conv1d(in_channels=self.P, out_channels=self.hidden_dim, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=kernel_size, padding=padding)

        self.dropout = nn.Dropout(self.dropout_rate)

        # 最后映射到Q个预测值（每个特征一个输出）
        self.fc = nn.Linear(self.hidden_dim, self.Q)

    def forward(self, X, SE=None, TE=None, WE=None):

        # (batch, P, num_features)

        X = F.relu(self.dropout(self.conv1(X)))
        X = F.relu(self.dropout(self.conv2(X)))

        X = X.permute(0, 2, 1)
        X = self.fc(X) 
        X = X.permute(0, 2, 1)

        return X

class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.P = args.P  # 历史时间步长
        self.Q = args.Q  # 预测时间步长
        self.hidden_size = 288
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size = self.P,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first=True
        )

        self.proj = nn.Linear(self.hidden_size, self.Q)

    def forward(self, x, SE=None, TE=None, WE=None):
        # x: (batch, P, 134)
        batch_size = x.size(0)
        x= x.permute(0,2,1)
        lstm_out, _ = self.lstm(x)  # (batch, 134, hidden_size)

        out = self.proj(lstm_out)  # (batch, 134, Q)

        return out.permute(0,2,1)
    
class GRUModel(nn.Module):
    def __init__(self, args):
        super(GRUModel, self).__init__()
        self.P = args.P  # 历史时间步长
        self.Q = args.Q  # 预测时间步长
        self.hidden_size = 288
        self.num_layers = 2

        self.gru = nn.GRU(
            input_size=self.P,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.proj = nn.Linear(self.hidden_size, self.Q)

    def forward(self, x, SE=None, TE=None, WE=None):
        # x: (batch, P, 134)
        x = x.permute(0, 2, 1)  # => (batch, 134, P)
        gru_out, _ = self.gru(x)  # => (batch, 134, hidden_size)
        out = self.proj(gru_out)  # => (batch, 134, Q)
        return out.permute(0, 2, 1)  # => (batch, Q, 134)
    
class CNNLSTM(nn.Module):
    def __init__(self, args):
        super(CNNLSTM, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.hidden_size = 288
        self.num_layers = 2
        self.conv_channels = 288
        self.W = 4

        # 风电功率主输入 CNN（逐风机）
        self.main_conv = nn.Conv1d(in_channels=self.P, out_channels=self.conv_channels, kernel_size=3, padding=1)

        self.we = nn.Linear(self.W,1)
        # 天气嵌入层（逐风机）
        self.we_conv = nn.Conv1d(in_channels=self.P, out_channels=self.conv_channels, kernel_size=3, padding=1)
        # LSTM 输入维度：主输入 + weather 特征
        self.lstm_input_dim = self.conv_channels * 2

        # LSTM 逐风机建模
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # 输出映射：逐风机输出Q步预测
        self.output_layer = nn.Linear(self.hidden_size, self.Q)

    def forward(self, x, SE, TE, WE):
        # x:  (batch, P, 134)
        # WE: (batch, P, 134, W)
        batch_size = x.size(0)

        # 主输入功率通过1D卷积
        x_conv = self.main_conv(x)  # (batch, conv_channels, 134)

        # 处理天气特征
        WE = self.we(WE)
        WE = torch.squeeze(WE, 3)          # (batch, P, 134)
        WE_conv = self.we_conv(WE)              # (batch, conv_channels, 134)

        x_conv = x_conv.permute(0, 2, 1) 
        WE_conv = WE_conv.permute(0, 2, 1) 
        # 拼接主输入和天气特征：在特征维度拼接
        x_all = torch.cat([x_conv, WE_conv], dim=-1)  # (batch, 134, lstm_input_dim)

        # LSTM 序列建模
        x_all, _ = self.lstm(x_all)                          # (batch, 134, hidden_size)
        out = self.output_layer(x_all)
        out = out.permute(0, 2, 1)                   

        return out  # (batch, Q, 134)
    
class Itransformer(nn.Module):
    def __init__(self, args):
        super(Itransformer, self).__init__()
        self.seq_len = args.P
        self.pred_len = args.Q
        self.freq = args.freq
        self.output_attention = False
        self.use_norm = False
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, args.d_model, args.embed, args.freq,
                                                    args.dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention= self.output_attention), args.d_model, args.K),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation= F.gelu
                ) for l in range(args.L)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        self.projector = nn.Linear(args.d_model, self.pred_len, bias=True)

    def forward(self, x_enc, SE, TE, WE):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: turbines
        x_mark_enc = generate_x_mark(x_enc,'2014-01-01 00:00:00',self.freq)
        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
    
class RLinear(nn.Module):
    def __init__(self, args):
        super(RLinear, self).__init__()

        self.individual = True
        self.rev = True
        self.Linear = nn.ModuleList([
            nn.Linear(args.P, args.Q) for _ in range(10)
        ]) if self.individual else nn.Linear(args.P, args.Q)

        self.dropout = nn.Dropout(args.dropout)
        self.rev = RevIN(10) if self.rev else None

    def forward(self, x, SE, TE, WE):
        # x: [B, L, D] = [B, P, 10]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)

        if self.individual:
            B, _, D = x.shape  # B=batch, P=seq_len, D=feature_dim=10
            pred = x.new_zeros(B, self.Linear[0].out_features, D)  # [B, Q, D]
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])  # 每个变量独立映射
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)

        pred = self.rev(pred, 'denorm') if self.rev else pred
        return pred

class TimePFN(nn.Module):

    def __init__(self, args):
        super(TimePFN, self).__init__()
        self.seq_len = args.P
        self.pred_len = args.Q
        self.output_attention = False
        self.use_norm = args.use_norm
        self.patch_size =12
        self.embed_dim = 128
        self.freq = args.freq
        # args
        self.enc_embedding = DataEmbedding_FeaturePatching(self.seq_len, self.patch_size, embed_dim = self.embed_dim, embed_type='fixed', freq= self.freq, dropout=0.1)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=self.output_attention), self.embed_dim, args.K),
                    self.embed_dim,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=F.gelu
                ) for l in range(args.L)
            ],
            norm_layer=torch.nn.LayerNorm(self.embed_dim),
        )

        self.head = nn.Linear(self.embed_dim, self.patch_size, bias=True)
        self.projector1 = nn.Linear(((self.seq_len-self.patch_size)//(self.patch_size//2)+1)*(self.embed_dim), args.d_model, bias=True)
        self.non_linear = nn.GELU()
        self.projector_mid = nn.Linear(args.d_model, args.d_model, bias=True)
        self.non_linear_mid = nn.GELU()
        self.projector2 = nn.Linear(args.d_model, self.pred_len, bias=True)


    def forecast(self, x_enc):
        
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-4)
            x_enc /= stdev

        x_mark_enc = generate_x_mark(x_enc,'2014-01-01 00:00:00',self.freq)
        
        B, _, N = x_enc.shape # B L N
        if x_mark_enc is not None:
            N_ = N +  x_mark_enc.shape[2]
        else :
            N_ = N


        enc_out = self.enc_embedding(x_enc, x_mark_enc) 
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out.reshape(B, N_, -1)

        dec_out = self.projector1(enc_out) 
        dec_out = self.non_linear(dec_out)
        dec_out = self.projector2(dec_out).permute(0, 2, 1)[:, :, :N]


        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]