import torch
import torch.nn as nn
import math
config = {
        "patch": 16,
        "dim": 768,
        "n_heads": 12,
        "layers": 12,
        "image_size": 128,
        "image_layer":128,
        "channel":4
        
    }
class Deconv(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.deconv=nn.ConvTranspose3d(in_dim,out_dim,kernel_size=2, stride=2, padding=0, output_padding=0)
    def forward(self,x):
        x=self.deconv(x)
        return x
    
class Conv(nn.Module):
    def __init__(self,in_dim,out_dim,k):
        super().__init__()
        self.conv=nn.Conv3d(in_dim,out_dim,kernel_size=k,stride=1,padding=((k - 1) // 2))
    def forward(self,x):
        return self.conv(x)
    
class ConvBlock(nn.Module):
    def __init__(self,indim,outdim):
        super().__init__()
        self.conv=Conv(indim,outdim,3)
        self.norm=nn.BatchNorm3d(outdim)
        self.activation=nn.ReLU()
    def forward(self,x):
        x=self.conv(x)
        x=self.norm(x)
        x=self.activation(x)
        return x
class DeconvBlock(nn.Module):
    def __init__(self,indim,outdim):
        super().__init__()
        self.deconv=Deconv(indim,outdim)
        self.conv=Conv(indim,outdim,3)
        self.norm=nn.BatchNorm3d(outdim)
        self.activation=nn.ReLU()
    def forward(self,x):
        x=self.deconv(x)
        x=self.conv(x)
        x=self.norm(x)
        x=self.activation(x)
        return x

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class PatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg["patch"]
        self.img_size = cfg["image_size"]
        self.hidden_size = cfg["dim"]
        self.depth=cfg["image_layer"]
        self.no_patch = (
            (self.img_size // self.patch_size) *
            (self.img_size // self.patch_size) *
            (self.depth // self.patch_size)
        )
        self.layer = nn.Conv3d(cfg["channel"], self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
    
    def forward(self, x):
        x = self.layer(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbeddings(config)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.no_patch , config["dim"]))

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size, _, _ = x.shape
        x = self.pos_embed + x
        return x


class SelfAttention(nn.Module):
    def __init__(self, inp_dim,outdim, cfg):
        super().__init__()
        self.patch_size = cfg["patch"]
        self.img_size = cfg["image_size"]
        self.context_length = (self.img_size // self.patch_size) ** 2 
        self.inp_dim = inp_dim
        self.w_q = nn.Linear(inp_dim, outdim)
        self.w_k = nn.Linear(inp_dim, outdim)
        self.w_v = nn.Linear(inp_dim, outdim)
        self.dropout = nn.Dropout(0.1)
        self.attn_weights = None  # Store attention weights
    
    def forward(self, x):
        b, p, d = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        attn = q @ k.transpose(1, 2)
        dim_k = k.shape[-1]
        attn = attn / dim_k ** 0.5
        attn_scores = torch.softmax(attn, dim=-1)
        self.attn_weights = attn_scores  # Save for visualization
        attn_scores = self.dropout(attn_scores)
        return attn_scores @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, inp_dim, cfg):
        super().__init__()
        self.head_dim = inp_dim // n_heads
        self.n_heads = n_heads
        self.heads = nn.ModuleList([SelfAttention(inp_dim,self.head_dim, cfg) for _ in range(n_heads)])
        
    
    def forward(self, x):
        outputs = []
        for  head in self.heads:
            
            outputs.append(head(x))
        
        x = torch.cat(outputs, dim=-1)
        return x
    
 


class MLP(nn.Module):
    def __init__(self, inp_dim, drop_rate):
        super().__init__()
        self.layer1 = nn.Linear(inp_dim, 4 * inp_dim)
        self.layer2 = nn.Linear(4 * inp_dim, inp_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.activation = GELU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, n_heads, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(n_heads, dim, cfg)
        self.mlp = MLP(dim, 0.1)
    
    def forward(self, x):
        residue = x
        x = self.norm1(x)
        x = self.attn(x)
        x = residue + x
        residue = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residue + x
        return x


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([
            Transformer(cfg["dim"], cfg["n_heads"], cfg) for _ in range(cfg["layers"])
        ])
        self.embedding = Embeddings(cfg)
    
    def forward(self, x,intermidiate=True):
        x = self.embedding(x)
        if not intermidiate:
            
            for layer in self.layers:
                x = layer(x)
            return x
        outputs = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Store outputs at layers 3, 6, 9, 12 (adjust indices as needed)
            if (i + 1) % 3 == 0:  # Every 3rd layer
                outputs[f'z{i+1}'] =x
        
        return outputs
class Unetr(nn.Module):
    def __init__(self,cfg):
        pass
    def forward(self,x):
        pass
    
        



    
  
