import torch
import torch.nn as nn
import numpy as np
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class PatchEmbeddings(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.patch_size=cfg["patch"]
        self.img_size=cfg["img"]
        self.hidden_size=cfg["dim"]
        self.no_patch=(self.img_size//self.patch_size)**2
        self.layer=nn.Conv2d(3,self.hidden_size,kernel_size=self.patch_size,stride=self.patch_size)
    def forward(self,x):
        x=self.layer(x)
        x=x.flatten(2).transpose(1,2)
        return x
class Embeedings(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.patch_embed=PatchEmbeddings(config)
        self.cls_token=nn.Parameter(torch.rand(1,1,config["dim"]))
        self.pos_embed=nn.Parameter(torch.rand(1,self.patch_embed.no_patch+1,config["dim"]))
               
    def forward(self,x):
        x=self.patch_embed(x)
        batch_size,_,_=x.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x= torch.cat((cls_tokens, x), dim=1)
        x=self.pos_embed+x
        return x
class Selfattention(nn.Module):
    def __init__(self,inp_dim ):
        super().__init__( )
        self.inp_dim=inp_dim
        self.w_q=nn.Linear(inp_dim,inp_dim)
        self.w_k=nn.Linear(inp_dim,inp_dim)
        self.w_v=nn.Linear(inp_dim,inp_dim)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.droupout=nn.Dropout(0.1)
    def forward(self,x):
        b,p,d=x.shape
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)
        attn=q@k.transpose(1,2)
        attn.masked_fill_(  
            self.mask.bool()[:p, :p], -torch.inf)
        dim_k=k.shape[-1]
        attn=attn/dim_k**0.5
        
        attnscores=torch.softmax(attn)
        attnscores=self.droupout(attnscores)
        
        return attnscores@v
        
class MultiAttention(nn.Module):
    def __init__(self,n_heads,inp_dim):
        super().__init__()
        self.head_dim=inp_dim//n_heads
        self.model=nn.ModuleList(
            [Selfattention(inp_dim) for _ in n_heads]
        )
    def forward(self,x):
        x=torch.cat([module(x) for module in self.model])
        return x
class MLP(nn.Module):
    def __init__(self,inp_dim,drop_rate):
        self.layer1=nn.Linear(inp_dim,4*inp_dim)
        self.layer2=nn.Linear(4*inp_dim,inp_dim)
        self.dropout=nn.Dropout(drop_rate)
    def froward(self,x):
        x=self.layer1(x)
        x=GELU(x)
        x=self.layer2(x)
        x=self.dropout(x)
        return x
class Transformer(nn.Module):
    def __init__(self,dim,n_heads):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.attn=MultiAttention(n_heads,dim)
        self.mlp=MLP(dim,0.1)
    def forward(self,x):
        residue=x
        x=self.norm(x)
        x=self.attn(x)
        x=residue+x
        residue=x
        x=self.norm(x)
        x=self.mlp(x)
        x=residue+x
        return x
class Encoder(nn.Module):
    def __init__(self,cfg):
        self.transformers=nn.Sequential(
            [
                Transformer(cfg["dim"],cfg["n_heads"])for _ in cfg["layers"]
                
            ]
        )
        
    
        
    

        
        