import torch
import torch.nn as nn

class VitEncoderBlock(nn.Module):

    def __init__(self, dim, nhead, mlp_ratio=4.0):
        super().__init__()
        self.norm1= nn.LayerNorm(dim)
        self.norm2= nn.LayerNorm(dim)
        self.atten= nn.MultiheadAttention(dim, nhead, dropout=0.1,batch_first=True)
        hidden_dim= int(mlp_ratio * dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        # x: [B, num_patches, d_model] is [B,49,512]
        residual = x
        x = self.norm1(x)          
        attn_out, _ = self.atten(x, x, x) 
        x = residual + attn_out    
        residual = x
        x = self.norm2(x)          
        x = self.mlp(x)            
        x = residual + x
        return x

class MyVisionEncoder(nn.Module):
    def __init__(self,img_size=224, patch_size=32, dim=512, nhead=8, num_layers=4):
        super().__init__()
        self.patch_size= patch_size
        self.num_patches= (img_size // patch_size) ** 2
        self.patch_embedding= nn.Conv2d(
            in_channels=3,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # tell the position of the patch embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        # use several vit encoder block
        blocks = []
        for _ in range(num_layers):
            blocks.append(VitEncoderBlock(dim, nhead))
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(dim)

    def forward(self,x):
        # x: [B, 3, 224, 224]
        
        B = x.shape[0]
        # Step1 Patch Embedding
        x = self.patch_embedding(x)  # [B,512,7,7]
        # Step2 reshape
        x = x.flatten(2).transpose(1, 2) # [B,49,512]
        # Step3 add position embedding
        x = x + self.pos_embedding
        # Step4 多层Transformer Encoder blocks
        x = self.blocks(x)
        x = self.norm(x)
        return x # output: [B,49,512]
        