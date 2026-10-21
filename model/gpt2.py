from datasets import load_dataset
import math
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import TensorDataset, DataLoader
import prompt.prompt_vision as prompt_vision
import model.vision_encoder as vision_encoder
from torchvision import transforms
from PIL import Image
import prompt.prompt_wiki as prompt_wiki

# InfoNCE temperature band, in log space (scale = exp(logit_scale) = 1/tau).
# Both sides of the InfoNCE are L2-normalised, so this scale has no legitimate job
# of tracking a changing feature norm; and the batch is a fixed 128, so the optimal
# sharpness does not need to move either. All a bigger value buys is a sharper
# softmax, which lowers the loss while the representation is exactly as
# (non-)discriminative as before. One run drifted 14.3 -> 20.8 under the old cap of
# 100. Held to tau in [0.05, 0.10] instead: the band still allows a 2x move, but
# caps what temperature alone can fake at ln(20) - ln(10) = 0.69 nats.
LOGIT_SCALE_MIN = math.log(1 / 0.10)
LOGIT_SCALE_MAX = math.log(1 / 0.05)


class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, n_heads=8, n_layers=6, block_size=320):
        super().__init__()
        self.block_size=block_size

        self.token_emb = nn.Embedding(vocab_size,d_model)

        self.pos_emb = nn.Embedding(block_size,d_model)
        
        self.layers= nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TransformerBlock(d_model, n_heads, block_size))

        self.ln = nn.LayerNorm(d_model)    # 最后的归一化
        self.head = nn.Linear(d_model, vocab_size)  # 输出 logits

        # create vision encoder
        self.vision_encoder = vision_encoder.MyVisionEncoder()
        self.no_image_emb= nn.Parameter(torch.randn(1,49, d_model))
        self.projector= nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),nn.LayerNorm(d_model))

        # ---- contrastive head (align stream only) ---------------------------
        # The wiki stream never touches these two. They live inside GPT2 so that
        # the paired features come out of the SAME forward as the CE logits (no
        # second pass over the model) and so a probe can recompute them without
        # duplicating a code path.
        self.con_img = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Linear(d_model, 256))
        self.con_txt = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Linear(d_model, 256))
        # similarity scale, i.e. exp(logit_scale) = 1/0.07 at init, constrained to
        # the narrow band declared at the top of this file.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, x, img=None, return_hidden=False):
        # x shape: (batch, seq_len) = (32, 255)，即 32 个样本，每个 255 个 token
        tok_emb = self.token_emb(x)      
        # 把每个 token ID 变成 512 维向量
        # (32, 255) → (32, 255, 512) 
        if img is not None:
            vision_emb = self.vision_encoder(img)
            vision_emb = self.projector(vision_emb) # (32, 49, 512)
        else:
            vision_emb= self.no_image_emb.expand(x.size(0), -1, -1)    
        tok_emb[:, :49, :] = vision_emb
              
        pos = torch.arange(x.size(1), device=x.device) 
        # 生成位置索引 [0, 1, 2, ..., 254]
        # device=x.device 确保在同一个设备上（CPU/GPU）

        pos_emb = self.pos_emb(pos)    
        # 查表得到位置向量
        # (32, 255) → (32, 255, 512)  
                   
        x = tok_emb + pos_emb  
        # 广播相加：(32, 255, 512) + (255, 512) → (32, 255, 512)
        # 每个 token 现在既有语义信息，也有位置信息
        
        for layer in self.layers:
            x = layer(x)
        # 依次通过 6 个 TransformerBlock，每层提取更复杂的特征

        x = self.ln(x)       # 最后归一化
        logits = self.head(x) 
        # 线性层：(32, 255, 512) → (32, 255, 50257)
        # 每个位置输出 50257 个分数，表示下一个 token 是词表中每个词的概率
        if return_hidden:
            # pre-head hidden. The contrastive term and the retrieval probe both
            # read this instead of the logits: matching in the 50257-dim logit
            # space would be dominated by the language prior.
            return logits, x
        return logits

    def align_features(self, hidden, anchor_idx):
        """Build the two anchors the contrastive term pairs up.

        image: mean over the 49 image slots. They occupy positions 0..48 and the
            attention is causal, so hidden[:, :49] is a function of vision_emb
            alone -- the caption can never leak into it. That is exactly what
            keeps the contrastive term from collapsing into "copy the sentence
            into an image slot", so this property is load-bearing, not a nicety.
        text: hidden at anchor_idx, the last real caption token. It is the
            position whose label is <|endoftext|>, so its context is the whole
            caption.

        slot count comes from no_image_emb so there is still one source of truth
        for 49 instead of a fourth copy.
        """
        slots = self.no_image_emb.size(1)
        img_feat = hidden[:, :slots, :].mean(dim=1)
        txt_feat = hidden[torch.arange(hidden.size(0), device=hidden.device), anchor_idx]
        return self.con_img(img_feat), self.con_txt(txt_feat)

    def infonce(self, img_feat, txt_feat):
        """Symmetric in-batch InfoNCE.

        Every other sample in the batch is a negative, so the style/format shared
        by all LAION captions contributes an equal amount to every entry of the
        similarity logits and cancels out of the softmax -- what is left to
        optimize is only what distinguishes this image's caption from the others.
        """
        img_feat = nn.functional.normalize(img_feat, dim=-1)
        txt_feat = nn.functional.normalize(txt_feat, dim=-1)
        scale = self.logit_scale.clamp(min=LOGIT_SCALE_MIN, max=LOGIT_SCALE_MAX).exp()
        logits = scale * (img_feat @ txt_feat.t())          # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (nn.functional.cross_entropy(logits, labels) +
                      nn.functional.cross_entropy(logits.t(), labels))
   
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.register_buffer('causal_mask', torch.triu(torch.ones(block_size-1, block_size-1) * float('-inf'), diagonal=1))
        # 第一层归一化
        self.ln1 = nn.LayerNorm(d_model) # 对每个样本做标准化，稳定训练
        # 手动多头自注意力（替换 nn.MultiheadAttention，MPS 上更快）
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        # 第二层归一化
        self.ln2 = nn.LayerNorm(d_model)
        # 前馈网络（两层 MLP）FFN 的作用：对每个 token 独立做非线性变换，增强表达能力。单个 token 内部处理（"提取更复杂的特征"）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), # 第 1 层
            nn.GELU(),                       # 激活函数
            nn.Linear(d_model * 4, d_model) # 第 2 层
        ) # 结构：384 → 1536 → 384（先扩展 4 倍再压缩）

    def forward(self, x):
        B, T, C = x.shape
        # Attention 部分
        normed = self.ln1(x)                    # 先归一化
        qkv = self.qkv(normed)                  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)          # 各 (B, T, C)
        # 拆多头: (B, T, C) → (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # 计算 attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, T)
        attn = attn + self.causal_mask[:T, :T]         # causal mask
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v                                  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # 合并多头
        out = self.attn_out(out)
        x = x + out                                     # 残差连接

        # FFN 部分
        x = x + self.ffn(self.ln2(x))           # 归一化 → FFN → 残差
        return x


