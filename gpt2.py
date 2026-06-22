from datasets import load_dataset
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import TensorDataset, DataLoader
# 加载数据集

def create_dataset():
    # 加载数据集
    data = load_dataset("wikitext", "wikitext-2-raw-v1")
    # 用gpt-2的分词规则
    encoder= tiktoken.get_encoding("gpt2")
    # tokens= encoder.encode("Hello world") # 对Hello world进行分词: [15496, 995]
    all_text = "\n".join(data['train']['text']) # all_text是所有训练文本的拼接，很长的字符串
    # 对所有训练文本进行分词, 并且返回对应的数字列表
    tokens= encoder.encode(all_text) 
    # tokens是一个很长的数字列表, 每个数字对应一个字符, 例如: [15496, 995, 15496, 995, ...]
    # 现在要把tokens切分为固定长度的很多块
    block_size = 256 # 每个块的长度
    n= len(tokens) // block_size # 计算可以切分为多少个块
    tokens = tokens[:n * block_size] # 只取前n个块, 每个块的长度都是block_size，多出来的部分不管了，失去几百个没事
    # 对tokens进行 tensor化, n行block_size列，每个元素是一个字符
    tokens = torch.tensor(tokens).view(n, block_size)
    # 设置输入和输出
    # y 是 x 往后移一位（预测下一个 token）
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    # dataloader来加载数据
    dataset = TensorDataset(x, y)
    '''
        - dataset：你的 TensorDataset，包含所有 (x, y) 对
        - batch_size=32：每次取 32 个样本一起训练（不是一次一个，而是 32 个并行计算）
        - shuffle=True：每个 epoch 重新打乱顺序，防止模型学到固定顺序
        取 32 个 (x, y) → 喂给模型 → 计算 loss → 反向传播 → 更新参数
    '''
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    return dataloader

class GPT2(nn.Module):
    '''
        GPT-2 参数（基于你的硬件调整的）：
        - vocab_size=50257：词表大小，GPT-2 tokenizer 有 50257 个 token
        - d_model=384：每个 token 的向量维度，越大模型越强，但越慢
        - n_heads=6：注意力头数，d_model 必须能被 n_heads 整除（384÷6=64）
        - n_layers=6：Transformer block 的层数，越多越强
        - block_size=256：最大序列长度，一次最多看 256 个 token（要和 dataloader 中的 block_size 一致）

        关系：
            vocab_size × d_model → Embedding 把 token ID 变成 384 维向量
            d_model ÷ n_heads = 64 → 每个头关注 64 维
            n_layers × (Attention + FFN) → 堆叠加深理解

        不一样：
        - vocab_size：tokenizer 有多少个不同的 token（GPT-2 是 50257 个）
        - *len(tokens)*：你的训练数据总共有多少个 token（2,403,644 个）
        类比：vocab_size 是字典有多少字，len(tokens) 是你写了多少字。
        用 encoder.n_vocab 获取：
        vocab_size = encoder.n_vocab  # 50257
    '''
    def __init__(self, vocab_size=50257, d_model=384, n_heads=6, n_layers=6, block_size=256):
        super().__init__()
        self.block_size=block_size

        self.token_emb = nn.Embedding(vocab_size,d_model)
        '''
            位置 embedding 是告诉模型"这个 token 在第几个位置"。
            输入序列: [token_0, token_1, token_2, ..., token_255]
            位置索引: [    0,       1,       2,   ...,      255]
            所以你需要 256 个位置向量，每个 384 维：
            pos_emb: (256, 384)
            位置 0 → [0.12, -0.34, ..., 0.56]  # 384维
            位置 1 → [0.78,  0.23, ..., -0.11]
            ...
            位置 255 → [...]
            和 token_emb 的区别：
            - token_emb(50257, 384)：按 token ID 查
            - pos_emb(256, 384)：按位置索引查
        '''
        self.pos_emb = nn.Embedding(block_size,d_model)
        
        self.layers= nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TransformerBlock(d_model, n_heads, block_size))

        self.ln = nn.LayerNorm(d_model)    # 最后的归一化
        self.head = nn.Linear(d_model, vocab_size)  # 输出 logits

    def forward(self, x):
        # x shape: (batch, seq_len) = (32, 255)，即 32 个样本，每个 255 个 token
        tok_emb = self.token_emb(x)      
        # 把每个 token ID 变成 384 维向量
        # (32, 255) → (32, 255, 384)        
              
        pos = torch.arange(x.size(1), device=x.device) 
        # 生成位置索引 [0, 1, 2, ..., 254]
        # device=x.device 确保在同一个设备上（CPU/GPU）

        pos_emb = self.pos_emb(pos)    
        # 查表得到位置向量
        # (32, 255) → (32, 255, 384)      
                   
        x = tok_emb + pos_emb  
        # 广播相加：(32, 255, 384) + (255, 384) → (32, 255, 384)
        # 每个 token 现在既有语义信息，也有位置信息
        
        for layer in self.layers:
            x = layer(x)
        # 依次通过 6 个 TransformerBlock，每层提取更复杂的特征

        x = self.ln(x)       # 最后归一化
        logits = self.head(x) 
        # 线性层：(32, 255, 384) → (32, 255, 50257)
        # 每个位置输出 50257 个分数，表示下一个 token 是词表中每个词的概率
        return logits
    
    def generate(self, prompt, device, max_new_tokens=100, temperature=0.8):
        self.eval()
        encoder = tiktoken.get_encoding("gpt2")
        tokens = encoder.encode(prompt)

        for _ in range(max_new_tokens):
            x = torch.tensor([tokens], device=device)  # (1, seq_len)
            logits = self(x)                           # (1, seq_len, vocab)
            logits = logits[:, -1, :] / temperature     # 只取最后一个位置
            probs = torch.softmax(logits, dim=-1)       # 变成概率
            next_token = torch.multinomial(probs, 1)    # 采样
            tokens.append(next_token.item())
    
        return encoder.decode(tokens)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size=256):
        super().__init__()
        self.register_buffer('causal_mask', torch.triu(torch.ones(block_size-1, block_size-1) * float('-inf'), diagonal=1))
        # 第一层归一化
        self.ln1 = nn.LayerNorm(d_model) # 对每个样本做标准化，稳定训练
        # 多头自注意力
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True) # batch_first=True 表示输入 shape 是 (batch, seq, feature) 注意：nn.MultiheadAttention 需要 causal mask 才能防止看到未来的 token，后面加
        # 第二层归一化
        self.ln2 = nn.LayerNorm(d_model)
        # 前馈网络（两层 MLP）FFN 的作用：对每个 token 独立做非线性变换，增强表达能力。单个 token 内部处理（"提取更复杂的特征"）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), # 第 1 层
            nn.GELU(),                       # 激活函数
            nn.Linear(d_model * 4, d_model) # 第 2 层
        ) # 结构：384 → 1536 → 384（先扩展 4 倍再压缩）
    
    def forward(self, x):
        # Attention 部分
        normed = self.ln1(x)                    # 先归一化
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=self.causal_mask[:x.size(1), :x.size(1)]) # 三个参数分别是 Q（查询）、K（键）、V（值）。一次完成所有计算，反正计算公式都一样，用 _ 表示不需要这个值
        x = x + attn_out                        # 残差连接（加回去）
        
        # FFN 部分
        x = x + self.ffn(self.ln2(x))           # 归一化 → FFN → 残差
        return x

def train():
    dataloader= create_dataset()

    # 2. 创建模型，自动选择可用设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple MPS")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    model = GPT2().to(device)

    # 3. 定义损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=3e-4)
    # 4. 训练循环
    epochs= 3
    for epoch in range(epochs):
        total_loss= 0
        print(epoch)
        for x,y in dataloader:
            x, y = x.to(device), y.to(device)

            logits= model(x)
            loss= criterion(logits.view(-1, 50257), y.view(-1)) # 注意 logits.view(-1, 50257) 是把 3D 变成 2D，因为 CrossEntropyLoss 需要 (batch*seq, vocab_size) 的格式。

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    # 5. 保存模型
    torch.save(model.state_dict(), "gpt2_model.pth")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train()
    model = GPT2()
    model.load_state_dict(torch.load("gpt2_model.pth"))
    model.to(device)
    print(model.generate("Hello", device))

