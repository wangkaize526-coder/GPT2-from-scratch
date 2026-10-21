
> 工作日志（中文）。这是踩坑记录，不是项目介绍 —— 项目介绍看根目录的 `README.md`。
> 按时间顺序展开：每个阶段当时以为的问题、真正的问题、怎么发现的、最后怎么改的。
> 注意：这一份是 10-18 前后的版本，比完整的九阶段版本短一些（那一版在同步服务器文件时被覆盖了）。

全貌看完了。你要的三块我都答，但先说一个会浪费你几周的前提错误。
    
    ═══════════════════════════════════
    一、先纠正前提：你把三个阶段的边界搞混了
    ═══════════════════════════════════
    
    ① VQA 不是「图文对齐 pretraining」。
    对齐用的数据是「图 + caption（图像描述）」，训练目标是描述图的下一句话（或对比学习把图文拉到同一空间）。你现在 prompt_vision.py 做的是 question→answer，那是 SFT。
    
    ② 更严重的：你的数据量做不了对齐。
    
        Oxford-IIIT Pet 训练图 = 3,312 张
    
    对齐 pretraining 的量级是 CC3M / CC12M / COCO Captions / Flickr30k —— 几十万到千万级图文对。你现在差 2–3 个数量级。这不是代码问题，是数据问题：不换数据，无论怎么调代码，视觉板块都不会真的「对齐」。
    
    所以你的路线图要改成：
    
        阶段0  文本 pretraining（wiki，39M token）        ← 你有数据
        阶段1  图文对齐 pretraining（caption 数据集）      ← 数据还没有，必须补
        阶段2  SFT（VQA + chat，同一个模型）              ← 语料还没有
        阶段3  Streamlit UI
    
    ③ 你现在的顺序是 SFT 跳在 pretraining 前面，所以模型没知识基础、SFT 只能学到「输出格式」。你自己察觉到了，判断是对的：先把 train("1") 那条 VQA 训练停掉，代码留着，别跑。
    
    ═══════════════════════════════════
    二、文件排布
    ═══════════════════════════════════
    
    现状问题（按严重度）：
    
    1. gpt2.py 313 行扛了 5 件事：模型定义、两个 generate、训练循环、CLI 入口、checkpoint 存取。加了 pretrain/SFT/UI 之后它会变成 1000 行。
    2. train(signal) 用 if signal=="0"/"1" 选数据集 —— 你现在 2 个阶段还好，加到 4–5 个阶段就是一堆嵌套 if，改一处崩一处。
    3. 49 硬编码在 4 个文件里：vision_encoder.py（注释×3，真值由 (224//32)**2 算出）、prompt_vision.py（2 处 49）、prompt_wiki.py（SLOTS=49）、gpt2.py（no_image_emb、tok_emb[:, :49]、[0]49、tokens[49+...]、n_img_patch）。真相源只有一个（patch 数），却抄了 4 份 —— 你哪天改图像分辨率或 patch_size，一定漏。
    4. 50257 硬编码在 gpt2.py 的 4 个 loss 调用里（第 226/238/253/261 行）。词表一变（下面第四节会讲），4 处都得改。
    5. input.py 是死代码 —— 没有任何文件 import 它，GPT2 用的是 nn.Embedding。删掉。
    6. README.md 过时：写 d_model=384 / WikiText-2，实际是 512 / wiki。
    
    建议按「职责」分层（不是为了好看，是为了每加一个阶段只动一层）：
    
        config.py              # 所有常量 + 架构版本号, 唯一真相源
        model/
          gpt2.py              # 只有 GPT2 backbone
          blocks.py            # TransformerBlock
          vision_encoder.py    # 从这里导出 N_SLOTS, 别处都 import 它
        data/
          wiki.py              # 原 prompt_wiki.py
          caption.py           # 阶段1 的图文对（新）
          vqa.py sft.py        # 阶段2
          sample.py            # 「样本 → x/y + loss mask」唯一一份
        train/
          loop.py              # 共用的 train_step / eval / 存 checkpoint
          pretrain_text.py
          pretrain_align.py
          sft.py
        inference/
          generate.py          # generate / generate_text / chat
        app/
          streamlit_app.py
        scripts/
          preprocess_*.py
        checkpoints/
    
    原则一句话：UI 只能调 inference 层，绝对不能直接碰模型内部。现在 main 里混着 input()、数据加载、generate —— 这个结构搬到 Streamlit 上会立刻失控。
    
    ═══════════════════════════════════
    三、训练的分布方式（最重要）
    ═══════════════════════════════════
    
    ① 不要串行分阶段，要混采。
    「先补完知识 → 再训视觉 → 再 SFT」这个顺序，后一阶段会把前一阶段覆盖掉（灾难性遗忘）。你 wiki 33.8 万条、vision 3 千条，差 100 倍，串行必崩。
    
    唯一的例外是 LLaVA 那套：冻结 backbone，只训一个投影层 —— 那种串行是成立的。但你这个模型太小，冻结 backbone 没意义，所以走混采。
    
    ② 配比用「token 预算」，不要用 epoch。
    1 个 wiki epoch = 100 个 vision epoch，「epoch」这个单位对两条数据完全不对等。改成：阶段 0 消耗 N 个 token，其中文本:对齐 = 7:3 这样定义。配比旋钮就是我之前给你的那个 K（每 K 个文本 step 插 1 个视觉 step）。
    
    ③ 必须做「遗忘检测」。
    每个 epoch 同时打三个数：wiki test perplexity、对齐 loss、VQA 准确率。只要文本 perplexity 因为视觉训练变差 → 配比错了。只盯一个数，你永远发现不了「字学会了图看丢了」或反过来。
    
    ④ 数据缺口要按顺序补：
    
        文本      wiki 39M token        ✓ 够（别再上 FineWeb，web 语料噪声大、和 gpt2 tokenizer 匹配也一般，先跑通再说）
        对齐      必须换/补 caption 数据集  ✗ 缺口最大
        SFT       还没有                 ✗
    
    ⑤ 评测集固定 + 存基线数字。 现在你每改一次架构（block_size 刚被我看到从 256 改成了 320），如果没有固定基线，你没法知道是变好还是变坏。
    
    ═══════════════════════════════════
    四、embedding 到底扩不扩
    ═══════════════════════════════════
    
    分三层，答案不同：
    
    A. 那 49 个模态槽 → 永远不要扩词表。
    它们是隐藏向量（nn.Parameter）。实测过：49 槽填 0 vs 填 9999，logits 完全相同 → 词表条目在这里永远查不到。这个结论不变。
    
    B. 语言 pretraining 阶段 → 不要扩。 保持 50257 和 gpt2 tokenizer 一致。
    
    C. 做 chat 的时候 → 要一次性决策，这是「改了就得重训」量级的事。
    
      选项1（零风险）：不扩。用现成 token 拼模板，比如
      ...\n\nUser: ...\n\nAssistant: ...<|endoftext|>
      可行，但你只有 <|endoftext|> 一个特殊 token，role 边界不够干净。
    
      选项2：扩，resize_token_embeddings(50257+3) 加 <|user|> <|assistant|> <|system|>。这是行业标准做法。代价：
      - state_dict 形状变 → 所有旧 checkpoint 全部作废
      - 新行必须用「已有 embedding 的均值」初始化，不能用随机
      - 词表变了，理论上该从头预训练（实践上「预训练完 → 扩词表 → 小 lr 再预训练一小段 → SFT」也work）
      - 只能扩一次，别反复 resize
    
    D. 比上面都更早的决策：tokenizer 本身。
    
        GPT-2 的 tokenizer 对中文极差（一个汉字 ≈ 2–3 token）
    
    你这个项目最终要做「chat」，如果你想让它输出中文，tokenizer 就得换（比如 Qwen 的 151k 词表）。而换 tokenizer = 全部数据重新 tokenize + 从头训。
    
    所以正确的决策顺序是：
    
        先定「要不要中文」 → 再定 tokenizer → 再定 vocab_size → 然后才动 pretraining
    
    这三步都是「定了就不能回头」的量级。越早定越省事 —— 你现在还没开始 pretrain，正是最好的时机。
    
    ═══════════════════════════════════
    五、两条「硬版本号」
    ═══════════════════════════════════
    
    架构上只有两个数改了就无法加载旧 checkpoint：
    
        vocab_size   → 决定 token_emb / head 形状   (SFT 扩词表会碰它)
        block_size   → 决定 pos_emb / causal_mask 形状
    
    顺便，你刚把 block_size 从 256 改成了 320 —— 这就是活生生的例子：现在 causal_mask 的 buffer 形状从 319×319 变成了别的，pos_emb 行数也变了，之前任何 checkpoint 都加载不进去了（会报 size mismatch）。
    
    建议：把架构配置和 state_dict 一起存进 checkpoint，并在文件名带上版本（ckpt_stage1_vocab50257_blocksize320.pt）。你现在存的是裸 state_dict，没有任何版本信息 —— 以后遇到「加载不进去但不知道为什么」会查很久。
    
    ═══════════════════════════════════
    六、散点
    ═══════════════════════════════════
    
    - train() 里的 input() 交互，阶段多了以后必须改成 CLI 参数/配置文件，否则无法复现、无法自动化、没法挂长时间训练。
    - 每个 checkpoint 配一张「模型卡」：哪个阶段、什么数据配比、多少 token、架构版本。不然一个月后你自己也分不清哪个是哪个。
    - 别急着写 Streamlit。先把 inference.generate() / inference.chat() 这两条接口做干净，UI 只是薄薄一层。
    - generate / generate_text 那个重名覆盖的坑（上一条）在拆分文件时顺手解决 —— 拆到 inference/generate.py 之后，共用核心 _sample 只有一份，就不会再各写各的了。
    
    ─────────────────────────────
    
    一句话总结优先级：先定 tokenizer（中英文）→ 决定 vocab 要不要扩 → 拆文件分层 → 补 caption 数据 → 混采跑 pretrain → 最后才是 SFT 和 UI。你现在卡在「SFT 跑在 pretrain 前面」，停掉它就行，代码不用删。
