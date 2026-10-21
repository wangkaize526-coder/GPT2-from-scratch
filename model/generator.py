import torch
from torch.utils.data import DataLoader, TensorDataset
import tiktoken
from model.gpt2 import GPT2
import torchvision.transforms as transforms
from PIL import Image


@torch.no_grad()
def generate( image_path, question, device, max_new_tokens=30, temperature=0.8):
    """
    image:    a preprocessed image tensor, [3, 224, 224] or [1, 3, 224, 224].
    question: the question string, e.g. "What breed is this?".
    returns:  the generated answer string.

    The frame must match training:
        49 image slots + <|endoftext|> question # answer <|endoftext|>
    The 49 slots hold placeholder ids that forward() overwrites with the
    vision-encoder output. Generation must prepend them too, otherwise the
    visual features would overwrite the first 49 *text* tokens.
    """
    model= GPT2().to(device)
    model.load_state_dict(torch.load("./check_point/gpt2_model_best.pth", map_location=device))
    
    model.eval()
    encoder = tiktoken.get_encoding("gpt2")
    eot = encoder.eot_token

    # load the image
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
    img_pil= Image.open(image_path).convert("RGB")
    img_tensor= clip_transform(img_pil).unsqueeze(0).to(device)
    
    prefix_ids = encoder.encode(f"<|endoftext|>{question}#", allowed_special="all")
    tokens = [0] * 49 + prefix_ids                     # 49 slots, overwritten in forward()

    for _ in range(max_new_tokens):
        x = torch.tensor([tokens], device=device)      # (1, seq_len)
        logits = model(x, img_tensor)                        # forward() also needs the image
        logits = logits[:, -1, :] / temperature        # only the last position
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        if next_token == eot:                          # stop at <|endoftext|>
            break
        tokens.append(next_token)

    answer_ids = tokens[49 + len(prefix_ids):]
    return encoder.decode(answer_ids)
# this is for pure text generation
@torch.no_grad()
def generate_text(prompt, device, max_new_tokens=80, temperature=0.8):
    model= GPT2().to(device)
    model.load_state_dict(torch.load("./check_point/gpt2_model_best.pth", map_location=device))
    
    model.eval()
    encoder = tiktoken.get_encoding("gpt2")
    eot = encoder.eot_token
    # Same frame as training: 49 image slots FIRST, text after. forward() overwrites
    # those slots with no_image_emb when img is None, so the text must start at
    # position 49 - not at 0, and the sequence must be at least 49 long or the
    # slice assignment in forward() fails outright (this is why the previous
    # version crashed on any prompt shorter than 49 tokens).
    prefix_ids = encoder.encode(f"<|endoftext|>{prompt}", allowed_special="all")
    tokens = [0] * 49 + prefix_ids
    n_prompt = len(tokens)                 # includes the 49 slots
    for _ in range(max_new_tokens):
        x = torch.tensor([tokens], device=device)
        logits = model(x)[:, -1, :] / temperature          # img=None -> pure text path
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        if next_token == eot:
            break
        tokens.append(next_token)
    return encoder.decode(tokens[n_prompt:])
