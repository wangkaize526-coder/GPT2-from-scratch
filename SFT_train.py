import torch
from model.gpt2 import GPT2
from torch.utils.data import DataLoader, TensorDataset
import prompt.prompt_vision

def SFT_train(epochs, lr):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model= GPT2().to(device)
    model.load_state_dict(torch.load("./checkpoint/gpt2_model_best.pth"),map_location=device)
    sft_dataloader= prompt.prompt_vision.build_phase0_dataloader()

    for epoch in range(epochs):
        model.train()
        



