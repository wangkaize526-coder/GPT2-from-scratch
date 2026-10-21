import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from Train_and_Eval import train_and_eval
import model.generator as generator

def main():
    # read the config file to prepare to train
    config_file_path = input("input config file path: ")
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")             # Apple Silicon 的 GPU（Metal）
        print("使用 GPU: Apple MPS (Metal)")
    else:
        device = torch.device("cpu")

    train_and_eval(config["train"]["epoch"], config["train"]["learning_rate"],config["train"]["K"])


    # both calls return the generated string - print it, otherwise a run that
    # reaches this point produces no visible output at all
    answer = generator.generate("./test3.png", "Tell me what inside the picture", device)
    print(f"[vision] Tell me what inside the picture -> {answer!r}")

    continuation = generator.generate_text("Beijing", device)
    print(f"[text] Beijing -> {continuation!r}")


if __name__ == "__main__":
    main()
