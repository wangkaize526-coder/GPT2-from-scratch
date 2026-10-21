"""
Build the cat/dog pet dataset used by the VQA-style loader.

Each sample: {"question", "answer", "image_path"}.
image_path is stored relative to the project root, e.g.
    data/oxford-iiit-pet/images/Abyssinian_100.jpg

Run from anywhere:
    python script/preprocess_Cat_Dog.py
"""

import json
import os
import random

from torchvision import datasets

# Locate the project root from this file's own location, so the script
# behaves the same no matter which directory it is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT = os.path.join(DATA_DIR, "pet", "cat_dog.json")

# root=DATA_DIR -> torchvision extracts into data/oxford-iiit-pet/
# download=True -> skips the download when data/oxford-iiit-pet already exists.
ds = datasets.OxfordIIITPet(root=DATA_DIR, split="trainval", target_types=["category", "binary-category"],download=True)

sample_list = []
q_cat_pool = [
    "What is this?",
    "What breed is this?",
    "Which breed is this?",
    "What kind of cat is this?",
    "Can you identify this cat?",
    "What type of cat is this?",
    "Identify this cat breed.",
    "What cat breed is in the image?",
    "Tell me this cat's breed.",
    "Do you know this cat breed?",
    "What breed of cat is this?",
    "What is the cat breed?",
    "Which cat breed is this?",
    "What is the breed of this cat?",
    "What is this?",
    "What breed is this?",
    "Which breed is shown?",
    "What animal breed is this?",
    "Can you tell the breed?",
    "What is the breed of the cat?",
    "Identify the cat.",
    "What breed does this cat belong to?",
    "What is the cat's breed?",
    "Guess this cat's breed.",
    "What kind of breed is this cat?"
]

a_cat_pool = [
    "It is a {class_name} cat.",
    "It is a {class_name} cat.",
    "It is a {class_name} cat.",
    "It is a {class_name} cat.",
    "It is a {class_name} cat.",
    "It is a {class_name} cat.",
    "It is a cat of the {class_name} breed.",
    "The image shows a {class_name} cat.",
    "This cat is a {class_name}.",
    "It is a {class_name} cat.",
    "This is a cat of the {class_name} breed.",
    "The breed is {class_name}.",
    "It is a {class_name} cat.",
    "This cat is a {class_name}.",
    "It is a {class_name} cat.",
    "It is a {class_name} cat.",
    "The image shows a {class_name} cat.",
    "It is a {class_name} cat.",
    "It is a {class_name} cat.",
    "The cat is a {class_name}.",
    "It is a {class_name} cat.",
    "It belongs to the {class_name} breed.",
    "It is a {class_name} cat.",
    "It looks like a {class_name} cat.",
    "It is a {class_name} cat."
]

q_dog_pool = [
    "What is this?",
    "What breed is this?",
    "Which breed is this?",
    "What kind of dog is this?",
    "Can you identify this dog?",
    "What type of dog is this?",
    "Identify this dog breed.",
    "What dog breed is in the image?",
    "Tell me this dog's breed.",
    "Do you know this dog breed?",
    "What breed of dog is this?",
    "What is the dog breed?",
    "Which dog breed is this?",
    "What is the breed of this dog?",
    "What is this?",
    "What breed is this?",
    "Which breed is shown?",
    "What animal breed is this?",
    "Can you tell the breed?",
    "What is the breed of the dog?",
    "Identify the dog.",
    "What breed does this dog belong to?",
    "What is the dog's breed?",
    "Guess this dog's breed.",
    "What kind of breed is this dog?"
]

a_dog_pool = [
    "It is a {class_name} dog.",
    "It is a {class_name} dog.",
    "It is a {class_name} dog.",
    "It is a {class_name} dog.",
    "It is a {class_name} dog.",
    "It is a {class_name} dog.",
    "It is a dog of the {class_name} breed.",
    "The image shows a {class_name} dog.",
    "This dog is a {class_name}.",
    "It is a {class_name} dog.",
    "This is a dog of the {class_name} breed.",
    "The breed is {class_name}.",
    "It is a {class_name} dog.",
    "This dog is a {class_name}.",
    "It is a {class_name} dog.",
    "It is a {class_name} dog.",
    "The image shows a {class_name} dog.",
    "It is a {class_name} dog.",
    "It is a {class_name} dog.",
    "The dog is a {class_name}.",
    "It is a {class_name} dog.",
    "It belongs to the {class_name} breed.",
    "It is a {class_name} dog.",
    "It looks like a {class_name} dog.",
    "It is a {class_name} dog."
]

# Fixed seed so the generated dataset is reproducible.
rng = random.Random(42)

for idx in range(len(ds)):
    pil_img, (breed_label, binary_label) = ds[idx]
    class_name = ds.classes[breed_label]
    species= "cat" if binary_label == 0 else "dog"

    # ds._images[idx] is ALREADY the full path
    # (DATA_DIR/oxford-iiit-pet/images/<file>.jpg) -- do NOT join root again.
    # Store it relative to the project root so the JSON stays portable.
    img_path = os.path.relpath(ds._images[idx], PROJECT_ROOT)

    # draw one question and one answer template at random for this image
    if species == "cat":
        question = rng.choice(q_cat_pool)
        answer = rng.choice(a_cat_pool).format(class_name=class_name)
    else:
        question = rng.choice(q_dog_pool)
        answer = rng.choice(a_dog_pool).format(class_name=class_name)

    sample_list.append({
        "question": question,
        "answer": answer,
        "species": species,
        "class_name": class_name,
        "image_path": img_path,
    })

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(sample_list, f, ensure_ascii=False, indent=4)

print(f"Wrote {len(sample_list)} samples to {OUTPUT}")
