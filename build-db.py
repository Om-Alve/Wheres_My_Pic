import argparse
import os
import time
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import chromadb

torch.set_float32_matmul_precision("high")

print("Loading model...")
device = "cpu" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
model = CLIPModel.from_pretrained("./CLIP-VIT", torch_dtype=dtype).to(device)
processor = CLIPProcessor.from_pretrained("./CLIP-VIT")
print("Model loaded!")

client = chromadb.PersistentClient("img_db/")


def print_gpu_usage():
    if device == "cuda":
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    else:
        print("GPU not available")


def extract_features_clip(image_paths, batch_size=32):
    with torch.no_grad():
        all_features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)
            features = image_features.cpu().to(torch.float16).numpy()
            all_features.extend(features)
        return np.array(all_features).tolist()


def process_images(folders, collection, mode="add"):
    start_time = time.time()
    print_gpu_usage()

    all_files = []
    for folder in folders:
        all_files.extend(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
            ]
        )

    batch_size = 128
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i : i + batch_size]
        batch_embeddings = extract_features_clip(batch_files)

        if mode == "add":
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_files,
                ids=batch_files,
            )
        elif mode == "update":
            existing_files = set(collection.get(ids=batch_files)["documents"])
            new_files = [file for file in batch_files if file not in existing_files]
            deleted_files = [file for file in existing_files if file not in batch_files]

            if deleted_files:
                collection.delete(ids=deleted_files)

            if new_files:
                new_embeddings = [
                    emb
                    for file, emb in zip(batch_files, batch_embeddings)
                    if file in new_files
                ]
                collection.upsert(
                    embeddings=new_embeddings, documents=new_files, ids=new_files
                )

    end_time = time.time()
    print_gpu_usage()
    print(
        f"Database {'updated' if mode == 'update' else 'created'} for {', '.join(folders)} successfully!"
    )
    print(f"Time taken: {(end_time - start_time):.2f} seconds")


def ensure_file_exists(filepath):
    if not os.path.isfile(filepath):
        open(filepath, "a").close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index image folders")
    parser.add_argument(
        "--add", nargs="+", help="Create a new database with the specified folders"
    )
    parser.add_argument(
        "--update", action="store_true", help="Update the existing database"
    )
    args = parser.parse_args()

    ensure_file_exists("img_db/indexed_folders.txt")
    collection_name = "images"

    if not client.list_collections():
        collection = client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        if args.add:
            process_images(args.add, collection, mode="add")
            with open("img_db/indexed_folders.txt", "w") as f:
                f.write("\n".join(args.add))
        else:
            print("No folders to add!")

    elif args.add:
        collection = client.get_collection(collection_name)
        with open("img_db/indexed_folders.txt", "r") as f:
            indexed_folders = set(f.read().splitlines())
        new_folders = set(args.add) - indexed_folders
        if new_folders:
            process_images(new_folders, collection, mode="add")
            with open("img_db/indexed_folders.txt", "a") as f:
                f.write("\n" + "\n".join(new_folders))

    elif args.update:
        collection = client.get_collection(collection_name)
        with open("img_db/indexed_folders.txt", "r") as f:
            indexed_folders = f.read().splitlines()
        if indexed_folders:
            process_images(indexed_folders, collection, mode="update")

    else:
        print("No action specified.")
