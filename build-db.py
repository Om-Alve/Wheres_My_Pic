import chromadb
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
import time
import argparse

print("Loading model...")
device = 'cuda'
model = CLIPModel.from_pretrained("./CLIP-VIT").to(torch.bfloat16).to(device)
processor = CLIPProcessor.from_pretrained("./CLIP-VIT")
print("Model loaded!")

def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor.image_processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        return image_features.to(torch.float16).cpu().squeeze(0).numpy().tolist()

client = chromadb.PersistentClient('img_db/')

def create_db(folder, collection):
    img_embeddings = []
    filenames = []
    start_time = time.time()
    for image in os.listdir(folder):
        filename = os.path.join(folder, image)
        img = Image.open(filename)
        img_embeddings.append(extract_features_clip(img))
        filenames.append(filename)
    collection.add(
        embeddings=img_embeddings,
        documents=filenames,
        ids=filenames,
    )
    end_time = time.time()
    print(f"Database updated for {folder} successfully!")
    print(f"Time taken: {(end_time - start_time) * 1000} milliseconds")

def update_db(folders, collection):
    start_time = time.time()
    current_files = []
    for folder in folders:
        current_files.extend([os.path.join(folder, image) for image in os.listdir(folder) if os.path.isfile(os.path.join(folder, image))])

    existing_files = collection.get()['documents']
    new_files = [file for file in current_files if file not in existing_files]
    deleted_files = [file for file in existing_files if file not in current_files]

    if len(new_files) == 0 and len(deleted_files) == 0:
        print(f"Nothing to update for {', '.join(folders)}!")
        return

    for file in deleted_files:
        idx = existing_files.index(file)
        collection.delete(ids=[file])

    for i, file in enumerate(new_files):
        img = Image.open(file)
        img_embeddings = extract_features_clip(img)
        collection.upsert(
            embeddings=[img_embeddings],
            documents=[file],
            ids=[file]
        )

    end_time = time.time()
    print("Database updated successfully!")
    print(f"Time taken: {(end_time - start_time) * 1000} milliseconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index image folders')
    parser.add_argument('--add', nargs='+', help='Create a new database with the specified folders')    
    parser.add_argument('--update', action='store_true', help='Update the existing database')
    args = parser.parse_args()
    if len(client.list_collections()) == 0:
        # if the database hasn't been created before
        collection = client.create_collection(
            name="images", metadata={"hnsw:space": "cosine"}
        )
        for folder in args.add:
            create_db(folder, collection)
        print("Created the database!")
        with open('img_db/indexed_folders.txt', 'w') as f:
            f.write(f"{'\n'.join(args.add)}") 
        print(f"Added Folders {args.add} successfully!")
        print(f"Total documents indexed: {len(collection.get()['documents'])}")  
    elif args.add:
        # if the db is already created and want to add new folders
        collection = client.get_collection('images')
        with open('img_db/indexed_folders.txt', 'r') as f:
            indexed_folders = f.read().split('\n')
        folders = set(args.add).difference(set(indexed_folders))
        for folder in folders:
            create_db(folder, collection)
        print(f"Added Folders {args.add} successfully!")
        print(f"Total documents indexed: {len(collection.get()['documents'])}")  
        with open('img_db/indexed_folders.txt', 'a') as f:
            f.write(f"\n{'\n'.join(args.add)}")    
    else:
        # if the db is already created and want to update the existing folders
        collection = client.get_collection('images')
        with open('img_db/indexed_folders.txt', 'r') as f:
            indexed_folders = f.read().split('\n')
        update_db(indexed_folders, collection)
        print(f"Total documents indexed: {len(collection.get()['documents'])}")  

