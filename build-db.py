import chromadb
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os

print("Loading model...")

model = CLIPModel.from_pretrained("./CLIP-VIT").to('cuda')
processor = CLIPProcessor.from_pretrained("./CLIP-VIT")

print("Model loaded!")

def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor.image_processor(images=image, return_tensors="pt").to('cuda')
        image_features = model.get_image_features(**inputs)
        return image_features

client = chromadb.PersistentClient('img_db/')

def create_db(folder,collection):
    img_embeddings = []
    filenames = []
    for image in os.listdir(folder):
        filename = os.path.join(folder,image)
        img = Image.open(filename)
        img_embeddings.append(extract_features_clip(img).cpu().squeeze(0).numpy().tolist())
        filenames.append(filename)
    
    collection.add(
        embeddings = img_embeddings, # embeddings are used to search through the documents
                                          # we pass in the previously calculated embeddings
        documents=filenames, # we send in the titles as the documents
        ids=filenames, # must be unique for each doc
    )
    print("Database updated successfully!")


def update_db(folder, collection):
    current_files = [os.path.join(folder, image) for image in os.listdir(folder) if os.path.isfile(os.path.join(folder, image))]

    existing_files = collection.get()['documents']
    new_files = [file for file in current_files if file not in existing_files]
    deleted_files = [file for file in existing_files if file not in current_files]
    if len(new_files) == 0 and len(deleted_files) == 0:
        print("Nothing to update!")
        return
    for file in deleted_files:
        idx = existing_files.index(file)
        collection.delete(ids=[file])

    for i,file in enumerate(new_files):
        img = Image.open(file)
        img_embeddings = extract_features_clip(img).cpu().squeeze(0).numpy().tolist()
        collection.upsert(
            embeddings=[img_embeddings],
            documents=[file],
            ids=[file]
        )
    

    print("Database updated successfully!")


if len(client.list_collections())  != 0:
    print("Database exists, Updating the database...")
    collection = client.get_collection('images')
    update_db('Image-Folder',collection)
    print(len(collection.get()['documents']))
else:
    collection =  client.create_collection(
        name="images",
        metadata={"hnsw:space": "cosine"}
    )
    create_db('Image-Folder',collection)
    print("Created the database!")
