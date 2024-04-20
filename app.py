import gradio as gr
import chromadb
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os

print("Loading model...")
device = 'cpu'
model = CLIPModel.from_pretrained("./CLIP-VIT").to(torch.bfloat16).to(device)
processor = CLIPProcessor.from_pretrained("./CLIP-VIT")
print("Model loaded!")

client = chromadb.PersistentClient('img_db/')

def search(query):
    collection = client.get_collection('images')
    with torch.no_grad():
        text_emb = model.get_text_features(**processor.tokenizer(query, return_tensors='pt').to(device))
    results = collection.query(
        query_embeddings=text_emb.cpu().squeeze(0).tolist(),
        n_results=4  # top n results
    )
    # Create a gallery of the top 5 results
    gallery_images = [Image.open(doc) for doc in results['documents'][0]]
    return gallery_images

demo = gr.Interface(
    fn=search,
    inputs=gr.Textbox(placeholder="Enter a query"),
    outputs=gr.Gallery(label="Results",selected_index=0,preview=True),
    title="Where's My Pic?",
    description="A local image search engine powered by CLIP!",
    theme = gr.themes.Default(primary_hue="purple")
)

demo.launch()