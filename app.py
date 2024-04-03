import gradio as gr
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

client = chromadb.PersistentClient('img_db/')
collection = client.get_collection('images')


def search(query):
    with torch.no_grad():
        text_emb = model.get_text_features(**processor.tokenizer(query, return_tensors='pt').to('cuda'))
    results = collection.query(
        query_embeddings=text_emb.cpu().squeeze(0).tolist(),
        n_results=5, # top n results
    )
    return Image.open(results['documents'][0][0])

interface = gr.Interface(
    fn=search,
    inputs=gr.Textbox(lines=1, placeholder="Enter your text query..."),
    outputs=gr.Image(type="pil"),
    title="Local Image Search Engine",
    description="Search for images in your local folder using text queries.",
)

interface.launch(debug=True)