import gradio as gr
import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif"]
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = CLIPModel.from_pretrained("./CLIP-VIT").to(device)
processor = CLIPProcessor.from_pretrained("./CLIP-VIT")
print("Model loaded!")

client = chromadb.PersistentClient("img_db/")


def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor.image_processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        return image_features.to(torch.float16).cpu().squeeze(0).numpy().tolist()


def create_db(folder, collection):
    img_embeddings = []
    filenames = []
    for image in os.listdir(folder):
        filename = os.path.join(folder, image)
        if os.path.isfile(filename) and any(
            filename.endswith(ext) for ext in IMAGE_EXTENSIONS
        ):
            img = Image.open(filename)
            img_embeddings.append(extract_features_clip(img))
            filenames.append(filename)
    collection.add(
        embeddings=img_embeddings,
        documents=filenames,
        ids=filenames,
    )


def update_db(folders, collection):
    current_files = []
    for folder in folders:
        current_files.extend(
            [
                os.path.join(folder, image)
                for image in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, image))
            ]
        )

    existing_files = collection.get()["documents"]
    new_files = [file for file in current_files if file not in existing_files]
    deleted_files = [file for file in existing_files if file not in current_files]

    for file in deleted_files:
        collection.delete(ids=[file])

    for file in new_files:
        if any(file.endswith(ext) for ext in IMAGE_EXTENSIONS):
            img = Image.open(file)
            img_embeddings = extract_features_clip(img)
            collection.upsert(embeddings=[img_embeddings], documents=[file], ids=[file])


def search(query):
    collection = client.get_collection("images")
    with torch.no_grad():
        text_emb = model.get_text_features(
            **processor.tokenizer(query, return_tensors="pt").to(device)
        )
    results = collection.query(
        query_embeddings=text_emb.cpu().squeeze(0).tolist(), n_results=4
    )
    gallery_images = [Image.open(doc) for doc in results["documents"][0]]
    return gallery_images


if __name__ == "__main__":
    demo = gr.Interface(
        fn=search,
        inputs=gr.Textbox(placeholder="Enter a query"),
        outputs=gr.Gallery(label="Results", selected_index=0, preview=True),
        title="Where's My Pic?",
        description="A local image search engine powered by CLIP!",
        theme=gr.themes.Default(primary_hue="purple"),
    )

    demo.launch(share=True)
