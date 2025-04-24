import torch
from diffusers import StableDiffusionPipeline
from langchain.tools import Tool
import gradio as gr
import PyPDF2
import os
from PIL import Image
import nltk
from nltk.corpus import stopwords

# Download NLTK stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Set memory management option to avoid CUDA memory issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
).to(device)
print(f"Using device: {device}")

# Step 1: Parse PDF and Extract Words (Exclude Stop Words)
def parse_pdf(pdf_file):
    words = set()  # Use a set to avoid duplicates
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text = page.extract_text()
        # Split text into words, remove special characters, and filter short words
        page_words = [word.strip().lower() for word in text.split() if len(word.strip()) > 2]
        # Exclude stop words
        page_words = [word for word in page_words if word not in stop_words]
        words.update(page_words)
    return list(words)

# Step 2: Generate Image for a Word
def generate_image(word: str):
    """Generate an image using Stable Diffusion."""
    # Adjusted prompt for more realistic output
    prompt = f"A highly realistic depiction of {word}, photorealistic, 4k resolution"
    try:
        with torch.inference_mode():
            torch.cuda.empty_cache()  # Clear memory before generation
            image = pipe(prompt, num_inference_steps=50).images[0]  # Increased to 50 for better quality
        return image
    except Exception as e:
        print(f"Error generating image for {word}: {str(e)}")
        return None

# Step 3: Process PDF and Generate Images
def process_pdf_and_generate_images(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF file.", []

    # Parse PDF to extract words (excluding stop words)
    words = parse_pdf(pdf_file)
    print(f"Extracted {len(words)} unique words (excluding stop words): {words[:10]}...")  # Print first 10 words

    # Generate images for each word and collect them with labels
    image_list = []
    for word in words:  # Process all words (no limit)
        print(f"Generating image for word: {word}")
        image = generate_image(word)
        if image:
            # Add the image and its label (word) to the list
            image_list.append((image, word))
            print(f"Generated image for word: {word}")
        else:
            print(f"Skipping word {word} due to generation error")
        # Clear memory after each image
        torch.cuda.empty_cache()

    return "Image generation complete!", image_list

# Step 4: Gradio Interface to Upload PDF and Display Images
def gradio_interface(pdf_file):
    message, images_with_labels = process_pdf_and_generate_images(pdf_file)
    return message, images_with_labels

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.File(label="Upload PDF"),
    outputs=[
        "text",
        gr.Gallery(label="Generated Images with Word Labels")  # Display images in a 2-column grid
    ],
    title="PDF to Word Images Generator",
    description="Upload a PDF file to extract words (excluding stop words) and generate realistic images for each word. Images will be displayed with their corresponding word labels."
)

if __name__ == "__main__":
    iface.launch(share=True)  # Set share=False for local Jupyter Notebook
