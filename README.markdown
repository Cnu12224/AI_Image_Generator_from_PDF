# PDF to Word Images Generator

## Overview
This project extracts unique words from a PDF file (excluding common stop words) and uses the Stable Diffusion model to generate photorealistic images for each word. The generated images are displayed with their corresponding word labels using a Gradio interface.

## Features
- Extracts words from a PDF file using PyPDF2, excluding stop words with NLTK.
- Generates high-quality, photorealistic images for each word using Stable Diffusion (runwayml/stable-diffusion-v1-5).
- Displays the generated images in a gallery format with word labels via Gradio.
- Handles memory management for CUDA devices to prevent memory issues during image generation.

## Requirements
- Python 3.8+
- PyTorch
- Diffusers
- LangChain
- Gradio
- PyPDF2
- Pillow (PIL)
- NLTK

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download NLTK stop words:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage
1. Run the script:
   ```
   python script.py
   ```
2. A Gradio interface will launch in your browser.
3. Upload a PDF file using the interface.
4. The script will process the PDF, extract words, generate images, and display them in a gallery with their corresponding word labels.

## Code Structure
- **PDF Parsing**: Uses `PyPDF2` to extract text from PDF files and `NLTK` to filter out stop words.
- **Image Generation**: Utilizes the Stable Diffusion model (`runwayml/stable-diffusion-v1-5`) to generate photorealistic images.
- **Interface**: Employs `Gradio` to create a user-friendly interface for uploading PDFs and viewing generated images.

## Dependencies
- `torch`: For model inference and CUDA support.
- `diffusers`: For Stable Diffusion model.
- `langchain.tools`: For tool creation (though not fully utilized in this script).
- `gradio`: For the web interface.
- `PyPDF2`: For PDF parsing.
- `Pillow`: For image handling.
- `nltk`: For stop words filtering.

## Notes
- The script checks for CUDA availability and adjusts the model precision accordingly (`float16` for CUDA, `float32` for CPU).
- Memory management is optimized for CUDA devices using `torch.cuda.empty_cache()` and `PYTORCH_CUDA_ALLOC_CONF`.
- The Gradio gallery displays images in a 2-column grid with word labels.
- Ensure sufficient GPU memory if running on CUDA to handle multiple image generations.

## Limitations
- Image generation can be slow depending on hardware and the number of words extracted.
- Some words may fail to generate images due to model limitations or memory constraints.
- The Gradio interface requires an internet connection to launch publicly (`share=True`).

## License
This project is licensed under the MIT License. See the LICENSE file for details.