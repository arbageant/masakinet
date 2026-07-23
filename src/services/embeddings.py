"""Handles CLIP/SigLIP text & image ONNX inference.

Design ref: design-doc.md §2.1, §3 — dual-encoder contrastive model
producing joint text/image embeddings for the vector index.
"""

import io
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel

from src.config import settings

class EmbeddingService:
    """Wraps a CLIP/SigLIP dual encoder for text and image embedding."""

    def __init__(self, model_name: str = settings.clip_model_name) -> None:
        self.model_name = model_name

        # Detect GPU/CPU execution device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor (handles both CLIP and SigLIP variants)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def create_clip_caption(self, data: dict) -> str:
        """processes .json data into a smaller, simplified text input"""
        types_str = ", ".join(data.get("types", []))
        name = data.get("name", "").title()
        flavor = data.get("flavor_text", "")
        
        # Keeps text concise and natural for CLIP (~40-60 tokens)
        return f"A {types_str} type creature named {name}. {flavor}"

    def embed_text(self, text: str) -> list[float]:
        """Project a text string into the shared embedding space."""
        # Preprocess text input
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            # Get text features mapped to the joint vision-text space
            text_output = self.model.get_text_features(**inputs)

            # Some model variants return a BaseModelOutputWithPooling instead of a tensor
            if isinstance(text_output, torch.Tensor):
                text_features = text_output
            else:
                text_features = text_output.pooler_output

            # L2 normalize embeddings for standard cosine/inner-product similarity search
            normalized_features = F.normalize(text_features, p=2, dim=-1)

        # Convert tensor to a standard Python list of floats
        return normalized_features.squeeze(0).cpu().tolist()

    def embed_image(self, image_bytes: bytes) -> list[float]:
        """Project raw image bytes into the shared embedding space."""
        # Read raw byte string into a PIL Image and convert to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess image input (resizing, normalization, tensor conversion)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Get image features mapped to the joint vision-text space
            image_output = self.model.get_image_features(**inputs)

            # Some model variants return a BaseModelOutputWithPooling instead of a tensor
            if isinstance(image_output, torch.Tensor):
                image_features = image_output
            else:
                image_features = image_output.pooler_output

            # L2 normalize embeddings
            normalized_features = F.normalize(image_features, p=2, dim=-1)

        return normalized_features.squeeze(0).cpu().tolist()
