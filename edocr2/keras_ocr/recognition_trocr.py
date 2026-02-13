"""
recognition_trocr.py — Drop-in TrOCR replacement for edocr2's CRNN Recognizer.

Same API as recognition.Recognizer:
  - recognize(image) -> str
  - recognize_from_boxes(images, box_groups) -> list of list of str

But uses TrOCR (transformer) instead of CRNN+CTC.
Batch processing + GPU + zero logs = fast.

Place this file in: edocr2/keras_ocr/recognition_trocr.py
"""

import os
import sys
import typing
import logging
import warnings

import numpy as np
import cv2

# Silence everything
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image as PILImage

from . import tools


class TrOCRRecognizer:
    """Drop-in replacement for keras_ocr.recognition.Recognizer using TrOCR.

    Args:
        model_path: Path to fine-tuned TrOCR model directory.
                    If None, uses microsoft/trocr-base-printed (pretrained).
        device: 'cuda', 'cpu', or None (auto-detect).
        batch_size: Max crops to process at once (memory vs speed tradeoff).
        beam_size: Beam search width (4=accurate, 1=fast greedy).
    """

    def __init__(self, model_path=None, device=None, batch_size=32, beam_size=4):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        model_name = model_path or "microsoft/trocr-base-printed"

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        # Use half precision on GPU for speed
        if self.device == "cuda":
            self.model.half()

        self.batch_size = batch_size
        self.beam_size = beam_size

        # Warmup (first call is slow due to graph compilation)
        self._warmup()

    def _warmup(self):
        """Run a dummy inference to warm up the model."""
        dummy = PILImage.new('RGB', (200, 64), (255, 255, 255))
        pixel_values = self.processor(dummy, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        if self.device == "cuda":
            pixel_values = pixel_values.half()
        with torch.no_grad():
            self.model.generate(pixel_values, max_length=8)

    def _np_to_pil(self, image_np):
        """Convert numpy array (RGB or grayscale) to PIL RGB image."""
        if image_np is None:
            return PILImage.new('RGB', (200, 64), (255, 255, 255))
        if len(image_np.shape) == 2:
            # Grayscale -> RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[-1] == 1:
            # (H, W, 1) -> RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[-1] == 3:
            # Could be BGR (from cv2) or RGB
            # edocr2 tools.read() returns RGB, so we keep it
            pass
        return PILImage.fromarray(image_np.astype(np.uint8))

    @torch.no_grad()
    def _predict_batch(self, pil_images):
        """Run TrOCR on a batch of PIL images.

        Returns: list of predicted strings.
        """
        if not pil_images:
            return []

        pixel_values = self.processor(
            pil_images, return_tensors="pt"
        ).pixel_values.to(self.device)

        if self.device == "cuda":
            pixel_values = pixel_values.half()

        generated = self.model.generate(
            pixel_values,
            max_length=64,
            num_beams=self.beam_size,
        )

        return self.processor.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )

    def recognize(self, image):
        """Recognize text from a single image.

        Compatible with edocr2 Recognizer.recognize() API.

        Args:
            image: filepath (str) or numpy array (RGB, H×W×3).
                   edocr2 passes RGB numpy arrays from tools.read().

        Returns:
            Predicted text string.
        """
        if isinstance(image, str):
            image = tools.read(image)

        # edocr2 resizes to (31, 200) for CRNN — we don't need to,
        # TrOCR handles variable sizes via its ViT encoder.
        # But we do want a reasonable crop, so use warpBox target if needed.
        pil_img = self._np_to_pil(image)
        results = self._predict_batch([pil_img])
        return results[0] if results else ""

    def recognize_from_boxes(
        self, images, box_groups, **kwargs
    ) -> typing.List[typing.List[str]]:
        """Recognize text from images using lists of bounding boxes.

        Compatible with edocr2 Recognizer.recognize_from_boxes() API.
        Called by Pipeline.recognize().

        Args:
            images: List of images (numpy RGB arrays or filepaths).
            box_groups: List of box groups, one per image.
                        Each box is a (4, 2) array of corner points.

        Returns:
            List of lists of predicted strings.
        """
        assert len(box_groups) == len(images), \
            "Must provide same number of box groups as images."

        # Extract all crops
        all_crops = []
        start_end: typing.List[typing.Tuple[int, int]] = []

        for image, boxes in zip(images, box_groups):
            image = tools.read(image)

            for box in boxes:
                crop = tools.warpBox(
                    image=image,
                    box=box,
                    target_height=64,   # TrOCR works well at 64px
                    target_width=384,   # wider than CRNN's 200 — more detail
                )
                all_crops.append(crop)

            start = 0 if not start_end else start_end[-1][1]
            start_end.append((start, start + len(boxes)))

        if not all_crops:
            return [[]] * len(images)

        # Convert to PIL
        pil_crops = [self._np_to_pil(crop) for crop in all_crops]

        # Batch predict (split into chunks for memory)
        all_predictions = []
        for i in range(0, len(pil_crops), self.batch_size):
            batch = pil_crops[i:i + self.batch_size]
            preds = self._predict_batch(batch)
            all_predictions.extend(preds)

        # Group by image
        return [all_predictions[start:end] for start, end in start_end]

    # --- Compatibility stubs (not needed for inference) ---

    def compile(self, *args, **kwargs):
        """Not needed for TrOCR (uses PyTorch, not Keras)."""
        pass

    def get_batch_generator(self, *args, **kwargs):
        """Not needed — use train_trocr.py for training."""
        raise NotImplementedError(
            "TrOCR training uses train_trocr.py, not this method."
        )
