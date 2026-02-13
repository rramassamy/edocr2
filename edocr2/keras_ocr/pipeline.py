# pylint: disable=too-few-public-methods
import numpy as np

from . import detection, recognition, tools


class Pipeline:
    """A wrapper for a combination of detector and recognizer.

    Args:
        detector: The detector to use
        recognizer: The recognizer to use. Can be:
                    - recognition.Recognizer (CRNN, original)
                    - recognition_trocr.TrOCRRecognizer (transformer, new)
                    - None (auto-creates CRNN recognizer)
        recognizer_backend: 'crnn' or 'trocr'. If set, overrides recognizer arg.
        trocr_model_path: Path to fine-tuned TrOCR model (for trocr backend).
        scale: The scale factor to apply to input images
        max_size: The maximum single-side dimension of images for inference.
    """

    def __init__(self, detector=None, recognizer=None, scale=2, max_size=2048,
                 recognizer_backend=None, trocr_model_path=None):
        if detector is None:
            detector = detection.Detector()

        if recognizer_backend == 'trocr' or trocr_model_path is not None:
            from . import recognition_trocr
            recognizer = recognition_trocr.TrOCRRecognizer(
                model_path=trocr_model_path
            )
        elif recognizer is None:
            recognizer = recognition.Recognizer()

        self.scale = scale
        self.detector = detector
        self.recognizer = recognizer
        self.max_size = max_size

    def recognize(self, images, detection_kwargs=None, recognition_kwargs=None):
        """Run the pipeline on one or multiples images.

        Args:
            images: The images to parse (list of images or filepaths)
            detection_kwargs: Arguments to pass to the detector call
            recognition_kwargs: Arguments to pass to the recognizer call

        Returns:
            A list of lists of (text, box) tuples.
        """

        # Make sure we have an image array to start with.
        if not isinstance(images, np.ndarray):
            images = [tools.read(image) for image in images]
        # This turns images into (image, scale) tuples temporarily
        images = [
            tools.resize_image(image, max_scale=self.scale, max_size=self.max_size)
            for image in images
        ]
        max_height, max_width = np.array(
            [image.shape[:2] for image, scale in images]
        ).max(axis=0)
        scales = [scale for _, scale in images]
        images = np.array(
            [
                tools.pad(image, width=max_width, height=max_height)
                for image, _ in images
            ]
        )
        if detection_kwargs is None:
            detection_kwargs = {}
        if recognition_kwargs is None:
            recognition_kwargs = {}
        box_groups = self.detector.detect(images=images, **detection_kwargs)
        prediction_groups = self.recognizer.recognize_from_boxes(
            images=images, box_groups=box_groups, **recognition_kwargs
        )
        box_groups = [
            tools.adjust_boxes(boxes=boxes, boxes_format="boxes", scale=1 / scale)
            if scale != 1
            else boxes
            for boxes, scale in zip(box_groups, scales)
        ]
        return [
            list(zip(predictions, boxes))
            for predictions, boxes in zip(prediction_groups, box_groups)
        ]
