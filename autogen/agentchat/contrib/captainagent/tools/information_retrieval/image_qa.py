# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import os

from autogen.coding.func_with_reqs import with_requirements


@with_requirements(["transformers", "torch", "pillow"], ["transformers", "torch", "os"])
def image_qa(image, question, ckpt="Salesforce/blip-vqa-base"):
    """Perform question answering on an image using a pre-trained VQA model.

    Args:
        image (Union[str, Image.Image]): The image to perform question answering on. It can be either file path to the image or a PIL Image object.
        question: The question to ask about the image.
        ckpt: The checkpoint name to use. Default is "Salesforce/blip-vqa-base".

    Returns:
        dict: The generated answer text.
    """
    import torch
    from PIL import Image
    from transformers import BlipForQuestionAnswering, BlipProcessor

    def image_processing(img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, str):
            if os.path.exists(img):
                return Image.open(img).convert("RGB")
            else:
                full_path = img
                if os.path.exists(full_path):
                    return Image.open(full_path).convert("RGB")
                else:
                    raise FileNotFoundError

    def text_processing(file_path):
        # Check the file extension
        if file_path.endswith(".txt"):
            with open(file_path) as file:
                content = file.read()
        else:
            # if the file is not .txt, then it is a string, directly return the string
            return file_path
        return content

    image = image_processing(image)
    question = text_processing(question)

    processor = BlipProcessor.from_pretrained(ckpt)
    model = BlipForQuestionAnswering.from_pretrained(ckpt, torch_dtype=torch.float16).to("cuda")

    raw_image = image

    inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs)
    result_formatted = processor.decode(out[0], skip_special_tokens=True)

    return result_formatted
