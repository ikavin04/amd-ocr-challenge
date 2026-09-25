import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-VL-3B-Instruct"
CONTAINER_MODEL_PATH = Path("/models/Qwen2.5-VL-3B-Instruct")

if CONTAINER_MODEL_PATH.exists():
    MODEL_PATH = CONTAINER_MODEL_PATH
else:
    MODEL_PATH = LOCAL_MODEL_PATH

CONTAINER_OUTPUT_PATH = Path("/app/output")
LOCAL_OUTPUT_PATH = PROJECT_ROOT / "output"

if Path("/app").exists():
    OUTPUT_DIR = CONTAINER_OUTPUT_PATH
else:
    OUTPUT_DIR = LOCAL_OUTPUT_PATH

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# OCR prompt
# ------------------------------------------------------------

PROMPT = """You are an OCR extraction system.

Look carefully at the image and identify the PRIMARY text requested from the image.

For a LICENSE PLATE:
- Return ONLY the license plate registration number/characters.
- Do NOT return the state/province name.
- Do NOT return slogans.
- Do NOT return decorative text.
- Do NOT return explanations.
- Do NOT return labels.
- Do NOT include centered dots or interpuncts.
- Preserve letters, numbers, and region characters belonging to the registration.

For a ROAD SIGN:
- Return the actual visible sign wording and/or numbers.
- For multi-line signs, read from top to bottom.
- Join multiple lines with spaces.
- Do not describe the sign.

For an ADVISORY SPEED PLAQUE:
- Return ONLY the visible number.

Output ONLY the final OCR text.
Nothing else.
"""


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

print("Loading processor...")
processor = AutoProcessor.from_pretrained(str(MODEL_PATH))

print("Loading Qwen2.5-VL-3B...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(MODEL_PATH),
    dtype="auto",
    device_map="auto",
)

model.eval()

print("Model loaded successfully.")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GiB",
)


# ------------------------------------------------------------
# Output cleanup
# ------------------------------------------------------------

def clean_output(text: str) -> str:
    """Remove unwanted formatting while preserving OCR characters."""

    text = text.strip()

    # Remove plate separators that the challenge normalizes away.
    for symbol in ("·", "•", "・"):
        text = text.replace(symbol, "")

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ------------------------------------------------------------
# OCR
# ------------------------------------------------------------

def read_text(image_path: Path) -> tuple[str, float]:
    """Run Qwen OCR on one image."""

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    start_time = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    elapsed = time.perf_counter() - start_time

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(inputs["input_ids"], generated_ids)
    ]

    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    result = clean_output(result)

    return result, elapsed


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMD OCR Challenge application"
    )

    parser.add_argument(
        "--input-image",
        required=True,
        help="Path to the input PNG, JPEG, or TIFF image",
    )

    args = parser.parse_args()

    input_path = Path(args.input_image)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image does not exist: {input_path}"
        )

    print("\nInput image:", input_path)
    print("Running OCR...")

    result, elapsed = read_text(input_path)

    output_filename = f"{input_path.stem}_output.json"
    output_path = OUTPUT_DIR / output_filename

    output_data = {
        "text": result,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n==============================")
    print("OCR RESULT:", result)
    print("TIME:", f"{elapsed:.2f} seconds")
    print("OUTPUT:", output_path)
    print("==============================")


if __name__ == "__main__":
    main()
