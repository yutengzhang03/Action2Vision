import os
import json
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio

# Load pipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    # "./ip2p-finetune-output",
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Image transforms
to_tensor = transforms.ToTensor()
resize = transforms.Resize((256, 256))

# Metrics
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to("cuda")

def load_image_tensor(path):
    image = Image.open(path).convert("RGB")
    image = resize(image)
    return to_tensor(image).unsqueeze(0).to("cuda")

def evaluate_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    ssim_scores = []
    psnr_scores = []

    for item in tqdm(data, desc="Evaluating"):
        source_path = Path(item["image"])
        target_path = Path(item["edited_image"])
        prompt = item["prompt"]

        if not (source_path.exists() and target_path.exists()):
            print(f"⚠️ Skipping missing files: {source_path}, {target_path}")
            continue

        # Load input image
        source_img = Image.open(source_path).convert("RGB").resize((256, 256))
        target_tensor = load_image_tensor(target_path)

        # Inference
        try:
            edited_img = pipe(prompt=prompt, image=source_img).images[0]
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            continue

        # Convert prediction to tensor
        pred_tensor = to_tensor(edited_img).unsqueeze(0).to("cuda")

        # Compute metrics
        try:
            ssim = ssim_metric(pred_tensor, target_tensor).item()
            psnr = psnr_metric(pred_tensor, target_tensor).item()
            ssim_scores.append(ssim)
            psnr_scores.append(psnr)
        except Exception as e:
            print(f"⚠️ Metric error: {e}")
            continue

    # Final results
    print("\n✅ Evaluation Complete")
    print(f"📊 Average SSIM: {sum(ssim_scores) / len(ssim_scores):.4f}")
    print(f"📊 Average PSNR: {sum(psnr_scores) / len(psnr_scores):.2f} dB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_json", type=str, default="val_dataset.json", help="Path to val_dataset.json")
    args = parser.parse_args()

    evaluate_from_json(args.val_json)


