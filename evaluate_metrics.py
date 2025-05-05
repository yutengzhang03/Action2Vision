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
    args.model_path,
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
    # ssim_scores_t = []
    # psnr_scores_t = []
    # ssim_scores_s = []
    # psnr_scores_s = []

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
        source_img = to_tensor(source_img).unsqueeze(0).to("cuda")

        # Compute metrics
        try:
            ssim = ssim_metric(pred_tensor, target_tensor).item()
            psnr = psnr_metric(pred_tensor, target_tensor).item()
            # ssim_t = ssim_metric(target_tensor, target_tensor).item()
            # psnr_t = psnr_metric(target_tensor, target_tensor).item()
            # ssim_s = ssim_metric(source_img, target_tensor).item()
            # psnr_s = psnr_metric(source_img, target_tensor).item()
            ssim_scores.append(ssim)
            psnr_scores.append(psnr)
            # ssim_scores_t.append(ssim_t)
            # psnr_scores_t.append(psnr_t)
            # ssim_scores_s.append(ssim_s)
            # psnr_scores_s.append(psnr_s)
        except Exception as e:
            print(f"⚠️ Metric error: {e}")
            continue

    # Final results
    print("\n✅ Evaluation Complete")
    print(f"📊 Average SSIM: {sum(ssim_scores) / len(ssim_scores):.4f}")
    # print(f"📊 Average PSNR: {sum(psnr_scores) / len(psnr_scores):.2f} dB")
    # print(f"📊 Average SSIM (Target): {sum(ssim_scores_t) / len(ssim_scores_t):.4f}")
    # print(f"📊 Average PSNR (Target): {sum(psnr_scores_t) / len(psnr_scores_t):.2f} dB")
    # print(f"📊 Average SSIM (Source): {sum(ssim_scores_s) / len(ssim_scores_s):.4f}")
    # print(f"📊 Average PSNR (Source): {sum(psnr_scores_s) / len(psnr_scores_s):.2f} dB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", type=str, default="test_dataset.json", help="Path to test_dataset.json")
    parser.add_argument("--model_path", type=str, default="timbrooks/instruct-pix2pix", help="Path to pretrained or fine-tuned model")
    args = parser.parse_args()

    evaluate_from_json(args.test_data_dir)
