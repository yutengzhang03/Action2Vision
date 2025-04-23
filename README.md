# InstructPix2Pix Fine-tuning on Custom Robotic Dataset

This project demonstrates how to convert simulation data (`.pkl` files) of robotic manipulation tasks into a training dataset and fine-tune the [`InstructPix2Pix`](https://github.com/timothybrooks/instruct-pix2pix) model using [Hugging Face Accelerate](https://github.com/huggingface/accelerate).

It is designed to work with camera-based robotic demonstrations and tasks such as block hammering, handover, and stacking.

---

## 📂 Project Structure 

. ├── create_ip2p_dataset.py # Convert .pkl logs to image-prompt training dataset ├── train-instruct-ip2p.py # Training script using Hugging Face Accelerate ├── original_data/ # Input simulation logs (.pkl) organized by episode ├── data/ # Auto-generated output dataset (images) ├── train_dataset.json # Metadata: prompt ↔ source/target image mapping ├── requirements.txt # Python dependencies


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Dependencies
We recommend using a virtual environment:

```bash
conda create -n ip2p python=3.10 -y
conda activate ip2p

pip install -r requirements.txt
```

#### If you haven’t used accelerate before, configure it with:
```bash
accelerate config
```

## Step 1: Create a Dataset from .pkl Files
Put your raw .pkl simulation data in the original_data/ folder. Then run:

```bash
accelerate launch create_ip2p_dataset.py \
  --samples_per_task 100 \
  --frame_gap 50 \
  --save_path data \
  --metadata_filename train_dataset.json \
  --tasks \
  block_hammer_beat_sf50_D435_pkl="Beat the block with the hammer" \
  block_handover_sf50_D435_pkl="Hand over the block from one gripper to another" \
  blocks_stack_easy_sf50_D435_pkl="Stack the block on top of another block"
```

This will create:

data/0000/source.jpg, data/0000/target.jpg, ...

A train_dataset.json file describing each pair and prompt.

### Example train_dataset.json entry:

{
  "image": "data/0000/source.jpg",
  "edited_image": "data/0000/target.jpg",
  "prompt": "Beat the block with the hammer"
}

## Step 2: Fine-Tune InstructPix2Pix

After preparing the dataset, you can fine-tune the model using:

```bash
accelerate launch train-instruct-ip2p.py \
  --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
  --train_data_dir="train_dataset.json" \
  --resolution=256 \
  --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=10 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --output_dir="./ip2p-lora-output" \
  --seed=42 \
  --original_image_column="image" \
  --edited_image_column="edited_image" \
  --edit_prompt_column="prompt"
```


