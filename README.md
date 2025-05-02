# InstructPix2Pix Fine-tuning for Robotic Action Frame Prediction

This project demonstrates how to convert simulation data (`.pkl` files) of robotic manipulation tasks into a training dataset and fine-tune the [`InstructPix2Pix`](https://github.com/timothybrooks/instruct-pix2pix) model using [`Hugging Face Accelerate`](https://github.com/huggingface/accelerate).

It is designed to work with camera-based robotic demonstrations and tasks such as block hammering, handover, and stacking.


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yutengzhang03/ip2p-finetune.git
cd ip2p-finetune
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
Put your raw .pkl simulation data in the original_data/ folder. 
### 📂 Dataset Structure 

<pre>
📁 original_data/
  ├── 📁 block_hammer_beat_sf50_D435_pkl/
  │ ├── 📁 episode0/
  │ │ ├── 0.pkl
  │ │ ├── 1.pkl
  │ │ └── ...
  │ ├── 📁 episode1/
  │ │ └── ...
  │ └── ...
  ├── 📁 block_handover_sf50_D435_pkl/
  │ └── 📁 episode0/...
  ├── 📁 blocks_stack_easy_sf50_D435_pkl/
  │ └── 📁 episode0/... 
</pre>


Then run:
```bash
accelerate launch create_ip2p_dataset.py \
  --samples_per_task 100 \
  --frame_gap 50 \
  --save_path data \
  --tasks \
  block_hammer_beat_sf50_D435_pkl="beat the block with the hammer" \
  block_handover_sf50_D435_pkl="handover the blocks" \
  blocks_stack_easy_sf50_D435_pkl="stack blocks" \
  --metadata_filename train_dataset.json
```
"--metadata_filename" followed by the output path
This will create:

data/0000/source.jpg, data/0000/target.jpg, ...

A train_dataset.json file describing each pair and prompt.

#### Example train_dataset.json entry:

{
  "image": "data/0000/source.jpg",
  "edited_image": "data/0000/target.jpg",
  "prompt": "Beat the block with the hammer"
}

#### You can change the "--samples_per_task" and "--metadata_filename" to create the valiadation dataset and test dataset.
For example:
```bash
accelerate launch create_ip2p_dataset.py \
  --samples_per_task 20 \
  --frame_gap 50 \
  --save_path data \
  --tasks \
  block_hammer_beat_sf50_D435_pkl="beat the block with the hammer" \
  block_handover_sf50_D435_pkl="handover the blocks" \
  blocks_stack_easy_sf50_D435_pkl="stack blocks" \
  --metadata_filename val_dataset.json
```
```bash
accelerate launch create_ip2p_dataset.py \
  --samples_per_task 20 \
  --frame_gap 50 \
  --save_path data \
  --tasks \
  block_hammer_beat_sf50_D435_pkl="beat the block with the hammer" \
  block_handover_sf50_D435_pkl="handover the blocks" \
  blocks_stack_easy_sf50_D435_pkl="stack blocks" \
  --metadata_filename test_dataset.json
```

## 📂 Project Structure 

```
. ├── create_ip2p_dataset.py # Convert .pkl logs to image-prompt training dataset 
  ├── train-instruct-ip2p.py # Training script using Hugging Face Accelerate 
  ├── original_data/ # Input simulation logs (.pkl) organized by episode 
  ├── data/ # Auto-generated output dataset (images) 
  ├── train_dataset.json # Metadata: prompt ↔ source/target image mapping 
  ├── requirements.txt # Python dependencies
```

## Step 2: Fine-Tune InstructPix2Pix

After preparing the dataset, you can fine-tune the model using:

```bash
accelerate launch fine-tune-ip2p.py \
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
  --output_dir="./ip2p-finetune-output" \
  --seed=42 \
  --original_image_column="image" \
  --edited_image_column="edited_image" \
  --edit_prompt_column="prompt"
```

## Step 3: Perpare Validation Dataset
Run:
```bash
accelerate launch create_ip2p_dataset.py \
  --samples_per_task 20 \
  --frame_gap 50 \
  --save_path val_data \
  --tasks \
  block_hammer_beat_sf50_D435_pkl="beat the block with the hammer" \
  block_handover_sf50_D435_pkl="handover the blocks" \
  blocks_stack_easy_sf50_D435_pkl="stack blocks" \
  --metadata_filename val_dataset.json
```

## Step 4: Evaluate the Model

Evaluate using the SSIM (Structural Similarity Index) and PSNR (Peak Signal-to-Noise Ratio) metrics

Run:
```bash
python evaluate_metrics.py --val_json val_dataset.json
```

