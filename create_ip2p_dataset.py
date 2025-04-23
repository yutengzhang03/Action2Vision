import os
import pickle
import random
import shutil
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

def pkl_to_img(pkl_file, camera):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    img_array = data["observation"][camera]["rgb"]
    return Image.fromarray(img_array.astype(np.uint8))

def parse_task_argument(task_args):
    tasks = {}
    for item in task_args:
        if '=' not in item:
            raise ValueError(f"Invalid task format: {item}. Use path=prompt.")
        path, prompt = item.split("=", 1)
        tasks[path.strip()] = prompt.strip().strip('"').strip("'")
    return tasks

def main(
    base_path="original_data",
    save_path="data",
    samples_per_task=100,
    num_episodes=20,
    frame_gap=50,
    tasks={},
    metadata_filename="metadata.json"
):
    base_path = Path(base_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # camera_views = ["head_camera", "left_camera", "right_camera", "front_camera"]
    camera_views = ["front_camera", ] # In this project, we only use images from the front camera

    # Clear existing save_path
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)

    metadata = []
    sample_id = 0

    for task_folder, prompt in tasks.items():
        for _ in range(samples_per_task):
            ep = random.randint(0, num_episodes - 1)
            
            episode_files = list((base_path / task_folder / f"episode{ep}").glob("*.pkl"))
            max_frame_id = max([int(f.stem) for f in episode_files if f.stem.isdigit()], default=-1)

            if max_frame_id < frame_gap:
                continue  # Skip this episode if it's too short

            frame_id = random.randint(0, max_frame_id - frame_gap)

            cam = random.choice(camera_views)

            ep_path = base_path / task_folder / f"episode{ep}"
            source_path = ep_path / f"{frame_id}.pkl"
            target_path = ep_path / f"{frame_id + frame_gap}.pkl"

            if not (source_path.exists() and target_path.exists()):
                continue

            try:
                source_img = pkl_to_img(source_path, cam)
                target_img = pkl_to_img(target_path, cam)

                folder_name = f"{sample_id:04d}"
                output_dir = save_path / folder_name
                output_dir.mkdir(parents=True)

                source_img_path = output_dir / "source.jpg"
                target_img_path = output_dir / "target.jpg"

                source_img.save(source_img_path)
                target_img.save(target_img_path)

                metadata.append({
                    "image": f"{save_path.name}/{folder_name}/source.jpg",
                    "edited_image": f"{save_path.name}/{folder_name}/target.jpg",
                    "prompt": prompt
                })

                sample_id += 1
            except Exception as e:
                print(f"Skipping sample due to error: {e}")
                continue

    # Write JSON metadata
    metadata_path = Path(__file__).parent / metadata_filename
    if metadata_path.exists():
        print(f"🧹 Clearing existing metadata file: {metadata_path}")
        metadata_path.unlink() 
        
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


    print(f"Dataset created with {sample_id} samples.")
    print(f"metadata.json saved at: {metadata_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training dataset for InstructPix2Pix")

    parser.add_argument("--base_path", type=str, default="original_data", help="Base path to original .pkl data")
    parser.add_argument("--save_path", type=str, default="data", help="Path to save processed dataset")
    parser.add_argument("--samples_per_task", type=int, default=100, help="Number of samples per task")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes in each task")
    parser.add_argument("--frame_gap", type=int, default=50, help="Frame gap between source and target")
    parser.add_argument("--tasks", nargs="+", required=True, help='Tasks in format: folder="prompt"')
    parser.add_argument("--metadata_filename", type=str, default="metadata.json", help="Filename for the output JSON metadata file")


    args = parser.parse_args()
    tasks = parse_task_argument(args.tasks)

    main(
        base_path=args.base_path,
        save_path=args.save_path,
        samples_per_task=args.samples_per_task,
        num_episodes=args.num_episodes,
        frame_gap=args.frame_gap,
        tasks=tasks,
        metadata_filename=args.metadata_filename
    )

