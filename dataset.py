import os
import random
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from frame_extractor import FrameExtractor

class AdDatasetBuilder:
    """
    Takes raw ad videos, extracts frames, organizes by ad ID.
    Call this once to prepare your dataset before training.
    """
    def __init__(self, ads_raw_folder="ads_raw", ads_frames_folder="ads_frames"):
        self.ads_raw_folder   = ads_raw_folder
        self.ads_frames_folder = ads_frames_folder
        self.extractor        = FrameExtractor(fps=2)  # 2 frames/sec for ads

    def build(self):
        os.makedirs(self.ads_frames_folder, exist_ok=True)
        video_files = [f for f in os.listdir(self.ads_raw_folder)
                       if f.endswith(('.mp4', '.mkv', '.avi'))]

        print(f"Found {len(video_files)} ad videos")

        for video_file in video_files:
            # Use filename (without extension) as the ad ID
            ad_id = os.path.splitext(video_file)[0]
            ad_id = ad_id.replace(" ", "_")[:50]  # clean up name

            output_folder = os.path.join(self.ads_frames_folder, ad_id)

            if os.path.exists(output_folder):
                print(f"Skipping {ad_id} — already extracted")
                continue

            video_path = os.path.join(self.ads_raw_folder, video_file)
            frames = self.extractor.extract(video_path, output_folder)
            print(f"Ad '{ad_id}': {len(frames)} frames extracted")

        print("Dataset build complete!")


class TripletDataset(Dataset):
    """
    For each item returned:
    - anchor:   a frame from ad X
    - positive: a different frame from ad X (same ad)
    - negative: a frame from ad Y (different ad)
    """
    def __init__(self, ads_frames_folder="ads_frames"):
        self.ads_frames_folder = ads_frames_folder
        self.extractor = FrameExtractor()

        # Build a dict: {ad_id: [list of frame paths]}
        self.ad_frames = {}
        for ad_id in os.listdir(ads_frames_folder):
            ad_path = os.path.join(ads_frames_folder, ad_id)
            if not os.path.isdir(ad_path):
                continue
            frames = [os.path.join(ad_path, f)
                      for f in os.listdir(ad_path)
                      if f.endswith('.jpg')]
            if len(frames) >= 2:  # need at least 2 frames for anchor+positive
                self.ad_frames[ad_id] = frames

        self.ad_ids = list(self.ad_frames.keys())
        print(f"Dataset loaded: {len(self.ad_ids)} ads, "
              f"{sum(len(v) for v in self.ad_frames.values())} total frames")

    def __len__(self):
        # Each ad contributes multiple triplets
        return len(self.ad_ids) * 20

    def augment(self, frame):
        frame = frame.transpose(1, 2, 0)

        # ONLY brightness and noise — safe for ads
        brightness = random.uniform(0.9, 1.1)  # smaller range
        frame = np.clip(frame * brightness, -3.0, 3.0)

        # Very small noise only
        if random.random() > 0.7:
            noise = np.random.randn(*frame.shape) * 0.02
            frame = frame + noise

        frame = frame.transpose(2, 0, 1)
        return frame.astype(np.float32)

    def __getitem__(self, idx):
        use_hard_negative = idx > len(self) * 0.3

        anchor_ad = random.choice(self.ad_ids)
        negative_ad = random.choice(
            [a for a in self.ad_ids if a != anchor_ad])

        anchor_path, positive_path = random.sample(
            self.ad_frames[anchor_ad], 2)

        # Preprocess all three frames
        anchor   = self.extractor.preprocess(anchor_path)
        positive = self.extractor.preprocess(positive_path)

        if use_hard_negative:
            candidates = random.sample(
                [a for a in self.ad_ids if a != anchor_ad],
                min(5, len(self.ad_ids) - 1))
            best_negative_path = None
            best_similarity    = -1
            anchor_tensor = torch.tensor(anchor).flatten().unsqueeze(0)
            for candidate_ad in candidates:
                candidate_path = random.choice(
                    self.ad_frames[candidate_ad])
                candidate = self.extractor.preprocess(candidate_path)
                candidate_tensor = torch.tensor(candidate).flatten().unsqueeze(0)
                sim = torch.nn.functional.cosine_similarity(
                    anchor_tensor, candidate_tensor).item()
                if sim > best_similarity:
                    best_similarity    = sim
                    best_negative_path = candidate_path
            negative = self.extractor.preprocess(best_negative_path)
        else:
            negative_path = random.choice(self.ad_frames[negative_ad])
            negative      = self.extractor.preprocess(negative_path)

        # Apply augmentation to ALL three frames
        anchor   = self.augment(anchor)
        positive = self.augment(positive)
        negative = self.augment(negative)

        return (torch.tensor(anchor,   dtype=torch.float32),
                torch.tensor(positive, dtype=torch.float32),
                torch.tensor(negative, dtype=torch.float32))


if __name__ == "__main__":
    # Step 1: Build dataset from raw videos
    builder = AdDatasetBuilder()
    builder.build()

    # Step 2: Test the dataset
    dataset = TripletDataset()
    anchor, positive, negative = dataset[0]
    print(f"Anchor shape:   {anchor.shape}")
    print(f"Positive shape: {positive.shape}")
    print(f"Negative shape: {negative.shape}")
    print("Dataset working perfectly!")