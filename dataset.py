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
    HARD_PAIRS = [
        ('dettol_1_30s', 'dettol_2_30s'),
        ('bru_20s', 'bru_25s'),
        ('samsung_1_35s', 'samsung_2_35s'),
        ('apple_1_20s', 'apple_2_20s'),
        ('amazon_20s', 'amazon_summer_20s'),
        ('kalyan_jewellers_17s', 'joyalukkas_69s'),  # both jewellery ads
        ('hyundai_20s', 'renault_45s'),               # both car ads
        ('closeup_20s', 'colgate_30s'),               # both toothpaste ads
    ]

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

        # After loading ad_frames, also load filler frames
        self.filler_frames = []
        filler_folder = "filler_frames"  # extract filler frames too

        if os.path.exists(filler_folder):
            for folder in os.listdir(filler_folder):
                folder_path = os.path.join(filler_folder, folder)
                if os.path.isdir(folder_path):
                    frames = [os.path.join(folder_path, f)
                            for f in os.listdir(folder_path)
                            if f.endswith('.jpg')]
                    self.filler_frames.extend(frames)
            print(f"Loaded {len(self.filler_frames)} filler frames")

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
        anchor_ad = random.choice(self.ad_ids)
        anchor_path, positive_path = random.sample(
            self.ad_frames[anchor_ad], 2)

        anchor   = self.extractor.preprocess(anchor_path)
        positive = self.extractor.preprocess(positive_path)

        # ── NEGATIVE SELECTION (one clean block) ──
        rand = random.random()

        if rand < 0.4:
            # Hard pair negative
            hard_negative_ad = None
            for a, b in self.HARD_PAIRS:
                if anchor_ad == a and b in self.ad_frames:
                    hard_negative_ad = b
                    break
                elif anchor_ad == b and a in self.ad_frames:
                    hard_negative_ad = a
                    break

            if hard_negative_ad:
                negative_path = random.choice(
                    self.ad_frames[hard_negative_ad])
            else:
                negative_ad   = random.choice(
                    [a for a in self.ad_ids if a != anchor_ad])
                negative_path = random.choice(self.ad_frames[negative_ad])
            negative = self.extractor.preprocess(negative_path)

        elif rand < 0.7 and self.filler_frames:
            # Filler negative
            negative_path = random.choice(self.filler_frames)
            negative      = self.extractor.preprocess(negative_path)

        else:
            # Random ad negative
            negative_ad   = random.choice(
                [a for a in self.ad_ids if a != anchor_ad])
            negative_path = random.choice(self.ad_frames[negative_ad])
            negative      = self.extractor.preprocess(negative_path)

        # ── AUGMENT ──
        anchor   = self.augment(anchor)
        positive = self.augment(positive)
        negative = self.augment(negative)

        return (torch.tensor(anchor,   dtype=torch.float32),
                torch.tensor(positive, dtype=torch.float32),
                torch.tensor(negative, dtype=torch.float32))

if __name__ == "__main__":
    # Extract filler frames
    filler_extractor = AdDatasetBuilder(
        ads_raw_folder="filler",
        ads_frames_folder="filler_frames"
    )
    filler_extractor.build()

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
