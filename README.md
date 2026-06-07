# AdDetector 

> Automated TV broadcast ad detection and delivery verification system

---

## What This Does

Ad agencies book TV slots for their clients' ads. Currently, verifying whether those ads actually aired requires either trusting the TV channel's word or doing it manually. AdDetector solves this by watching the broadcast itself.

You give it:
- A list of booked ads (`bookings.xlsx`)
- A broadcast recording (`broadcast_test.mp4`)

It gives you back a full delivery report (`ad_delivery_report.xlsx`) showing exactly which ads aired, how many times, at what timestamps, and which ones were missed.

---

## How It Works

The system uses two independent signals to detect ads:

**Video Fingerprinting**
Each ad video is passed through a custom-trained CNN that converts every frame into a 512-dimensional feature vector. When scanning a broadcast, each frame gets compared against the reference database using a custom KD-tree for fast nearest-neighbor search. Consecutive matching frames confirm a detection.

**Audio Fingerprinting**
Audio is extracted from each ad and converted into a spectrogram using a custom FFT implementation. Prominent peaks are extracted and hashed into compact fingerprints — similar to how Shazam works, built from scratch. These hashes are matched against the broadcast audio to confirm detections.

When both video and audio agree, the detection is marked as high confidence (`BOTH`). When only one signal fires, it's logged as `VIDEO` or `AUDIO_ONLY`.

---

## Architecture

```
broadcast_test.mp4
        |
        +──► Frame Extractor (OpenCV + FFmpeg)
        |            |
        |     Custom CNN (11M params, PyTorch)
        |            |
        |     KD-tree Similarity Search
        |            |
        +──► Audio Extractor (FFmpeg subprocess)
                     |
              Custom FFT + Spectrogram
                     |
              Peak Detection + Hashing
                     |
              Hash Matching Engine
                     |
        ─────────────────────────────
                     |
              Signal Combiner
                     |
              Delivery Report Generator
                     |
        ad_delivery_report.xlsx (4 sheets)
```

---

## Components

| Component | What it does | No external AI library |
|---|---|---|
| `cnn.py` | ResNet-style CNN with residual blocks | Pure PyTorch — no pretrained weights |
| `audio_fingerprint.py` | STFT, spectrogram, peak detection, hashing | Pure NumPy FFT |
| `kdtree.py` | KD-tree with cosine similarity search | Pure Python/NumPy — no FAISS |
| `frame_extractor.py` | Video frame extraction and preprocessing | OpenCV + NumPy |
| `dataset.py` | Triplet dataset with hard negative mining | Custom curriculum learning |
| `train.py` | Training loop with triplet loss, LR decay, early stopping | PyTorch |
| `pipeline.py` | Full end-to-end orchestration | — |

---

## CNN Architecture

```
Input (3, 224, 224)
    │
Entry Block: Conv2d 7×7, stride 2 → BN → ReLU → MaxPool
    │
Residual Block 1: 2× Conv 3×3, 64 filters
    │
Residual Block 2: 2× Conv 3×3, 128 filters, stride 2
    │
Residual Block 3: 2× Conv 3×3, 256 filters, stride 2
    │
Residual Block 4: 2× Conv 3×3, 512 filters, stride 2
    │
Global Average Pool → Flatten
    │
FC: 512 → 512 (L2 normalized)
    │
Output: 512-d feature vector
```

Total parameters: **11,439,168**

Trained using **Triplet Loss** with hard negative mining and curriculum learning — the model learns visual similarity between ad frames rather than classification.

---

## Output Report

The Excel report has 4 sheets:

**Sheet 1 — Delivery Report**
| Ad Name | Duration | Booked | Detected | Missing | Extra | Confidence | First Seen | All Timestamps | Status |
|---|---|---|---|---|---|---|---|---|---|

Status cells are color-coded:
- 🟢 Green — DELIVERED (detected = booked)
- 🔴 Red — NOT AIRED (detected = 0)
- 🟡 Yellow — PARTIAL or EXTRA AIRINGS

**Sheet 2 — Timeline**
Every detection in chronological order with timestamp, confidence score, and signal type (BOTH / VIDEO / AUDIO_ONLY).

**Sheet 3 — Suspicious Airings**
Ads that appeared in the broadcast but were NOT in the bookings file — unbooked content detection.

**Sheet 4 — System Metrics**
Processing stats, signal breakdown, average confidence, total detections.

---

## Project Structure

```
AdDetector/
├── ads_raw/              ← reference ad videos (mp4)
├── ads_frames/           ← extracted frames per ad
├── filler/               ← non-ad broadcast content
├── filler_frames/        ← extracted filler frames (for training)
├── temp_frames/          ← temporary frames during registration
├── temp_broadcast_frames/← temporary frames during scanning
├── frame_extractor.py    ← video frame extraction + preprocessing
├── cnn.py                ← CNN architecture + Triplet Loss
├── audio_fingerprint.py  ← audio FFT + fingerprinting + matching
├── kdtree.py             ← KD-tree + VideoMatcher
├── dataset.py            ← TripletDataset with hard negatives
├── train.py              ← training loop with early stopping
├── pipeline.py           ← full end-to-end pipeline
├── bookings.xlsx         ← input: booked ad slots
├── broadcast_test.mp4    ← input: broadcast recording
├── model_final.pth       ← trained model weights
└── ad_delivery_report.xlsx ← output: delivery report
```

---


## Usage

**Step 1 — Prepare your ads**

Place your ad videos in `ads_raw/` named meaningfully:
```
ads_raw/surfexcel_30s.mp4
ads_raw/nescafe_30s.mp4
```

**Step 2 — Create bookings.xlsx**

| Ad Name | Booked Slots | Duration (sec) |
|---|---|---|
| surfexcel_30s | 3 | 30 |
| nescafe_30s | 2 | 30 |

Ad Name must match the filename without `.mp4`

**Step 3 — Add your broadcast video**

Place your broadcast recording as `broadcast_test.mp4`

**Step 4 — Run**

```bash
python pipeline.py
```

**Step 5 — Get your report**

Open `ad_delivery_report.xlsx`

---



## Test Data

The broadcast test video is not included in this repo 
due to file size. To create your own:

1. Download ad videos using yt-dlp
2. Download filler content
3. Create broadcast_list.txt with your desired order
4. Run: ffmpeg -f concat -safe 0 -i broadcast_list.txt 
         -c copy broadcast_test.mp4

---

## Pretrained Model

Download the pretrained model (100 epochs, 31 ads):
https://drive.google.com/file/d/1YC-Bu9G78inUwKBbt7sboOxdnKc30KaV/view?usp=sharing

Place it in the AdDetector root folder before running.

---

## Performance

Tested on a 1hr 38min simulated broadcast with 31 ads, mixed filler content:

| Metric | Value |
|---|---|
| Detection accuracy | 87% (27/31 ads within ±1) |
| Exact match | 65% (20/31 ads exact) |
| Average confidence | ~94% |
| Processing time | ~45 minutes (CPU) |
| Training data | 31 ads, ~2000 frames |

Accuracy improves significantly with more training data per ad.

---

## Roadmap

**Phase 1 — File-based detection** ✅ Complete
- Custom CNN video fingerprinting
- Custom audio fingerprinting
- KD-tree similarity search
- Excel delivery report with 4 sheets

**Phase 2 — Real-time detection** 🔨 In Progress
- Webcam/camera feed pointing at live TV
- Real-time frame processing
- Live detection dashboard
- Instant alerts when ad airs

**Phase 3 — Multi-channel support** 📋 Planned
- Monitor multiple channels simultaneously
- Channel-wise delivery reports
- Competitor ad intelligence
- Best time slot recommendations

---

## Tech Stack

- Python 3.11
- PyTorch (CNN training and inference)
- OpenCV (video processing)
- NumPy (FFT, KD-tree math)
- FFmpeg (audio/video extraction)
- openpyxl (Excel report generation)

---

## About

Built by a 2nd year CSE student M Sarvesh as a from-scratch implementation — no pretrained models, no FAISS, no audio libraries. Every component designed and coded independently.

Inspired by the real pain point of manual ad delivery verification in Indian advertising agencies.

---

## License

MIT License 
