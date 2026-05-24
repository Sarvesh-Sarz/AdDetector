import cv2
import numpy as np
import os

class FrameExtractor:
    def __init__(self, fps=1):
        self.fps = fps  # how many frames per second to extract

    def extract(self, video_path, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = int(total_frames / video_fps)
        interval = int(video_fps / self.fps)

        print(f"Video FPS: {video_fps}")
        print(f"Duration: {duration_sec} seconds")
        print(f"Extracting 1 frame every {interval} raw frames")

        count = 0
        saved = 0
        frame_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % interval == 0:
                timestamp_sec = count / video_fps
                filename = f"{output_folder}/frame_{saved:05d}_t{int(timestamp_sec)}s.jpg"
                cv2.imwrite(filename, frame)
                frame_data.append({
                    'frame_index': saved,
                    'timestamp_sec': timestamp_sec,
                    'filepath': filename
                })
                saved += 1

            count += 1

        cap.release()
        print(f"Done. Extracted {saved} frames from {duration_sec}s video")
        return frame_data

    def preprocess(self, frame_path):
        # Read frame
        frame = cv2.imread(frame_path)
        # Resize to 224x224
        frame = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB (OpenCV loads as BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to 0-1
        frame = frame.astype(np.float32) / 255.0
        # Subtract mean, divide by std (standard normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        # Transpose: (H, W, C) -> (C, H, W) for PyTorch
        frame = frame.transpose(2, 0, 1)
        return frame  # shape: (3, 224, 224)


if __name__ == "__main__":
    extractor = FrameExtractor(fps=1)

    # Extract frames
    frames = extractor.extract("test_video.mp4", "output_frames")

    # Test preprocessing on first frame
    if frames:
        processed = extractor.preprocess(frames[0]['filepath'])
        print(f"Preprocessed shape: {processed.shape}")
        print(f"Min value: {processed.min():.3f}")
        print(f"Max value: {processed.max():.3f}")
        print("Frame extractor working perfectly!")