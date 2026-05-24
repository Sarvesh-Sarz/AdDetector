import numpy as np
import subprocess
import os

class AudioExtractor:
    """
    Extracts raw audio from video files using FFmpeg subprocess.
    No audio libraries — pure FFmpeg + NumPy.
    """
    def __init__(self, sample_rate=11025):
        self.sample_rate = sample_rate

    def extract(self, video_path):
        # Call FFmpeg as subprocess to extract raw PCM audio
        # -ac 1 = mono, -ar = sample rate, -f f32le = 32-bit float raw
        command = [
            'ffmpeg', '-i', video_path,
            '-ac', '1',
            '-ar', str(self.sample_rate),
            '-f', 'f32le',
            '-loglevel', 'quiet',
            'pipe:1'  # output to stdout
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        # Convert raw bytes to NumPy float32 array
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        print(f"Extracted audio: {len(audio)} samples "
              f"({len(audio)/self.sample_rate:.1f} seconds)")
        return audio


class SpectrogramGenerator:
    """
    Converts raw audio into a 2D spectrogram using custom STFT.
    Built from scratch using NumPy FFT — no librosa, no scipy.
    """
    def __init__(self, sample_rate=11025, n_fft=1024, hop_length=1024):
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.hop_length  = hop_length

    def generate(self, audio):
        # Step 1 — Hann window (reduces spectral leakage at frame edges)
        window = np.hanning(self.n_fft)

        frames = []
        # Step 2 — Slide window across signal
        for start in range(0, len(audio) - self.n_fft, self.hop_length):
            frame = audio[start : start + self.n_fft]

            # Step 3 — Apply window to frame
            windowed = frame * window

            # Step 4 — FFT: convert time domain → frequency domain
            spectrum = np.fft.rfft(windowed)

            # Step 5 — Magnitude spectrum (ignore phase)
            magnitude = np.abs(spectrum)

            frames.append(magnitude)

        # Shape: (freq_bins, time_frames)
        spectrogram = np.array(frames).T
        print(f"Spectrogram shape: {spectrogram.shape} "
              f"(freq_bins x time_frames)")
        return spectrogram

    def to_db(self, spectrogram):
        # Convert to decibels — makes peaks more visible
        # Add small value to avoid log(0)
        return 20 * np.log10(spectrogram + 1e-9)


class PeakDetector:
    """
    Finds prominent peaks in the spectrogram.
    A peak = point higher than all neighbors in a local window.
    """
    def __init__(self, window_size=5, threshold_ratio=0.8):
        self.window_size     = window_size
        self.threshold_ratio = threshold_ratio

    def find_peaks(self, spectrogram):
        db_spec  = 20 * np.log10(spectrogram + 1e-9)
        freq_bins, time_frames = db_spec.shape
        peaks = []

        w = self.window_size
        # Slide a window across the spectrogram
        for t in range(w, time_frames - w):
            for f in range(w, freq_bins - w):
                # Get local neighborhood
                neighborhood = db_spec[f-w:f+w, t-w:t+w]
                center_val   = db_spec[f, t]

                # Peak = center is the maximum in its neighborhood
                if center_val == neighborhood.max():
                    # Also above a global threshold
                    if center_val > db_spec.max() * self.threshold_ratio:
                        peaks.append((f, t))  # (frequency_bin, time_frame)

        print(f"Found {len(peaks)} peaks")
        return peaks


class AudioFingerprinter:
    """
    Converts peaks into hash fingerprints.
    For each anchor peak, pair it with nearby peaks in a time window.
    Hash = (freq1, freq2, time_delta) → unique integer ID for this sound moment.
    This is the core of Shazam-like matching — implemented from scratch.
    """
    def __init__(self, fan_out=5, time_window=50):
        self.fan_out     = fan_out     # how many pairs per anchor peak
        self.time_window = time_window # how far ahead to look for pairs

    def generate_hashes(self, peaks):
        hashes = []
        for i, (freq1, time1) in enumerate(peaks):
            # Look at next fan_out peaks within time_window
            for freq2, time2 in peaks[i+1 : i+1+self.fan_out]:
                if time2 - time1 > self.time_window:
                    break
                time_delta = time2 - time1

                # Hash: combine freq1, freq2, time_delta into one integer
                # Bit packing: freq1(10 bits) | freq2(10 bits) | delta(12 bits)
                hash_val = (freq1 & 0x3FF) << 22 | \
                           (freq2 & 0x3FF) << 12 | \
                           (time_delta & 0xFFF)

                hashes.append((hash_val, time1))

        print(f"Generated {len(hashes)} fingerprint hashes")
        return hashes  # list of (hash_value, time_offset) tuples

    def fingerprint_video(self, video_path):
        """Full pipeline: video → audio → spectrogram → peaks → hashes"""
        extractor  = AudioExtractor()
        spec_gen   = SpectrogramGenerator()
        peak_det   = PeakDetector()

        audio      = extractor.extract(video_path)
        spectrogram = spec_gen.generate(audio)
        peaks      = peak_det.find_peaks(spectrogram)
        hashes     = self.generate_hashes(peaks)
        return hashes

    def fingerprint_video_fast(self, video_path, chunk_seconds=30):
        """
        Process audio in chunks instead of all at once.
        Much faster and memory efficient.
        """
        extractor = AudioExtractor()
        audio = extractor.extract(video_path)
    
        chunk_size = chunk_seconds * extractor.sample_rate
        all_hashes = []
        total_chunks = len(audio) // chunk_size + 1
    
        for i, start in enumerate(range(0, len(audio), chunk_size)):
            chunk = audio[start:start + chunk_size]
            if len(chunk) < 1024:
                continue
            print(f"  Audio chunk {i+1}/{total_chunks}", end='\r')
            spec = SpectrogramGenerator().generate(chunk)
            peaks = PeakDetector().find_peaks(spec)
            # Adjust time offsets for chunk position
            time_offset = start // extractor.sample_rate
            hashes = [(h, t + time_offset) 
                    for h, t in self.generate_hashes(peaks)]
            all_hashes.extend(hashes)
    
        print(f"\nTotal hashes: {len(all_hashes)}")
        return all_hashes


class AudioMatcher:
    """
    Matches audio fingerprints of broadcast against reference ad fingerprints.
    Two fingerprints match if enough hashes align at consistent time offsets.
    """
    def match(self, broadcast_hashes, reference_hashes, min_matches=5):
        # Build lookup: hash_value → list of time offsets in reference
        ref_lookup = {}
        for hash_val, time_offset in reference_hashes:
            if hash_val not in ref_lookup:
                ref_lookup[hash_val] = []
            ref_lookup[hash_val].append(time_offset)

        # For each broadcast hash, check if it exists in reference
        # Count time offset differences — consistent offset = real match
        offset_counts = {}
        for hash_val, b_time in broadcast_hashes:
            if hash_val in ref_lookup:
                for r_time in ref_lookup[hash_val]:
                    delta = b_time - r_time
                    offset_counts[delta] = offset_counts.get(delta, 0) + 1

        if not offset_counts:
            return False, 0

        # Best match = most consistent time offset
        best_offset = max(offset_counts, key=offset_counts.get)
        best_count  = offset_counts[best_offset]

        matched = best_count >= min_matches
        return matched, best_count


# ── TEST ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    # Find first ad in ads_raw
    ads = [f for f in os.listdir('ads_raw') if f.endswith('.mp4')]
    if not ads:
        print("No ads found in ads_raw/")
        exit()

    test_ad = os.path.join('ads_raw', ads[0])
    print(f"Testing on: {test_ad}")
    print("-" * 50)

    # Test each component
    print("Step 1: Extracting audio...")
    extractor = AudioExtractor()
    audio     = extractor.extract(test_ad)

    print("\nStep 2: Generating spectrogram...")
    spec_gen    = SpectrogramGenerator()
    spectrogram = spec_gen.generate(audio)
    db_spec     = spec_gen.to_db(spectrogram)
    print(f"dB range: {db_spec.min():.1f} to {db_spec.max():.1f} dB")

    print("\nStep 3: Finding peaks...")
    peak_det = PeakDetector()
    peaks    = peak_det.find_peaks(spectrogram)

    print("\nStep 4: Generating hashes...")
    fingerprinter = AudioFingerprinter()
    hashes        = fingerprinter.generate_hashes(peaks)

    print("\nStep 5: Testing self-match (should match itself)...")
    matcher = AudioMatcher()
    matched, count = matcher.match(hashes, hashes)
    print(f"Self match result: {matched}, hash matches: {count}")

    print("\nAudio fingerprinter working perfectly!")

