import os
import numpy as np
import torch
import torch.nn.functional as F
from frame_extractor import FrameExtractor
from cnn import AdFingerprintCNN
from audio_fingerprint import AudioFingerprinter, AudioMatcher
from kdtree import VideoMatcher

class AdDatabase:
    """
    Stores all reference ad fingerprints.
    For each ad: video vectors in KD-tree + audio hashes in dict.
    """
    def __init__(self):
        self.video_matcher    = VideoMatcher(similarity_threshold=0.80)
        self.audio_fingerprinter = AudioFingerprinter()
        self.audio_matcher    = AudioMatcher()

        # Storage before building KD-tree
        self.all_vectors    = []
        self.all_ad_ids     = []
        self.all_frame_idxs = []

        # Audio hashes: {ad_id: [(hash_val, time_offset), ...]}
        self.audio_hashes   = {}

        # Ad metadata: {ad_id: {name, duration_sec, booked_slots}}
        self.ad_metadata    = {}

    def compute_ad_uniqueness(self):
        """
        For each ad, compute how similar it is to other ads.
        High similarity to others = needs higher threshold.
        """
        self.ad_thresholds = {}

        if len(self.all_vectors) == 0:
            return

        # Group vectors by ad_id
        ad_vector_map = {}
        for vec, ad_id in zip(self.all_vectors, self.all_ad_ids):
            if ad_id not in ad_vector_map:
                ad_vector_map[ad_id] = []
            ad_vector_map[ad_id].append(vec)

        # For each ad, find max similarity to all other ads
        for ad_id, ad_vecs in ad_vector_map.items():
            ad_center = np.mean(ad_vecs, axis=0)
            ad_center = ad_center / np.linalg.norm(ad_center)

            max_sim_to_others = 0
            for other_id, other_vecs in ad_vector_map.items():
                if other_id == ad_id:
                    continue
                other_center = np.mean(other_vecs, axis=0)
                other_center = other_center / np.linalg.norm(other_center)
                sim = float(np.dot(ad_center, other_center))
                max_sim_to_others = max(max_sim_to_others, sim)

            # Higher similarity to others = higher threshold needed
            self.ad_thresholds[ad_id] = min(0.98,
                0.91 + max_sim_to_others * 0.06)
            print(f"  {ad_id:<30} threshold: {self.ad_thresholds[ad_id]:.3f} "
                  f"(max_sim_to_others: {max_sim_to_others:.3f})")

        # ── LEARNED CORRECTIONS ──
        # Based on observed performance — override auto threshold
        corrections = {
            # Consistently under-detected — lower threshold
            'hyundai_20s':           0.75,
            'dettol_2_30s':          0.87,
            # Consistently over-detected — raise threshold
            'apple_1_20s':           0.975,
            'kelloggs_30s':          0.975,
            'levi_30s':              0.975,
            'tanishq_jewellery_50s': 0.965,
        }
        print("\n  Applying learned corrections:")
        for ad_id, threshold in corrections.items():
            if ad_id in self.ad_thresholds:
                old = self.ad_thresholds[ad_id]
                self.ad_thresholds[ad_id] = threshold
                print(f"  {ad_id:<30} {old:.3f} → {threshold:.3f}")

    def register_ad(self, ad_id, video_path, booked_slots=1):
        """
        Register a new ad into the database.
        Extracts both video vectors and audio hashes.
        """
        print(f"\nRegistering ad: {ad_id}")

        # ── VIDEO FINGERPRINTING ──
        extractor = FrameExtractor(fps=2)
        frames    = extractor.extract(video_path,
                                      f"temp_frames/{ad_id}")

        # Load trained CNN
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
        model  = AdFingerprintCNN(embedding_dim=512).to(device)

        if os.path.exists("model_final_100ep_3.pth"):
            model.load_state_dict(torch.load("model_final_100ep_3.pth",
                                  map_location=device))
            print(f"Loaded trained model weights")
        else:
            print("Warning: No trained model found, using random weights")

        model.eval()

        # Extract feature vector for each frame
        for frame_data in frames:
            preprocessed = extractor.preprocess(frame_data['filepath'])
            tensor = torch.tensor(preprocessed,
                                  dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                vector = model(tensor).squeeze().cpu().numpy()

            self.all_vectors.append(vector)
            self.all_ad_ids.append(ad_id)
            self.all_frame_idxs.append(frame_data['frame_index'])

        # ── AUDIO FINGERPRINTING ──
        hashes = self.audio_fingerprinter.fingerprint_video(video_path)
        self.audio_hashes[ad_id] = hashes

        # ── METADATA ──
        duration = len(frames) / 2  # fps=2
        self.ad_metadata[ad_id] = {
            'name':         ad_id,
            'duration_sec': duration,
            'booked_slots': booked_slots,
            'video_path':   video_path
        }

        print(f"Registered '{ad_id}': "
            f"{len(frames)} frames, "
            f"{len(self.audio_hashes[ad_id])} audio hashes")

    def build_index(self):
        """Build KD-tree from all registered ad vectors."""
        if not self.all_vectors:
            print("No ads registered yet!")
            return
        vectors = np.array(self.all_vectors)
        self.video_matcher.build_database(
            vectors, self.all_ad_ids, self.all_frame_idxs)
        print(f"\nIndex built: {len(self.all_vectors)} total vectors "
              f"from {len(self.ad_metadata)} ads")


class BroadcastScanner:
    """
    Scans a broadcast video frame by frame.
    Matches each frame against the ad database using both
    video CNN vectors and audio hashes.
    Combines both signals for high confidence detection.
    """
    def __init__(self, database: AdDatabase):
        self.db        = database
        self.extractor = FrameExtractor(fps=1)

    def scan(self, broadcast_path):
        """
        Full scan of broadcast video.
        Returns list of detections with timestamps.
        """
        print(f"\nScanning broadcast: {broadcast_path}")
        print("-" * 50)

        # ── EXTRACT FRAMES ──
        frames = self.extractor.extract(broadcast_path,
                                        "temp_broadcast_frames")
        print(f"Extracted {len(frames)} frames to scan")

        # ── LOAD MODEL ──
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
        model  = AdFingerprintCNN(embedding_dim=512).to(device)

        if os.path.exists("model_final_100ep_3.pth"):
            model.load_state_dict(torch.load("model_final_100ep_3.pth",
                                  map_location=device))
        model.eval()

        # ── VIDEO SCANNING ──
        print("\nRunning video matching...")
        query_vectors = []
        timestamps    = []

        for frame_data in frames:
            preprocessed = self.extractor.preprocess(
                frame_data['filepath'])
            tensor = torch.tensor(preprocessed,
                                  dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                vector = model(tensor).squeeze().cpu().numpy()
            query_vectors.append(vector)
            timestamps.append(frame_data['timestamp_sec'])

        # Match sequence against database
        video_detections = self.db.video_matcher.match_sequence(
            query_vectors, 
            min_consecutive=12,
            ad_thresholds=self.db.ad_thresholds if hasattr(self.db, 'ad_thresholds') else None
        )

        # Add real timestamps to detections
        for det in video_detections:
            det['timestamp_sec'] = timestamps[det['start_frame']]
            det['end_timestamp'] = timestamps[min(det['end_frame'],
                                                   len(timestamps)-1)]
            det['signal'] = 'VIDEO'

        print(f"Video detections: {len(video_detections)}")

        # ── AUDIO SCANNING ──
        print("\nRunning audio matching...")
        audio_detections = []

        fingerprinter = AudioFingerprinter()
        broadcast_hashes = fingerprinter.fingerprint_video_fast(broadcast_path)

        for ad_id, ref_hashes in self.db.audio_hashes.items():
            if not ref_hashes:
                continue
            matched, count, timestamp_sec = self.db.audio_matcher.match(
                broadcast_hashes, ref_hashes, min_matches=10)
            if matched:
                audio_detections.append({
                    'ad_id':        ad_id,
                    'hash_matches': count,
                    'timestamp_sec': timestamp_sec,
                    'signal':       'AUDIO'
                })

        print(f"Audio detections: {len(audio_detections)}")

        # ── COMBINE SIGNALS ──
        final_detections = self._combine_signals(
            video_detections, audio_detections, timestamps)

        return final_detections

    def _combine_signals(self, video_dets, audio_dets, timestamps):
        # Get set of ad_ids confirmed by audio
        audio_confirmed_ads = {d['ad_id'] for d in audio_dets}
    
        print(f"\nAudio confirmed ads: {audio_confirmed_ads}")
    
        final = []
        for det in video_dets:
            if det['ad_id'] in audio_confirmed_ads:
                det['confidence'] = min(det['confidence'] * 1.2, 0.99)
                det['signal']     = 'BOTH'
            else:
                det['signal']     = 'VIDEO'
            final.append(det)

        # Audio only detections
        video_ad_ids = {d['ad_id'] for d in video_dets}
        for det in audio_dets:
            if det['ad_id'] not in video_ad_ids:
                final.append({
                    'ad_id':         det['ad_id'],
                    'timestamp_sec': det.get('timestamp_sec', 0),
                    'end_timestamp': det.get('timestamp_sec', 0),
                    'confidence':    0.75,
                    'signal':        'AUDIO_ONLY'
                })

        return final


class DetectionReport:
    """
    Compares detected airings against booked orders.
    Generates Excel report — only Status cell is colored.
    """
    def __init__(self, database: AdDatabase):
        self.db = database

    def generate(self, detections, output_path="ad_delivery_report.xlsx"):
        import openpyxl
        from openpyxl.styles import (PatternFill, Font, Alignment,
                                    Border, Side)
        from openpyxl.utils import get_column_letter
        import time

        start_time = time.time()
        wb = openpyxl.Workbook()

        # ── COLORS ──
        GREEN  = PatternFill("solid", fgColor="70AD47")
        RED    = PatternFill("solid", fgColor="FF0000")
        YELLOW = PatternFill("solid", fgColor="FFC000")
        HEADER = PatternFill("solid", fgColor="1F3864")
        WHITE  = PatternFill("solid", fgColor="FFFFFF")
        GRAY   = PatternFill("solid", fgColor="F2F2F2")
        ORANGE = PatternFill("solid", fgColor="FF6600")

        thin       = Side(style="thin", color="CCCCCC")
        border     = Border(left=thin, right=thin,
                            top=thin,  bottom=thin)
        centered   = Alignment(horizontal="center",
                            vertical="center", wrap_text=True)
        left       = Alignment(horizontal="left", vertical="center")
        white_bold = Font(bold=True, color="FFFFFF")
        bold       = Font(bold=True)

        # ════════════════════════════════════════════
        # SHEET 1 — MAIN DELIVERY REPORT
        # ════════════════════════════════════════════
        ws1 = wb.active
        ws1.title = "Delivery Report"

        # Title
        ws1.merge_cells("A1:J1")
        ws1["A1"]           = "AD DELIVERY VERIFICATION REPORT"
        ws1["A1"].font      = Font(bold=True, size=14, color="FFFFFF")
        ws1["A1"].alignment = Alignment(horizontal="center",
                                        vertical="center")
        ws1["A1"].fill      = HEADER
        ws1.row_dimensions[1].height = 30

        ws1.merge_cells("A2:J2")
        ws1["A2"]           = "Generated by AdDetector — Automated Broadcast Monitoring System"
        ws1["A2"].alignment = Alignment(horizontal="center")
        ws1["A2"].font      = Font(italic=True, color="666666")

        # Headers
        headers = ["Ad Name", "Duration\n(sec)", "Booked\nSlots",
                "Detected", "Missing", "Extra",
                "Avg\nConfidence", "First Seen",
                "All Timestamps", "Status"]

        for col, header in enumerate(headers, 1):
            cell           = ws1.cell(row=4, column=col, value=header)
            cell.font      = white_bold
            cell.fill      = HEADER
            cell.alignment = centered
            cell.border    = border
        ws1.row_dimensions[4].height = 30

        # Count detections per ad
        detection_counts     = {}
        detection_times      = {}
        detection_confidence = {}

        for det in detections:
            ad_id = det['ad_id']
            detection_counts[ad_id] = detection_counts.get(ad_id, 0) + 1
            if ad_id not in detection_times:
                detection_times[ad_id]      = []
                detection_confidence[ad_id] = []
            ts = det.get('timestamp_sec', 0)
            detection_times[ad_id].append(
                f"{int(ts)//60:02d}:{int(ts)%60:02d}")
            detection_confidence[ad_id].append(
                det.get('confidence', 0))

        # Fill rows
        row = 5
        summary = {'delivered': 0, 'missing': 0,
                'partial': 0,   'extra': 0}

        for ad_id, meta in self.db.ad_metadata.items():
            booked   = meta['booked_slots']
            detected = detection_counts.get(ad_id, 0)
            missing  = max(0, booked - detected)
            extra    = max(0, detected - booked)
            times    = detection_times.get(ad_id, [])
            confs    = detection_confidence.get(ad_id, [])

            first_seen   = times[0] if times else "—"
            all_times    = ", ".join(times) if times else "—"
            avg_conf     = f"{sum(confs)/len(confs)*100:.1f}%" \
                        if confs else "—"

            if detected == 0:
                status = "NOT AIRED"
                color  = RED
                summary['missing'] += 1
            elif detected == booked:
                status = "DELIVERED"
                color  = GREEN
                summary['delivered'] += 1
            elif detected < booked:
                status = "PARTIAL"
                color  = YELLOW
                summary['partial'] += 1
            else:
                status = "EXTRA AIRINGS"
                color  = YELLOW
                summary['extra'] += 1

            row_data = [ad_id, meta['duration_sec'],
                        booked, detected, missing, extra,
                        avg_conf, first_seen, all_times, status]

            for col, value in enumerate(row_data, 1):
                cell           = ws1.cell(row=row, column=col,
                                        value=value)
                cell.border    = border
                cell.alignment = centered if col != 9 else left
                cell.fill      = WHITE
                if col == 10:
                    cell.fill = color
                    cell.font = Font(bold=True, color="FFFFFF")
            ws1.row_dimensions[row].height = 20
            row += 1

        # Summary section
        row += 1
        ws1.merge_cells(f"A{row}:J{row}")
        ws1[f"A{row}"]           = "SUMMARY"
        ws1[f"A{row}"].font      = Font(bold=True, size=12,
                                        color="FFFFFF")
        ws1[f"A{row}"].fill      = HEADER
        ws1[f"A{row}"].alignment = Alignment(horizontal="center")
        ws1.row_dimensions[row].height = 25
        row += 1

        summary_rows = [
            ("Total Ads Booked",    len(self.db.ad_metadata), WHITE),
            ("Fully Delivered ✓",   summary['delivered'],     GREEN),
            ("Not Aired ✗",         summary['missing'],       RED),
            ("Partial / Extra ⚠",   summary['partial'] +
                                    summary['extra'],         YELLOW),
        ]
        for label, value, fill in summary_rows:
            ws1.cell(row=row, column=1, value=label).font = bold
            val_cell           = ws1.cell(row=row, column=2,
                                        value=value)
            val_cell.fill      = fill
            val_cell.font      = Font(bold=True,
                                    color="FFFFFF"
                                    if fill != WHITE else "000000")
            val_cell.alignment = centered
            row += 1

        # Column widths sheet 1
        widths = [25, 10, 10, 10, 10, 8, 12, 12, 40, 16]
        for i, width in enumerate(widths, 1):
            ws1.column_dimensions[get_column_letter(i)].width = width

        # ════════════════════════════════════════════
        # SHEET 2 — TIMELINE
        # ════════════════════════════════════════════
        ws2          = wb.create_sheet("Timeline")
        ws2["A1"]    = "AD DETECTION TIMELINE"
        ws2["A1"].font      = Font(bold=True, size=14, color="FFFFFF")
        ws2["A1"].fill      = HEADER
        ws2["A1"].alignment = Alignment(horizontal="center",
                                        vertical="center")
        ws2.merge_cells("A1:D1")
        ws2.row_dimensions[1].height = 30

        # Headers
        for col, header in enumerate(
                ["Timestamp", "Ad Name", "Confidence", "Signal"], 1):
            cell           = ws2.cell(row=2, column=col, value=header)
            cell.font      = white_bold
            cell.fill      = HEADER
            cell.alignment = centered
            cell.border    = border

        # Sort detections by timestamp
        sorted_dets = sorted(detections,
                            key=lambda x: x.get('timestamp_sec', 0))

        for i, det in enumerate(sorted_dets):
            ts    = det.get('timestamp_sec', 0)
            ts_str = f"{int(ts)//60:02d}:{int(ts)%60:02d}"
            conf  = f"{det.get('confidence', 0)*100:.1f}%"
            signal = det.get('signal', 'VIDEO')
            row_num = i + 3

            # Alternate row shading
            fill = GRAY if i % 2 == 0 else WHITE

            # Signal color
            sig_fill = GREEN if signal == 'BOTH' else \
                    YELLOW if signal == 'VIDEO' else ORANGE

            data = [ts_str, det['ad_id'], conf, signal]
            for col, value in enumerate(data, 1):
                cell           = ws2.cell(row=row_num, column=col,
                                        value=value)
                cell.alignment = centered
                cell.border    = border
                cell.fill      = fill
                if col == 4:
                    cell.fill = sig_fill
                    cell.font = Font(bold=True, color="FFFFFF")

        # Column widths sheet 2
        for i, width in enumerate([12, 30, 14, 12], 1):
            ws2.column_dimensions[get_column_letter(i)].width = width

        # ════════════════════════════════════════════
        # SHEET 3 — SUSPICIOUS AIRINGS
        # ════════════════════════════════════════════
        ws3       = wb.create_sheet("Suspicious Airings")
        ws3["A1"] = "SUSPICIOUS AIRINGS — Ads detected but NOT in bookings"
        ws3["A1"].font      = Font(bold=True, size=12, color="FFFFFF")
        ws3["A1"].fill      = ORANGE
        ws3["A1"].alignment = Alignment(horizontal="center",
                                        vertical="center")
        ws3.merge_cells("A1:C1")
        ws3.row_dimensions[1].height = 30

        for col, header in enumerate(
                ["Ad ID Detected", "Timestamp", "Confidence"], 1):
            cell           = ws3.cell(row=2, column=col, value=header)
            cell.font      = white_bold
            cell.fill      = HEADER
            cell.alignment = centered
            cell.border    = border

        # Find detections not in bookings
        booked_ids   = set(self.db.ad_metadata.keys())
        suspicious   = [d for d in detections
                        if d['ad_id'] not in booked_ids]

        if suspicious:
            for i, det in enumerate(suspicious):
                ts     = det.get('timestamp_sec', 0)
                ts_str = f"{int(ts)//60:02d}:{int(ts)%60:02d}"
                conf   = f"{det.get('confidence', 0)*100:.1f}%"
                row_num = i + 3
                for col, value in enumerate(
                        [det['ad_id'], ts_str, conf], 1):
                    cell           = ws3.cell(row=row_num,
                                            column=col, value=value)
                    cell.alignment = centered
                    cell.border    = border
                    cell.fill      = PatternFill("solid",
                                                fgColor="FFE0CC")
        else:
            ws3.merge_cells("A3:C3")
            ws3["A3"]           = "No suspicious airings detected ✓"
            ws3["A3"].font      = Font(bold=True, color="70AD47")
            ws3["A3"].alignment = Alignment(horizontal="center")

        for i, width in enumerate([30, 14, 14], 1):
            ws3.column_dimensions[get_column_letter(i)].width = width

        # ════════════════════════════════════════════
        # SHEET 4 — SYSTEM METRICS
        # ════════════════════════════════════════════
        ws4       = wb.create_sheet("System Metrics")
        ws4["A1"] = "SYSTEM PERFORMANCE METRICS"
        ws4["A1"].font      = Font(bold=True, size=14, color="FFFFFF")
        ws4["A1"].fill      = HEADER
        ws4["A1"].alignment = Alignment(horizontal="center",
                                        vertical="center")
        ws4.merge_cells("A1:B1")
        ws4.row_dimensions[1].height = 30

        # Count signals
        both_count  = sum(1 for d in detections
                        if d.get('signal') == 'BOTH')
        video_count = sum(1 for d in detections
                        if d.get('signal') == 'VIDEO')
        audio_count = sum(1 for d in detections
                        if d.get('signal') == 'AUDIO_ONLY')
        avg_conf_all = sum(d.get('confidence', 0)
                        for d in detections) / len(detections) \
                    if detections else 0

        metrics = [
            ("Total Ads in Database",     len(self.db.ad_metadata)),
            ("Total Detections",          len(detections)),
            ("Video + Audio Confirmed",   both_count),
            ("Video Only Detections",     video_count),
            ("Audio Only Detections",     audio_count),
            ("Average Confidence",        f"{avg_conf_all*100:.1f}%"),
            ("Fully Delivered Ads",       summary['delivered']),
            ("Not Aired Ads",             summary['missing']),
            ("Partial/Extra Ads",         summary['partial'] +
                                        summary['extra']),
            ("Suspicious Airings",        len(suspicious)
                                        if suspicious else 0),
            ("Processing Time",           f"{time.time()-start_time:.1f}s"),
            ("Model File",                "model_final.pth"),
        ]

        for i, (label, value) in enumerate(metrics):
            row_num = i + 2
            label_cell           = ws4.cell(row=row_num, column=1,
                                            value=label)
            label_cell.font      = bold
            label_cell.fill      = GRAY if i % 2 == 0 else WHITE
            label_cell.border    = border
            label_cell.alignment = left

            val_cell             = ws4.cell(row=row_num, column=2,
                                            value=value)
            val_cell.fill        = GRAY if i % 2 == 0 else WHITE
            val_cell.border      = border
            val_cell.alignment   = centered

        ws4.column_dimensions['A'].width = 30
        ws4.column_dimensions['B'].width = 20

        wb.save(output_path)
        print(f"\nReport saved: {output_path}")
        print(f"Sheets: Delivery Report, Timeline, "
            f"Suspicious Airings, System Metrics")
        print(f"Summary: {summary}")
        return output_path


if __name__ == "__main__":

    # ── STEP 1: Build database ──
    print("=" * 50)
    print("STEP 1: Building ad database")
    print("=" * 50)

    db = AdDatabase()

    import openpyxl
    bookings        = {}
    booking_details = {}

    if os.path.exists("bookings.xlsx"):
        wb_book = openpyxl.load_workbook("bookings.xlsx")
        ws_book = wb_book.active
        for row in ws_book.iter_rows(min_row=2, values_only=True):
            if row[0] is None:
                continue
            ad_name      = str(row[0]).strip()
            booked_slots = int(row[1])
            duration     = float(row[2]) if row[2] else 30
            bookings[ad_name] = booked_slots
            booking_details[ad_name] = {
                'booked_slots': booked_slots,
                'duration_sec': duration,
            }
        print(f"Loaded {len(bookings)} bookings from bookings.xlsx")
    else:
        print("No bookings.xlsx found — using 1 slot per ad")

    ads_raw_folder = "ads_raw"
    for video_file in os.listdir(ads_raw_folder):
        if not video_file.endswith('.mp4'):
            continue
        ad_id = os.path.splitext(video_file)[0]
        if ad_id not in bookings:
            print(f"Skipping {ad_id} — not in bookings.xlsx")
            continue
        video_path = os.path.join(ads_raw_folder, video_file)
        booked     = bookings[ad_id]
        db.register_ad(ad_id, video_path, booked_slots=booked)

    db.build_index()
    db.compute_ad_uniqueness()

    # Debug thresholds
    print("\nFinal thresholds:")
    for ad_id, thresh in sorted(db.ad_thresholds.items()):
        print(f"  {ad_id:<30} {thresh:.3f}")

    # ── STEP 2: Scan broadcast ──
    print("\n" + "=" * 50)
    print("STEP 2: Scanning broadcast video")
    print("=" * 50)

    broadcast_path = "broadcast_test.mp4"
    if not os.path.exists(broadcast_path):
        print(f"Error: {broadcast_path} not found!")
        exit()

    scanner    = BroadcastScanner(db)
    detections = scanner.scan(broadcast_path)

    print(f"\nTotal detections: {len(detections)}")
    for det in detections:
        ts = det.get('timestamp_sec', 0)
        print(f"  {det['ad_id']} at "
              f"{int(ts)//60:02d}:{int(ts)%60:02d} "
              f"| confidence: {det['confidence']:.2f} "
              f"| signal: {det['signal']}")

    # ── STEP 3: Generate report ──
    print("\n" + "=" * 50)
    print("STEP 3: Generating Excel report")
    print("=" * 50)

    reporter = DetectionReport(db)
    output   = reporter.generate(detections)
    print(f"Done! Open {output} to see your report.")
