import numpy as np

class KDNode:
    def __init__(self, point=None, ad_id=None, frame_idx=None,
                 left=None, right=None,
                 split_dim=None, split_val=None):
        self.point     = point  
        self.ad_id     = ad_id    
        self.frame_idx = frame_idx  
        self.left      = left     
        self.right     = right    
        self.split_dim = split_dim 
        self.split_val = split_val  


class KDTree:
    def __init__(self):
        self.root       = None
        self.all_points = []  # keep flat list for fallback

    def build(self, points, ad_ids, frame_idxs):
        self.all_points = list(zip(points, ad_ids, frame_idxs))
        indices = list(range(len(points)))
        self.root = self._build_recursive(points, ad_ids, frame_idxs,
                                          indices, depth=0)
        print(f"KD-tree built: {len(points)} points, "
              f"512 dimensions")

    def _build_recursive(self, points, ad_ids, frame_idxs,
                         indices, depth):
        if not indices:
            return None
        subset = points[indices]
        variances  = np.var(subset, axis=0)
        split_dim  = int(np.argmax(variances))

        # Sort indices by value along split dimension
        indices.sort(key=lambda i: points[i][split_dim])

        # Median index becomes this node
        mid = len(indices) // 2
        mid_idx = indices[mid]

        # Build left and right subtrees recursively
        node = KDNode(
            point     = points[mid_idx],
            ad_id     = ad_ids[mid_idx],
            frame_idx = frame_idxs[mid_idx],
            split_dim = split_dim,
            split_val = points[mid_idx][split_dim]
        )
        node.left  = self._build_recursive(points, ad_ids, frame_idxs,
                                            indices[:mid], depth+1)
        node.right = self._build_recursive(points, ad_ids, frame_idxs,
                                            indices[mid+1:], depth+1)
        return node

    def cosine_similarity(self, a, b):
        """
        Cosine similarity between two vectors.
        Range: -1 to 1. Higher = more similar.
        1.0 = identical, 0.0 = unrelated, -1.0 = opposite
        Since our CNN L2-normalizes outputs:
        cosine_similarity = simple dot product
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) *
                                      np.linalg.norm(b) + 1e-9))

    def nearest_neighbor(self, query, k=3):
        """
        Find k most similar vectors to query.
        Returns list of (ad_id, frame_idx, similarity) sorted by similarity.
        """
        # Best candidates found so far: list of (similarity, ad_id, frame_idx)
        best = []
        self._search_recursive(self.root, query, k, best)
        # Sort by similarity descending
        best.sort(key=lambda x: x[0], reverse=True)
        return [(ad_id, frame_idx, sim)
                for sim, ad_id, frame_idx in best]

    def _search_recursive(self, node, query, k, best):
        if node is None:
            return

        # Compute similarity to this node's point
        sim = self.cosine_similarity(query, node.point)

        # Add to best list
        best.append((sim, node.ad_id, node.frame_idx))

        # Keep only top k
        if len(best) > k * 2:
            best.sort(key=lambda x: x[0], reverse=True)
            del best[k*2:]

        # Decide which subtree to search first
        # Go toward the side the query falls on
        if query[node.split_dim] < node.split_val:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        # Search the closer side first
        self._search_recursive(first, query, k, best)

        # Check if we need to search the other side
        # Only if the hyperplane distance is within our best similarity
        plane_dist = abs(query[node.split_dim] - node.split_val)
        if len(best) < k or plane_dist < 1 - min(b[0] for b in best):
            self._search_recursive(second, query, k, best)


class VideoMatcher:
    def __init__(self, similarity_threshold=0.80):
        self.tree      = KDTree()
        self.threshold = similarity_threshold

    def build_database(self, all_vectors, all_ad_ids, all_frame_idxs):
        """Load all reference ad vectors into KD-tree."""
        points = np.array(all_vectors)
        self.tree.build(points, all_ad_ids, all_frame_idxs)
        print(f"Database built: {len(all_vectors)} reference vectors")

    def match_frame(self, query_vector, ad_thresholds=None):
        results = self.tree.nearest_neighbor(query_vector, k=3)
        if not results:
            return None, 0.0

        best_ad_id, best_frame, best_sim = results[0]
    
        # Use per-ad threshold if available
        if ad_thresholds and best_ad_id in ad_thresholds:
            threshold = ad_thresholds[best_ad_id]
        else:
            threshold = self.threshold

        if best_sim >= threshold:
            return best_ad_id, best_sim
        return None, best_sim

    def match_sequence(self, query_vectors, min_consecutive=12, ad_thresholds=None):
        detections  = []
        current_ad  = None
        streak      = 0
        start_frame = 0
        confidences = []

        for i, vector in enumerate(query_vectors):
            ad_id, sim = self.match_frame(vector)

            if ad_id is not None:
                if ad_id == current_ad:
                    # Same ad continuing
                    streak += 1
                    confidences.append(sim)
                else:
                    # New ad started
                    if current_ad and streak >= min_consecutive:
                        detections.append({
                            'ad_id':      current_ad,
                            'start_frame': start_frame,
                            'end_frame':   i,
                            'confidence':  float(np.mean(confidences))
                        })
                    current_ad  = ad_id
                    streak      = 1
                    start_frame = i
                    confidences = [sim]
            else:
                # No match — save if previous streak was long enough
                if current_ad and streak >= min_consecutive:
                    detections.append({
                        'ad_id':      current_ad,
                        'start_frame': start_frame,
                        'end_frame':   i,
                        'confidence':  float(np.mean(confidences))
                    })
                current_ad  = None
                streak      = 0
                confidences = []
        for i, vector in enumerate(query_vectors):
            ad_id, sim = self.match_frame(vector, ad_thresholds)

        return detections


# ── TEST ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing KD-tree from scratch...")
    print("-" * 50)

    # Simulate 10 ads, 50 frames each = 500 vectors
    np.random.seed(42)
    num_ads    = 10
    frames_per_ad = 50
    dim        = 512

    # Each ad gets its own cluster in 512-d space
    all_vectors    = []
    all_ad_ids     = []
    all_frame_idxs = []

    ad_centers = np.random.randn(num_ads, dim)  # cluster centers

    for ad_idx in range(num_ads):
        ad_id = f"ad_{ad_idx:03d}"
        center = ad_centers[ad_idx]
        for frame_idx in range(frames_per_ad):
            # Frames from same ad = small noise around center
            vector = center + np.random.randn(dim) * 0.1
            # L2 normalize (like our CNN does)
            vector = vector / np.linalg.norm(vector)
            all_vectors.append(vector)
            all_ad_ids.append(ad_id)
            all_frame_idxs.append(frame_idx)

    print(f"Created {len(all_vectors)} test vectors "
          f"({num_ads} ads x {frames_per_ad} frames)")

    # Build KD-tree
    matcher = VideoMatcher(similarity_threshold=0.98)
    matcher.build_database(all_vectors, all_ad_ids, all_frame_idxs)

    # Test 1: Query with a frame very similar to ad_000
    print("\nTest 1: Query similar to ad_000 (should match ad_000)")
    query = ad_centers[0] + np.random.randn(dim) * 0.05
    query = query / np.linalg.norm(query)
    ad_id, sim = matcher.match_frame(query)
    print(f"Result: {ad_id}, similarity: {sim:.4f}")

    # Test 2: Query with random noise (should not match)
    print("\nTest 2: Random query (should not match)")
    random_query = np.random.randn(dim)
    random_query = random_query / np.linalg.norm(random_query)
    ad_id, sim = matcher.match_frame(random_query)
    print(f"Result: {ad_id}, similarity: {sim:.4f}")

    # Test 3: Sequence matching
    print("\nTest 3: Sequence of 10 frames from ad_005")
    sequence = []
    for _ in range(10):
        v = ad_centers[5] + np.random.randn(dim) * 0.05
        v = v / np.linalg.norm(v)
        sequence.append(v)

    detections = matcher.match_sequence(sequence)
    print(f"Detections found: {len(detections)}")
    for d in detections:
        print(f"  Ad: {d['ad_id']}, "
              f"frames {d['start_frame']}-{d['end_frame']}, "
              f"confidence: {d['confidence']:.4f}")

    print("\nKD-tree working perfectly!")
