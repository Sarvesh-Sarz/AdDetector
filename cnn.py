import torch
import torch.nn as nn
import torch.nn.functional as F

# ── BUILDING BLOCK ──────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        # If input and output shapes differ, we need a 1x1 conv to match them
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # ADD the original input back
        out = F.relu(out)
        return out


# ── FULL CNN ────────────────────────────────────────────────────
class AdFingerprintCNN(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        # Entry block: large 7x7 conv to capture broad features first
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # After entry: (64, 56, 56)

        # Residual blocks — progressively deeper features
        self.block1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        # After block1: (64, 56, 56)

        self.block2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        # After block2: (128, 28, 28)

        self.block3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        # After block3: (256, 14, 14)

        self.block4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512)
        )
        # After block4: (512, 7, 7)

        # Collapse spatial dimensions → single vector
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # After pool: (512, 1, 1)

        # Final embedding layer
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        # x shape: (batch, 3, 224, 224)
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)   # flatten: (batch, 512)
        x = self.fc(x)

        # L2 normalize — makes cosine similarity = dot product
        # All vectors will live on a unit hypersphere
        x = F.normalize(x, p=2, dim=1)
        return x


# ── TRIPLET LOSS ────────────────────────────────────────────────
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Distance between anchor and positive (same ad) — want this SMALL
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)

        # Distance between anchor and negative (different ad) — want this LARGE
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

        # Loss: push positives closer, negatives further
        # max(0, ...) means loss is 0 when already well separated
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()


# ── TEST IT ─────────────────────────────────────────────────────
if __name__ == "__main__":
    model = AdFingerprintCNN(embedding_dim=512)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass with fake data
    # Batch of 3 frames (anchor, positive, negative)
    dummy_input = torch.randn(3, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Verify L2 normalization — every vector should have length = 1.0
    norms = torch.norm(output, dim=1)
    print(f"Vector norms (should all be 1.0): {norms}")

    # Test triplet loss
    criterion = TripletLoss(margin=0.3)
    anchor   = output[0].unsqueeze(0)
    positive = output[1].unsqueeze(0)
    negative = output[2].unsqueeze(0)
    loss = criterion(anchor, positive, negative)
    print(f"Triplet loss (random data): {loss.item():.4f}")

    print("\nCNN architecture working perfectly!")
    print(model)
