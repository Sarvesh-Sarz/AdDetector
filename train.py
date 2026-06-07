import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn import AdFingerprintCNN, TripletLoss
from dataset import TripletDataset

def train(epochs=100, batch_size=16, lr=1e-4):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset    = TripletDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    model     = AdFingerprintCNN(embedding_dim=512).to(device)
    criterion = TripletLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(
      #  optimizer, step_size=20, gamma=0.5)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6)

    best_loss     = float('inf')
    patience      = 8  # stop if no improvement for 8 epochs
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches    = 0

        for anchor, positive, negative in dataloader:
            anchor   = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb   = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches    += 1

        avg_loss = total_loss / batches
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{epochs}] Loss: {avg_loss:.4f}")

        # ── BEST MODEL CHECKPOINT ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "model_best.pth")
            print(f"  ✓ New best model saved (loss: {best_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        # ── EARLY STOPPING ──
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Use best model as final
    import shutil
    shutil.copy("model_best.pth", "model_final.pth")
    print(f"Training complete! Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()
