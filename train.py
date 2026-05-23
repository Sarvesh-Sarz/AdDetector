import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn import AdFingerprintCNN, TripletLoss
from dataset import TripletDataset

def train(epochs=50, batch_size=16, lr=1e-4):
    device     = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
    print(f"Training on: {device}")

    dataset    = TripletDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    model     = AdFingerprintCNN(embedding_dim=512).to(device)
    criterion = TripletLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate decay — reduce lr every 30 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1)

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
        scheduler.step()  # decay learning rate

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:03d}/{epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"LR: {current_lr:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       f"model_epoch_{epoch+1}.pth")
            print(f"Saved: model_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "model_final.pth")
    print("Training complete!")

if __name__ == "__main__":
    train()