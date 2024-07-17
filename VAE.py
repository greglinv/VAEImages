import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(128 * 4 * 4, 256)
        self.fc3 = nn.Linear(256, 128 * 4 * 4)
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        z = self.fc3(z)
        z = z.view(z.size(0), 128, 4, 4)
        return self.decoder(z), mu, logvar

# Initialize model, loss, and optimizer
model = VAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop for VAE
def train_vae(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = criterion(recon_batch, data)  # Add KL divergence as well
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_vae(model, train_loader, criterion, optimizer)

def evaluate_vae(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            recon_batch, mu, logvar = model(data)
            test_loss += criterion(recon_batch, data).item()
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss}')

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
evaluate_vae(model, test_loader)

