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
