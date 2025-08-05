import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os
import wandb

from model import Conditional_UNet
from dataloader import PolygonDataset
from utils import extract_named_colors, weighted_bce_loss

def main(args):
    wandb.init(project=args.project_name)

    train_color_map = extract_named_colors(args.train_json, base_path=os.path.join(args.data_path, 'training/outputs'))
    colour_map = {name: torch.tensor(tensor_color.tolist()) for name, tensor_color in train_color_map.items()}

    model = Conditional_UNet(in_ch=1, out_ch=3, c_embd=1024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = PolygonDataset(base_path=args.data_path, colour_map=colour_map, split='training', transform=transform)
    val_dataset = PolygonDataset(base_path=args.data_path, colour_map=colour_map, split='validation', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_imgs = batch['input'].to(device)
            colors = batch['colour'].to(device)
            target_imgs = batch['output'].to(device)

            outputs = model(input_imgs, colors)
            loss = weighted_bce_loss(outputs, target_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch['input'].to(device)
                val_colors = val_batch['colour'].to(device)
                val_targets = val_batch['output'].to(device)

                val_outputs = model(val_inputs, val_colors)
                loss = weighted_bce_loss(val_outputs, val_targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            def get_samples(loader, name):
                samples = []
                for i, batch in enumerate(loader):
                    if i >= 1: break
                    inputs = batch['input'].to(device)
                    colors = batch['colour'].to(device)
                    targets = batch['output'].to(device)
                    outputs = model(inputs, colors)

                    for j in range(min(2, inputs.shape[0])):
                        img_input = inputs[j].detach().cpu().numpy().transpose(1, 2, 0).astype('float32')
                        gt = targets[j].detach().cpu().numpy().transpose(1, 2, 0).astype('float32')
                        pred = outputs[j].detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1).astype('float32')
                        samples.extend([
                            wandb.Image(img_input, caption=f"{name} Sample {j+1} - Input"),
                            wandb.Image(gt, caption=f"{name} Sample {j+1} - Ground Truth"),
                            wandb.Image(pred, caption=f"{name} Sample {j+1} - Prediction"),
                        ])
                return samples

            wandb.log({
                "Train Samples": get_samples(train_loader, "Train"),
                "Val Samples": get_samples(val_loader, "Val"),
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Conditional UNet")
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--train_json', type=str, default='./data/training/data.json')
    parser.add_argument('--val_json', type=str, default='./data/validation/data.json')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='c_unet_weighted_bce_rgb.pth')
    parser.add_argument('--project_name', type=str, default='conditional-unet')
    args = parser.parse_args()

    main(args)
