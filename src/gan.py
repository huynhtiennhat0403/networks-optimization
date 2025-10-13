import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# ========= CONFIG =========
DATA_PATH = "data/processed/train.csv"
OUTPUT_DIR = "data/synthetic"
MODEL_DIR = "models"
AUG_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "train_augmented.csv")
GEN_PATH = os.path.join(MODEL_DIR, "tabular_cgan_generator.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500  
BATCH_SIZE = 64  
NOISE_DIM = 64
HIDDEN_DIM = 256  
LR_G = 1e-4  
LR_D = 4e-4  
BETA1 = 0.5
LAMBDA_GP = 10  
LABEL_SMOOTH = 0.1  # Label smoothing
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========  MODEL =========
class Generator(nn.Module):
    def __init__(self, noise_dim, n_classes, out_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + n_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),  # Tanh thay vì Sigmoid để tránh vanishing gradient
        )
    
    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class Discriminator(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + n_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=1))


def compute_gradient_penalty(D, real_data, fake_data, labels):
    """WGAN-GP gradient penalty"""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=DEVICE)
    
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    d_interpolates = D(interpolates, labels)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def evaluate_synthetic_quality(real_X, real_y, syn_X, syn_y, features):
    """Đánh giá chất lượng synthetic data bằng ML classifier"""
    # Train classifier trên real data
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    clf.fit(real_X, real_y)
    
    # Test trên synthetic data
    y_pred = clf.predict(syn_X)
    
    print("\n" + "="*50)
    print("📊 SYNTHETIC DATA QUALITY EVALUATION")
    print("="*50)
    print("\nClassification Report on Synthetic Data:")
    print(classification_report(syn_y, y_pred, target_names=['Poor', 'Moderate', 'Good']))
    
    # Statistical comparison
    print("\n📈 Feature Statistics Comparison:")
    print(f"{'Feature':<30} {'Real Mean':<12} {'Syn Mean':<12} {'Diff':<10}")
    print("-" * 64)
    
    for i, feat in enumerate(features[:10]):  # Show first 10 features
        real_mean = real_X[:, i].mean()
        syn_mean = syn_X[:, i].mean()
        diff = abs(real_mean - syn_mean)
        print(f"{feat:<30} {real_mean:<12.4f} {syn_mean:<12.4f} {diff:<10.4f}")


# ========= TRAINING =========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df = pd.read_csv(DATA_PATH)
    target_col = "RF Link Quality"

    if target_col not in df.columns:
        raise ValueError("Không thấy cột RF Link Quality trong train.csv")

    # ---- Feature columns ----
    mod_cols = [c for c in df.columns if c.startswith("Modulation Scheme_")]
    cat_cols = ["Network Congestion"]
    cont_cols = [c for c in df.columns if c not in mod_cols + cat_cols + [target_col]]

    features = cont_cols + cat_cols + mod_cols
    print(f"Số feature: {len(features)} (cont={len(cont_cols)}, cat={len(cat_cols)}, onehot={len(mod_cols)})")

    # ---- Encode label ----
    label_map = {0.0: 0, 0.5: 1, 1.0: 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    y = df[target_col].map(label_map).astype(int).values

    # Scale data to [-1, 1] for Tanh
    X = df[features].values.astype(np.float32)
    X = 2 * X - 1  # [0,1] -> [-1,1]
    
    n_classes = len(label_map)

    # ---- Check imbalance ----
    vals, counts = np.unique(y, return_counts=True)
    print("\n📊 Original RF Link Quality distribution:")
    for v, c in zip(vals, counts):
        print(f"  Class {inv_label_map[v]} ({v}): {c} samples")

    X_tensor = torch.tensor(X, device=DEVICE)
    y_tensor = torch.tensor(y, device=DEVICE)
    
    # Stratified DataLoader để cân bằng batch
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    n_features = X.shape[1]

    # ---- Models ----
    G = Generator(NOISE_DIM, n_classes, n_features, hidden_dim=HIDDEN_DIM).to(DEVICE)
    D = Discriminator(n_features, n_classes, hidden_dim=HIDDEN_DIM).to(DEVICE)
    
    optG = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    optD = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    
    # BCEWithLogitsLoss tốt hơn BCE + Sigmoid
    loss_fn = nn.BCEWithLogitsLoss()

    print("\n🚀 Training  Conditional GAN...")
    print(f"Device: {DEVICE}")
    
    # Training history
    history = {'d_loss': [], 'g_loss': [], 'gp': []}
    
    for epoch in trange(EPOCHS, desc="Training"):
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_gp = []
        
        for Xb, yb in loader:
            bs = Xb.size(0)
            y_onehot = nn.functional.one_hot(yb, num_classes=n_classes).float()

            # ---- Train D ----
            D.zero_grad()
            
            # Label smoothing: real=0.9, fake=0.1
            real_label = torch.full((bs, 1), 1.0 - LABEL_SMOOTH, device=DEVICE)
            fake_label = torch.full((bs, 1), LABEL_SMOOTH, device=DEVICE)

            # Real
            out_real = D(Xb, y_onehot)
            loss_real = loss_fn(out_real, real_label)

            # Fake for D training
            z = torch.randn(bs, NOISE_DIM, device=DEVICE)
            
            # Sample fake labels proportional to real distribution
            fake_y = yb[torch.randperm(bs)]  # Shuffle real labels instead of random
            fake_y_oh = nn.functional.one_hot(fake_y, n_classes).float()
            
            fake_X_d = G(z, fake_y_oh).detach()  # Detach để không backprop vào G
            out_fake = D(fake_X_d, fake_y_oh)
            loss_fake = loss_fn(out_fake, fake_label)

            # Gradient penalty
            gp = compute_gradient_penalty(D, Xb, fake_X_d, y_onehot)
            
            loss_D = loss_real + loss_fake + LAMBDA_GP * gp
            loss_D.backward()
            optD.step()

            # ---- Train G ----
            G.zero_grad()
            
            # Generate NEW samples for G training
            z_g = torch.randn(bs, NOISE_DIM, device=DEVICE)
            fake_y_g = yb[torch.randperm(bs)]
            fake_y_oh_g = nn.functional.one_hot(fake_y_g, n_classes).float()
            
            fake_X_g = G(z_g, fake_y_oh_g)  # Không detach - cần gradient
            out_gen = D(fake_X_g, fake_y_oh_g)
            loss_G = loss_fn(out_gen, real_label)
            loss_G.backward()
            optG.step()
            
            # Record
            epoch_d_loss.append(loss_D.item())
            epoch_g_loss.append(loss_G.item())
            epoch_gp.append(gp.item())

        # Epoch summary
        avg_d = np.mean(epoch_d_loss)
        avg_g = np.mean(epoch_g_loss)
        avg_gp = np.mean(epoch_gp)
        
        history['d_loss'].append(avg_d)
        history['g_loss'].append(avg_g)
        history['gp'].append(avg_gp)
        
        if (epoch + 1) % 50 == 0:
            print(f"\nEpoch {epoch+1}/{EPOCHS}:")
            print(f"  D_loss={avg_d:.4f} | G_loss={avg_g:.4f} | GP={avg_gp:.4f}")

    torch.save(G.state_dict(), GEN_PATH)
    print(f"\n✅ Generator saved to {GEN_PATH}")

    # ---- Generate synthetic data ----
    print("\n🎨 Generating synthetic samples...")

    target_counts = {0: 2703, 1: 2000, 2: 2000}
    generated_rows = []

    G.eval()
    with torch.inference_mode():
        for cls in vals:
            current = counts[list(vals).index(cls)]
            target = target_counts.get(cls, current)
            need = max(0, target - current)
            
            print(f"Class {inv_label_map[cls]} ({cls}): {current} -> {target} (generate {need})")

            if need == 0:
                continue

            z = torch.randn(need, NOISE_DIM, device=DEVICE)
            c = nn.functional.one_hot(
                torch.full((need,), cls, device=DEVICE), 
                num_classes=n_classes
            ).float()
            
            gen = G(z, c).cpu().numpy()
            
            # Scale back to [0, 1]
            gen = (gen + 1) / 2

            gen_df = pd.DataFrame(gen, columns=features)
            gen_df[target_col] = inv_label_map[cls]
            generated_rows.append(gen_df)

    if not generated_rows:
        print("Dataset đã cân bằng, không cần sinh thêm.")
        return

    gen_df = pd.concat(generated_rows, ignore_index=True)

    # ---- Post-process ----
    def quantize_to_steps(series, valid_steps=[0.0, 0.5, 1.0]):
        arr = np.array(valid_steps)
        return series.apply(lambda x: arr[np.argmin(np.abs(arr - x))])

    gen_df["Network Congestion"] = quantize_to_steps(gen_df["Network Congestion"])

    # One-hot enforcement
    mod_array = gen_df[mod_cols].values
    argmax_idx = np.argmax(mod_array, axis=1)
    mod_array[:] = 0
    mod_array[np.arange(len(mod_array)), argmax_idx] = 1
    gen_df[mod_cols] = mod_array

    # Clip continuous
    gen_df[cont_cols] = gen_df[cont_cols].clip(0, 1)

    # ---- Evaluate quality ----
    X_real = df[features].values
    y_real = y
    X_syn = gen_df[features].values
    y_syn = gen_df[target_col].map(label_map).values
    
    evaluate_synthetic_quality(X_real, y_real, X_syn, y_syn, features)

    # ---- Combine & save ----
    # Scale real data back to [0,1] for combining
    df_original = df.copy()
    
    df_aug = pd.concat([df_original, gen_df], ignore_index=True)
    df_aug.to_csv(AUG_OUTPUT_PATH, index=False)

    print(f"\n✅ Saved augmented dataset to: {AUG_OUTPUT_PATH}")
    print("\n📊 Final class distribution:")
    for k, v in df_aug[target_col].value_counts().sort_index().items():
        print(f"  Class {k}: {v} samples")


if __name__ == "__main__":
    main()