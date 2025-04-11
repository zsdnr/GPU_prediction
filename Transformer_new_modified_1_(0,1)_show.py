import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



# --------------------------------------------------
class CloudDataset(Dataset):
    def __init__(self, df, cat_features, num_features, target):

        self.cat_dict = {}
        for col in cat_features:

            cat_col = df[col].astype('category')

            self.cat_dict[col] = dict(enumerate(cat_col.cat.categories))
            print("self.cat_dict[col]", self.cat_dict[col])

            df[col] = cat_col.cat.codes
            print("df[col]", df[col])


            num_categories = len(self.cat_dict[col])

            invalid_indices = df[col][(df[col] < 0) | (df[col] >= num_categories)].index
            if not invalid_indices.empty:
                print(f"Removing rows with invalid values in column {col}: {invalid_indices.tolist()}")
                df = df.drop(invalid_indices)
                df = df.reset_index(drop=True)


            max_index = df[col].max()
            num_categories = len(self.cat_dict[col])
            if max_index >= num_categories:
                raise ValueError(
                    f"Index out of range in column {col}. Max index: {max_index}, Num categories: {num_categories}")


        self.scaler = StandardScaler()
        print("df[num_features]1", df[num_features])
        df[num_features] = self.scaler.fit_transform(df[num_features])
        print("df[num_features]2", df[num_features])


        self.cat_data = torch.LongTensor(df[cat_features].values)
        self.num_data = torch.FloatTensor(df[num_features].values)
        self.target = torch.FloatTensor(df[target].values)  # 回归任务

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {
            'cat_features': self.cat_data[idx],
            'num_features': self.num_data[idx],
            'target': self.target[idx]
        }



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out_proj(attn_output)
        return output



class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x



class CrossLayerAdaptiveAttentionFusion(nn.Module):
    def __init__(self, num_layers, embed_dim):
        super(CrossLayerAdaptiveAttentionFusion, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.gating_mechanisms = nn.ModuleList([
            nn.Linear(embed_dim, num_layers) for _ in range(num_layers)
        ])

    def forward(self, layer_outputs):
        batch_size, seq_length, _ = layer_outputs[0].size()
        fused_output = torch.zeros_like(layer_outputs[0])

        for i in range(self.num_layers):
            gating_weights = torch.nn.functional.softmax(self.gating_mechanisms[i](layer_outputs[i]), dim=-1)
            for j in range(self.num_layers):
                fused_output += gating_weights[:, :, j:j + 1] * layer_outputs[j]

        return fused_output



class CLATransformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout):
        super(CLATransformer, self).__init__()
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.cross_layer_fusion = CrossLayerAdaptiveAttentionFusion(num_layers, embed_dim)

    def forward(self, x):
        layer_outputs = []
        for layer in self.transformer_layers:
            x = layer(x)
            layer_outputs.append(x)

        fused_output = self.cross_layer_fusion(layer_outputs)
        return fused_output



# --------------------------------------------------
class CompleteCLATransformerModel(nn.Module):
    def __init__(self, cat_dims, num_features, embedding_dim=16, d_model=256, num_layers=3, num_heads=8, hidden_dim=512,
                 dropout=0.1):
        super().__init__()


        self.embeddings = nn.ModuleList()
        for dim in cat_dims:
            self.embeddings.append(nn.Embedding(dim, embedding_dim))

        total_embed_size = len(cat_dims) * embedding_dim
        input_size = total_embed_size + num_features

        self.input_proj = nn.Linear(input_size, d_model)

        self.cla_transformer = CLATransformer(
            num_layers=num_layers,
            embed_dim=d_model,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, cat_features, num_features):

        embeddings = []
        for i in range(len(self.embeddings)):
            emb = self.embeddings[i](cat_features[:, i])
            embeddings.append(emb)
        embedded = torch.cat(embeddings, dim=1)


        combined = torch.cat([embedded, num_features], dim=1)


        src = self.input_proj(combined).unsqueeze(1)


        output = self.cla_transformer(src)


        output = output[:, -1, :]


        output = self.fc(output)
        return output.squeeze()



# --------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=200, lr=0.0005, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = model.to(device)


    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float('inf')
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            cat_feats = batch['cat_features'].to(device)
            num_feats = batch['num_features'].to(device)
            targets = batch['target'].to(device)

            outputs = model(cat_feats, num_feats)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                cat_feats = batch['cat_features'].to(device)
                num_feats = batch['num_features'].to(device)
                targets = batch['target'].to(device)

                outputs = model(cat_feats, num_feats)
                val_loss += criterion(outputs, targets).item() * len(targets)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'bestModel/gpu_ratio_prediction_CLA_Transformer_best_model_##.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break



# --------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv("train_data.csv")
    df['gpu_wrk_util'] = df['gpu_wrk_util'].fillna(0)
    df = df.dropna(subset=['gpu_ratio'])
    print("df.len()",len(df))
    # ---------------------------------------------------------
    df1 = pd.read_csv("test_data.csv")
    df1['gpu_wrk_util'] = df1['gpu_wrk_util'].fillna(0)
    df1 = df1.dropna(subset=['gpu_ratio'])
    print("df1.len()",len(df1))

    categorical_features = ['task_name', 'gpu_type'
                            # , 'status'
                            ]
    numerical_features = [
        'inst_num',
        'plan_gpu', 'avg_gpu_wrk_mem',
        'mem_ratio', 'cpu_ratio'
    ]
    target = 'gpu_ratio'

    

    combined_df = pd.concat([df, df1], ignore_index=True)
    print("combined_df",len(combined_df))

    dataset = CloudDataset(combined_df, categorical_features, numerical_features, target)

    train_dataset = torch.utils.data.Subset(dataset, list(range(len(df))))
    val_dataset = torch.utils.data.Subset(dataset, list(range(len(df), len(dataset))))
    print("len(val_dataset)",len(val_dataset))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    cat_dims = [len(dataset.cat_dict[col]) for col in categorical_features]
    print(f"cat_dims: {cat_dims}")

    model = CompleteCLATransformerModel(
        cat_dims=cat_dims,
        num_features=len(numerical_features),
        embedding_dim=16
    )

    # train_model(model, train_loader, val_loader, epochs=200, lr=0.0005, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device1", device)

    model.load_state_dict(torch.load('gpu_ratio_prediction_CLA_Transformer_best_model.pth'))
    # ---------------
    model = model.to(device)
    # ---------------------------------------------
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in val_loader:
            cat_feats = batch['cat_features'].to(device)
            num_feats = batch['num_features'].to(device)
            targets = batch['target'].to(device)

            outputs = model(cat_feats, num_feats)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Final Validation RMSE: {rmse:.4f}")



