import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix
import numpy as np

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


        self.label_encoder = LabelEncoder()
        df[target] = self.label_encoder.fit_transform(df[target])


        self.cat_data = torch.LongTensor(df[cat_features].values)
        self.num_data = torch.FloatTensor(df[num_features].values)
        self.target = torch.LongTensor(df[target].values)  # 分类任务

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {
            'cat_features': self.cat_data[idx],
            'num_features': self.num_data[idx],
            'target': self.target[idx]
        }



# --------------------------------------------------
class GPUUsageModel(nn.Module):
    def __init__(self, cat_dims, num_features, embedding_dim=16, num_heads=2, ff_dim=128, num_layers=2, num_classes=2):
        super().__init__()


        self.embeddings = nn.ModuleList()
        for dim in cat_dims:
            self.embeddings.append(nn.Embedding(dim, embedding_dim))

        total_embed_size = len(cat_dims) * embedding_dim
        print("total_embed_size", total_embed_size)
        input_size = total_embed_size + num_features
        print("num_features", num_features)
        print("input_size", input_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, cat_features, num_features):

        embeddings = []
        for i in range(len(self.embeddings)):
            emb = self.embeddings[i](cat_features[:, i])
            embeddings.append(emb)
        embedded = torch.cat(embeddings, dim=1)


        combined = torch.cat([embedded, num_features], dim=1).unsqueeze(1)


        combined = self.transformer_encoder(combined)
        combined = combined.squeeze(1)


        output = self.fc(combined)
        return output



# --------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float('inf')
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
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
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = 100 * correct / total


        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                cat_feats = batch['cat_features'].to(device)
                num_feats = batch['num_features'].to(device)
                targets = batch['target'].to(device)

                outputs = model(cat_feats, num_feats)
                val_loss += criterion(outputs, targets).item() * len(targets)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'bestModel/best_model_sort2.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break



# --------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv("train_sort.csv")
    df1 = pd.read_csv("test_sort.csv")

    df['gpu_wrk_util'] = df['gpu_wrk_util'].fillna(0)
    df = df.dropna(subset=['gpu_ratio'])


    df_columns = df.columns.tolist()
    print("df.df_columns", df_columns)

    df1_columns = df1.columns.tolist()
    print("df1.df_columns", df1_columns)

    if set(df.columns) != set(df1.columns):
        raise ValueError("The columns of df and df1 are not the same.")


    categorical_features = ['task_name', 'gpu_type', 'status']
    numerical_features = [
        'inst_num', 'cpu_usage', 'avg_mem', 'plan_cpu', 'plan_gpu',
        'cap_cpu', 'cap_gpu',
        'avg_gpu_wrk_mem',
        'mem_ratio', 'cpu_ratio', 'plan_mem', 'cap_mem'
    ]
    target = 'gpu_ratio'



    combined_df = pd.concat([df, df1], ignore_index=True)
    print("combined_df", len(combined_df))


    dataset = CloudDataset(combined_df, categorical_features, numerical_features, target)


    train_dataset = torch.utils.data.Subset(dataset, list(range(len(df))))
    val_dataset = torch.utils.data.Subset(dataset, list(range(len(df), len(dataset))))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    cat_dims = [len(dataset.cat_dict[col]) for col in categorical_features]

    num_classes = len(dataset.label_encoder.classes_)
    print("num_classes", num_classes)

    model = GPUUsageModel(
        cat_dims=cat_dims,
        num_features=len(numerical_features),
        embedding_dim=11,  # 16
        num_heads=5,  # 4
        ff_dim=128,
        num_layers=2,
        num_classes=num_classes
    )

    # train_model(model, train_loader, val_loader, epochs=100)

    model.load_state_dict(torch.load('best_model_sort2.pth'))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in val_loader:
            cat_feats = batch['cat_features']
            num_feats = batch['num_features']
            targets = batch['target']

            outputs = model(cat_feats, num_feats)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(targets.numpy())
            y_pred.extend(predicted.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

    classes = dataset.label_encoder.classes_
    print("classes", classes)


    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_percent = cm / row_sums * 100
#-----------------------------------------------------------------------------
    # 打印详细分类指标
    print("\n" + "=" * 60)
    print("Per-Class Detailed Metrics:")
    for i, class_name in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp + 1e-9)  # 避免除以0
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        print(f"\nClass {class_name} ({i}):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  TP: {tp:4d} | FP: {fp:4d} | FN: {fn:4d} | TN: {tn:4d}")
