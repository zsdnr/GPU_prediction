import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


xy_max = 0
xy_min = 0

class DecisionTree(nn.Module):
    def __init__(self, num_features, max_depth=3):
        super(DecisionTree, self).__init__()
        self.max_depth = max_depth
        self.split_features = nn.ParameterList(
            [nn.Parameter(torch.randint(0, num_features, (1,)), requires_grad=False) for _ in
             range(2 ** max_depth - 1)])
        self.thresholds = nn.ParameterList(
            [nn.Parameter(torch.rand(1), requires_grad=True) for _ in range(2 ** max_depth - 1)])
        self.leaf_linear = nn.Linear(num_features, 1)

    def _build_tree(self, depth, index, x):
        if depth >= self.max_depth:
            return self.leaf_linear(x)

        split_feature = self.split_features[index]
        threshold = self.thresholds[index]
        x_feature = x[:, split_feature.item()]
        left_mask = x_feature < threshold
        right_mask = ~left_mask

        left_output = self._build_tree(depth + 1, 2 * index + 1, x[left_mask])
        right_output = self._build_tree(depth + 1, 2 * index + 2, x[right_mask])

        output = torch.zeros(x.size(0), 1).to(x.device)
        if left_mask.sum() > 0:
            output[left_mask] = left_output
        if right_mask.sum() > 0:
            output[right_mask] = right_output
        return output

    def forward(self, x):
        return self._build_tree(0, 0, x)



class XGBoostLikeModel(nn.Module):
    def __init__(self, num_features, num_trees=100, max_depth=30):
        super(XGBoostLikeModel, self).__init__()
        self.num_trees = num_trees
        self.trees = nn.ModuleList([DecisionTree(num_features, max_depth) for _ in range(num_trees)])
        self.weights = nn.Parameter(torch.randn(num_trees, 1))
        self.fc = nn.Sequential(
            nn.Linear(num_trees, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        tree_outputs = []
        for i, tree in enumerate(self.trees):
            tree_output = self.weights[i] * tree(x)
            tree_outputs.append(tree_output)

        tree_outputs = torch.cat(tree_outputs, dim=1)
        output = self.fc(tree_outputs)
        return output.squeeze()



class CloudDataset(Dataset):
    def __init__(self, df, cat_features, num_features, target, all):

        self.cat_dict = {}
        for col in cat_features:
            cat_col = df[col].astype('category')
            self.cat_dict[col] = dict(enumerate(cat_col.cat.categories))
            df[col] = cat_col.cat.codes

            num_categories = len(self.cat_dict[col])
            invalid_indices = df[col][(df[col] < 0) | (df[col] >= num_categories)].index
            if not invalid_indices.empty:
                df = df.drop(invalid_indices)
                df = df.reset_index(drop=True)


        self.scaler = StandardScaler()
        df[num_features] = self.scaler.fit_transform(df[num_features])

        self.cat_data = torch.LongTensor(df[cat_features].values)
        self.num_data = torch.FloatTensor(df[num_features].values)
        self.target = torch.FloatTensor(df[target].values)

        # =======================================================================================
        if all == 1:
            self.target_o = torch.FloatTensor(df[target].values)
            x_min = self.target.min(dim=0, keepdim=True)[0]
            x_max = self.target.max(dim=0, keepdim=True)[0]
            self.target = (self.target - x_min) / (x_max - x_min)
            print("self.target_o", self.target_o)
            print("self.target", self.target)

            print("all = 1")
            global xy_max
            xy_max = x_max
            global xy_min
            xy_min = x_min
        else :
            self.target = (self.target - xy_min) / (xy_max - xy_min)
        # =======================================================================================

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {
            'cat_features': self.cat_data[idx],
            'num_features': self.num_data[idx],
            'target': self.target[idx]
        }


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    criterion = nn.HuberLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            num_feats = batch['num_features'].to(device)
            targets = batch['target'].to(device)

            outputs = model(num_feats)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                num_feats = batch['num_features'].to(device)
                targets = batch['target'].to(device)

                outputs = model(num_feats)
                val_loss += criterion(outputs, targets).item() * len(targets)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'bestModel/best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


if __name__ == "__main__":

    df = pd.read_csv("train_data.csv")
    df['gpu_wrk_util'] = df['gpu_wrk_util'].fillna(0)
    df = df.dropna(subset=['gpu_ratio'])
    print("print(len(df))1", len(df))
    # --------------------------------------------------------------------------------------------------------
    df1 = pd.read_csv("test_data.csv")
    df1['gpu_wrk_util'] = df1['gpu_wrk_util'].fillna(0)
    df1 = df1.dropna(subset=['gpu_ratio'])

    df_columns = df.columns.tolist()
    print("df.df_columns", df_columns)
    df1_columns = df1.columns.tolist()
    print("df1.df_columns", df1_columns)


    if set(df.columns) != set(df1.columns):
        raise ValueError("The columns of df and df1 are not the same.")

    # --------------------------------------------------------------------------------------------------------
    categorical_features = ['task_name', 'gpu_type']
    numerical_features = ['inst_num', 'plan_gpu', 'avg_gpu_wrk_mem',
                          'mem_ratio', 'cpu_ratio']
    target = 'gpu_ratio'

    combined_df = pd.concat([df, df1], ignore_index=True)
    print("combined_df",len(combined_df))

    dataset = CloudDataset(combined_df, categorical_features, numerical_features, target,1)
    print("dataset", len(dataset))
    # --------------------------------------------------------------------------------------------------------
    train_dataset = torch.utils.data.Subset(dataset, list(range(len(df))))
    val_dataset = torch.utils.data.Subset(dataset, list(range(len(df), len(dataset))))
    print("train_dataset1", len(train_dataset))
    print("val_dataset1", len(val_dataset))
    # --------------------------------------------------------------------------------------------------------

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    num_features = len(numerical_features)
    model = XGBoostLikeModel(num_features, num_trees=20, max_depth=3)

    # train_model(model, train_loader, val_loader, epochs=200, lr=0.0005)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    y_true = []
    y_pred = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch in val_loader:
            num_feats = batch['num_features'].to(device)
            targets = batch['target'].to(device)

            outputs = model(num_feats)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)

    print(f"Final Validation RMSE: {rmse:.4f}")
    print(f"Final Validation MSE: {mse:.4f}")



