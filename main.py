import torch
import numpy as np
import random
from torch import nn, optim
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy import sparse
from utils.config import CFG
from model.GNN import GCN, GAT, GraphSAGE, GIN
from dataset import photo, questions, tolokers, weibo

cfg = CFG()

cfg.DATASET = "weibo"
cfg.MODEL = "GCN"
cfg.GNN_LAYERS = 2
cfg.FEATURE_OUT = 2

def seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Dataset info: https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection (download the .mat format of dataset from this website)
seed(cfg.SEED)
features_size = [("weibo",400),  # (dataset_name, feature_size)
                 ("tolokers",10),
                 ("photo",745),
                 ("questions",301)
                ]

if cfg.MODEL=="GCN":  
    gnn_selected = GCN
elif cfg.MODEL == "GAT":
    gnn_selected = GAT
elif cfg.MODEL == "SAGE":
    gnn_selected = GraphSAGE
elif cfg.MODEL == "GIN":
    gnn_selected = GIN
else:
    print("Model not found!!")

print(f"Using {cfg.MODEL} model")

if cfg.DATASET == "weibo":
    X_train, train_edge_index, y_train, X_test, test_edge_index, y_test = weibo.load_weibo()
    attr_size = features_size[0][1]
elif cfg.DATASET == "tolokers":
    X_train, train_edge_index, y_train, X_test, test_edge_index, y_test = tolokers.load_weibo()
    attr_size = features_size[1][1]
elif cfg.DATASET == "photo":
    X_train, train_edge_index, y_train, X_test, test_edge_index, y_test= photo.load_weibo()
    attr_size = features_size[1][1]
elif cfg.DATASET == "questions":
    X_train, train_edge_index, y_train, X_test, test_edge_index, y_test= questions.load_weibo()
    attr_size = features_size[1][1]
else:
    print("Please select the available dataset!!")

print(f"Using {cfg.DATASET} model")
X_train = X_train.toarray() 
X_train = torch.tensor(X_train, dtype = torch.float32).to(cfg.DEVICE)
X_test = X_test.toarray() 
X_test = torch.tensor(X_test, dtype = torch.float32).to(cfg.DEVICE)
y_train = torch.tensor(y_train).to(cfg.DEVICE)
y_test = torch.tensor(y_test).to(cfg.DEVICE)
train_edge_index = train_edge_index.to(cfg.DEVICE)
test_edge_index = test_edge_index.to(cfg.DEVICE)

model = gnn_selected(cfg.GNN_LAYERS,attr_size, cfg.FEATURE_OUT).to(cfg.DEVICE)
opt = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay = 1e-5)
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
history = {
    "train_loss": [], "train_acc": [],
    "eval_loss": [], "eval_acc": [], "eval_f1": [], "eval_auc": []
}


def evaluate(model, X_test, test_edge_index, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test, test_edge_index).squeeze()           
        loss = criterion(logits, y_test.float())

        probs = torch.sigmoid(logits)                         
        preds = (probs >= 0.5).long()

        yt = y_test.detach().cpu().numpy()
        yp = preds.detach().cpu().numpy()
        pr = probs.detach().cpu().numpy()

        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="binary")
        auc = roc_auc_score(yt, pr)
        metrics = {"loss": loss.item(), "acc": acc, "f1": f1, "auc": auc}
    return metrics


for epoch in range(1, cfg.EPOCHS + 1):
    model.train()
    opt.zero_grad()

    logits = model(X_train, train_edge_index).squeeze()  
    loss = criterion(logits, y_train.float())
    loss.backward()
    opt.step()

    with torch.no_grad(): # No grad, just for GPU memory efficiency
        probs_tr = torch.sigmoid(logits)
        preds_tr = (probs_tr >= 0.5).long()
        acc_tr = (preds_tr == y_train).float().mean().item()

    history["train_loss"].append(loss.item())
    history["train_acc"].append(acc_tr)

    # Evaluate every 5 epochs
    if epoch % 5 == 0:
        metrics = evaluate(model, X_test, test_edge_index, y_test)
        history["eval_loss"].append(metrics["loss"])
        history["eval_acc"].append(metrics["acc"])
        history["eval_f1"].append(metrics["f1"])
        history["eval_auc"].append(metrics["auc"])

        print(f"[{epoch:03d}] "
              f"train_loss={loss.item():.4f} train_acc={acc_tr:.4f} | "
              f"eval_loss={metrics['loss']:.4f} eval_acc={metrics['acc']:.4f} "
              f"eval_f1={metrics['f1']:.4f} eval_auc={metrics['auc']:.4f}")
    else:
        print(f"[{epoch:03d}] train_loss={loss.item():.4f} train_acc={acc_tr:.4f}")




