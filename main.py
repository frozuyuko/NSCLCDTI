from NSCLCDTI.utils import *
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt 

datapath = './dataset/finaldata.csv'
dataset = NSCLC_Representation(datapath)


def custom_collate(batch):
    batch_fppubchem = torch.stack([item['fppubchem'] for item in batch])
    batch_scores = torch.stack([item['score'] for item in batch])
    drug_graphs = Batch.from_data_list([item['druggraph'] for item in batch])
    protein_graphs = Batch.from_data_list([item['proteingraph'] for item in batch])
    
    return {
        'fppubchem': batch_fppubchem,
        'score': batch_scores,
        'druggraph': drug_graphs,
        'proteingraph': protein_graphs
    }

train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=1)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNDTI().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000025, weight_decay=0.025)

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        druggraph = batch['druggraph'].to(device)
        proteingraph = batch['proteingraph'].to(device)
        fppubchem = batch['fppubchem'].to(device)
        score = batch['score'].to(device).unsqueeze(1)
        output = model(druggraph, proteingraph,fppubchem)
        loss = criterion(output, score)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(score)
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    total_loss = 0
    all_scores = []
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            druggraph = batch['druggraph'].to(device)
            proteingraph = batch['proteingraph'].to(device)
            fppubchem = batch['fppubchem'].to(device)
            score = batch['score'].to(device).unsqueeze(1)
            output = model(druggraph, proteingraph, fppubchem)
            loss = criterion(output, score)
            total_loss += loss.item() * len(score)
            pred = torch.sigmoid(output).cpu().numpy()
            all_scores.extend((torch.sigmoid(score) > 0.5).long().cpu().numpy()) 
            all_preds.extend(pred)
    
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    all_preds_bin = (all_preds >= 0.5).astype(int)
    auc_score = roc_auc_score(all_scores, all_preds)
    precision, recall, _ = precision_recall_curve(all_scores, all_preds)
    aupr_score = auc(recall, precision)
    accuracy = accuracy_score(all_scores, all_preds_bin)
    conf_matrix = confusion_matrix(all_scores, all_preds_bin)
    
    return total_loss / len(loader.dataset), accuracy, auc_score, aupr_score, conf_matrix

train_losses = []
test_losses = []
auc_scores = []
accuracies = []

for epoch in range(1, 26):
    train_loss = train()
    test_loss, accuracy, auc_score, aupr_score, conf_matrix = test(test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    auc_scores.append(auc_score)
    accuracies.append(accuracy)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, AUPR: {aupr_score:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    best_auc = 0
    if auc_score > best_auc:
        best_auc = auc_score
        torch.save(model.state_dict(), '~/model/GCNDTI_finalmodel_gonode_kmersedges.pth')




