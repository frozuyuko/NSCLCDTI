import os
import pandas as pd 
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, rdMolDescriptors, Descriptors
import deepchem as dc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class Kmers: 
    def __init__(self,sequence):
        self.sequence = sequence
    def getkmers(self):
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2)) #set max_features=512
        self.kmers = vectorizer.fit_transform(self.sequence).to_array()
        return self.kmers

class DrugLikeness: #For ADME Lipinski's Rule of Five
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Change this smiles: {smiles}")
    def hydrogendonor(self):
        return Lipinski.NumHDonors(self.mol)
    def hydrogenacceptor(self):
        return Lipinski.NumHAcceptors(self.mol)
    def molecularweight(self):
        return Descriptors.MolWt(self.mol)
    def logp(self):
        return Chem.Crippen.MolLogP(self.mol)

class NSCLC_Representation(Dataset):
    def __init__(self, datapath):
        super(NSCLC_Representation, self).__init__()
        self.data = pd.read_csv(datapath)
        smiles = self.data['smiles']
        self.fppubchem = pd.read_csv('./dataset/allsmiles_pubchem.csv').iloc[:, 1:] #can be precomputed from PadelPy or using PyFingerprint package
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        self.drug_graphs = [self.smilefeaturizer(smile, featurizer) for smile in smiles]
        #GO Information is obtained from tokenized gene ontology terms from uniprot
        #GO is obtained from uniprot go terms curation and binary tokenization on defined go term
        self.uniprotbp = pd.read_csv('./dataset/finalbp.csv')
        self.uniprotcc = pd.read_csv('./dataset/finalcc.csv')
        self.uniprotmf = pd.read_csv('./dataset/finalmf.csv')
        self.kmers = pd.read_csv('./dataset/twomersupd.csv', header=None).iloc[:, 1:513] 
        protein_names = self.data['protein']
        self.protein_encoded = pd.factorize(protein_names)[0]
        num_proteins = len(set(self.protein_encoded))
        self.protein_graphs = self.proteingraphsgenerator(num_proteins)
        self.scores = self.data['score'].values
    def smilefeaturizer(self, smile, featurizer):
        featurized = featurizer.featurize(smile)
        if featurized.size > 0 and hasattr(featurized[0], 'to_pyg_graph'):
            return featurized[0].to_pyg_graph()
        return None
    def proteingraphsgenerator(self, num_proteins):
        protein_graphs = []
        for i in range(num_proteins):
            edge_indices = self.kmers.iloc[i].dropna().astype(int).values
            edge_index = torch.tensor(edge_indices.reshape(-1, 2).T, dtype=torch.long)
            node_features = torch.cat([
                torch.tensor(self.uniprotbp.iloc[i].values).unsqueeze(0),
                torch.tensor(self.uniprotcc.iloc[i].values).unsqueeze(0),
                torch.tensor(self.uniprotmf.iloc[i].values).unsqueeze(0)
            ], dim=1)
            protein_graphs.append(Data(x=node_features, edge_index=edge_index))
        return protein_graphs
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        drug_graph = self.drug_graphs[index]
        protein_index = self.protein_encoded[index]
        protein_graph = self.protein_graphs[protein_index]
        return {
            'unigobp': torch.tensor(self.uniprotbp.iloc[index].values, dtype=torch.float),
            'unigocc': torch.tensor(self.uniprotcc.iloc[index].values, dtype=torch.float),
            'unigomf': torch.tensor(self.uniprotmf.iloc[index].values, dtype=torch.float),
            'kmers': torch.tensor(self.kmers.iloc[index].values, dtype=torch.float),
            'fppubchem': torch.tensor(self.fppubchem.iloc[index].values, dtype=torch.float),
            'score': torch.tensor(self.scores[index], dtype=torch.float),
            'druggraph': drug_graph,
            'proteingraph': protein_graph
        }
    
class GCNDTI(nn.Module):
    def __init__(self):
        super(GCNDTI, self).__init__()
        self.conv1 = GCNConv(30, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.fc_drug = nn.Linear(64, 128)
        self.fc_protein = nn.Linear(979, 128)
        concat_dim = 128 + 128 + 881
        self.fc1 = nn.Linear(concat_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
    def forward(self, druggraph, proteingraph,fppubchem):
        druggraph_x, druggraph_edge_index = druggraph.x.float(), druggraph.edge_index
        druggraph_out = F.relu(self.conv1(druggraph_x, druggraph_edge_index))
        druggraph_out = F.relu(self.conv2(druggraph_out, druggraph_edge_index))
        druggraph_out = F.relu(self.conv3(druggraph_out, druggraph_edge_index))
        druggraph_out = global_mean_pool(druggraph_out, druggraph.batch)
        druggraph_out = F.relu(self.fc_drug(druggraph_out))
        proteingraph_out = F.relu(self.fc_protein(proteingraph.x.float()))
        combined = torch.cat([druggraph_out, proteingraph_out,fppubchem], dim=1)
        combined = nn.LayerNorm(128+128+881)(combined)
        combined = nn.Dropout(0.3)(combined)
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        output = self.fc3(combined)
        return output
    
from PyFingerprint.fingerprint import get_fingerprint
class PubchemFp: #requires java available, if can't compile package then run java -jar from PadelPy
    def __init__(self,smiles):
        self.smiles = smiles 
    def getpubchemfp(self):
        self.pubchemfp = get_fingerprint(self.smiles, 'pubchem').to_numpy()
        return self.pubchemfp

