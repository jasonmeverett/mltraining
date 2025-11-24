import torch
import torch.nn as nn
 




class HeartDiseaseClassifier(nn.Module):

    def __init__(self, emb_dim=64, hidden_dim=1024, dropout=0.2):
        super().__init__()
        self.emb_sex = nn.Embedding(2, emb_dim)
        self.emb_cpt = nn.Embedding(4, emb_dim)
        self.emb_fbs = nn.Embedding(2, emb_dim)
        self.emb_recg = nn.Embedding(3, emb_dim)
        self.emb_exa = nn.Embedding(2, emb_dim)
        self.emb_sts = nn.Embedding(3, emb_dim)
                
        self.l1 = nn.Linear(6*emb_dim + 5, hidden_dim)
        self.r1 = nn.GELU()
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.r2 = nn.GELU()
        self.d2 = nn.Dropout(dropout)
        self.l2 = nn.Linear(hidden_dim, 10)
        self.r2 = nn.GELU()
        self.out = nn.Linear(10, 1)

    def forward(self, age, sex, cpt, rbp, cho, fbs, recg, mhr, exa, olp, sts):
        em_sex = self.emb_sex(sex)
        em_cpt = self.emb_cpt(cpt)
        em_fbs = self.emb_fbs(fbs)
        em_recg = self.emb_recg(recg)
        em_exa = self.emb_exa(exa)
        em_sts = self.emb_sts(sts)
        X = torch.cat([
            age.reshape(-1, 1),
            rbp.reshape(-1, 1),
            cho.reshape(-1, 1),
            mhr.reshape(-1, 1),
            olp.reshape(-1, 1),
            em_sex,
            em_cpt,
            em_fbs,
            em_recg,
            em_exa,
            em_sts
        ], dim=1)
        
        X = self.r1(self.l1(X))
        X = self.d1(X)
        X = self.r2(self.l2(X))
        X = self.d2(X)
        X = self.out(X)
        return X

    



