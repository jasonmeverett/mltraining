import torch
import torch.nn as nn
 




class HeartDiseaseClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb_sex = nn.Embedding(2, 64)
        self.emb_cpt = nn.Embedding(4, 64)
        self.emb_fbs = nn.Embedding(2, 64)
        self.emb_recg = nn.Embedding(3, 64)
        self.emb_exa = nn.Embedding(2, 64)
        self.emb_sts = nn.Embedding(3, 64)
                
        self.l1 = nn.Linear(6*64 + 5, 1024)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(1024, 1024)
        self.r2 = nn.ReLU()
        self.out = nn.Linear(1024, 1)

    def forward(self, age, sex, cpt, rbp, cho, fbs, recg, mhr, exa, olp, sts):
        em_sex = self.emb_sex(sex.int())
        em_cpt = self.emb_cpt(cpt.int())
        em_fbs = self.emb_fps(fbs.int())
        em_recg = self.emb_recg(recg.int())
        em_exa = self.emb_exa(exa.int())
        em_sts = self.emb_sts(sts.int())
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
        X = self.r2(self.l2(X))
        X = self.out(X)
        return X

    



