import torch
import torch.nn.functional as F
from einops import rearrange


class SimCLRLossSlow:
    def __init__(self, batch_size, temperature):
        self.batch_size = batch_size
        self.temperature = temperature

    def __call__(self, vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
        table = self._get_table(vector1, vector2)
        similarity = self._get_similarity(table)
        loss = self._get_loss(similarity)
        return loss
    
    def _get_table(self, vector1, vector2):
        embed_dim = vector1.shape[1]
        table = torch.empty((2*self.batch_size, embed_dim))
        for i in range(self.batch_size):
            table[2*i] = vector1[i]
            table[2*i+1] = vector2[i]
        return table
    
    def _get_similarity(self, table):
        for i in range(2*self.batch_size):
            table[i] /= table[i].norm()
        similarity = torch.empty((2*self.batch_size, 2*self.batch_size))
        for i in range(2*self.batch_size):
            for j in range(2*self.batch_size):
                similarity[i][j] = (table[i]*table[j]).sum()
        return similarity
    
    def _get_loss(self, similarity):
        loss = torch.tensor(0, dtype=torch.float)
        for i in range(self.batch_size):
            similarity[2*i] /= self.temperature
            similarity[2*i][2*i] = torch.finfo(torch.float).min
            loss += (similarity[2*i].exp().sum().log() - similarity[2*i][2*i+1])
            similarity[2*i+1] /= self.temperature
            similarity[2*i+1][2*i+1] = torch.finfo(torch.float).min
            loss += (similarity[2*i+1].exp().sum().log()) - similarity[2*i+1][2*i]
        return loss / (2*self.batch_size)


class SimCLRLoss:
    def __init__(self, batch_size, temperature):
        self.batch_size = batch_size
        self.temperature = temperature

    def __call__(self, vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
        table = rearrange([vector1, vector2], "n b d -> (b n) d")
        similarity = self._get_similarity(table)
        loss = self._get_loss(similarity)
        return loss

    def _get_similarity(self, table: torch.Tensor) -> torch.Tensor:
        table = F.normalize(table, dim=1)
        similarity = torch.einsum("i d, j d -> i j", table, table)
        return similarity

    def _get_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        similarity.div_(self.temperature)
        similarity.fill_diagonal_(torch.finfo(torch.float).min)
        loss = F.cross_entropy(similarity, rearrange([torch.arange(1, 2*self.batch_size, 2), torch.arange(0, 2*self.batch_size, 2)], "n b -> (b n)"))
        return loss
