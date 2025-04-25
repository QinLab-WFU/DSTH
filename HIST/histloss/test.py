from hist import CDs2Hg,HGNN

hgnn = HGNN(nb_classes=10,hidden=512,sz_embed=16)

cds2hg = CDs2Hg(nb_classes=10,sz_embed=16)

import torch

a = torch.randn((16,16))
b = torch.randn((16,10))
dist_loss , H = cds2hg(a,b)

