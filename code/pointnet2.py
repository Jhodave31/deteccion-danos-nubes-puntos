"""PointNet++ (SSG) para segmentación semántica, implementado en PyTorch puro.

Basado en Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets
in a Metric Space" (NeurIPS 2017). Sin operadores CUDA personalizados: el muestreo
por punto más lejano (FPS), la agrupación por bola (ball query) y la interpolación
por distancia inversa se implementan con operaciones tensoriales estándar.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(a, b):
    """a: [B,N,3], b: [B,M,3] -> [B,N,M] distancias euclidianas al cuadrado."""
    return (a.pow(2).sum(-1, keepdim=True) - 2 * a @ b.transpose(1, 2) + b.pow(2).sum(-1).unsqueeze(1)).clamp_min(0)


def index_points(points, idx):
    """points: [B,N,C], idx: [B,...] -> [B,...,C]"""
    B = points.shape[0]
    batch = torch.arange(B, device=points.device).view(B, *([1] * (idx.dim() - 1))).expand_as(idx)
    return points[batch, idx]


@torch.no_grad()
def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    batch = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return centroids


@torch.no_grad()
def ball_query(radius, nsample, xyz, new_xyz):
    """Devuelve, para cada centro, hasta `nsample` vecinos dentro de `radius`
    (rellenando con el primer vecino encontrado, como en la implementación original)."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    sqr = square_distance(new_xyz, xyz)  # [B,S,N]
    idx = torch.arange(N, device=xyz.device).view(1, 1, N).expand(B, S, N).clone()
    idx[sqr > radius ** 2] = N
    idx = idx.sort(dim=-1)[0][:, :, :nsample]
    first = idx[:, :, :1].expand(-1, -1, nsample)
    mask = idx == N
    idx[mask] = first[mask]
    return idx


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_ch, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers, last = [], in_ch + 3
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feats):
        # xyz: [B,N,3], feats: [B,N,C] o None
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)                       # [B,S,3]
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)  # [B,S,K]
        grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
        grouped = grouped_xyz if feats is None else torch.cat([grouped_xyz, index_points(feats, idx)], -1)
        grouped = grouped.permute(0, 3, 2, 1)                      # [B,C+3,K,S]
        new_feats = self.mlp(grouped).max(2)[0].transpose(1, 2)    # [B,S,C']
        return new_xyz, new_feats


class FeaturePropagation(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        layers, last = [], in_ch
        for out in mlp:
            layers += [nn.Conv1d(last, out, 1, bias=False), nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, feats1, feats2):
        # interpola feats2 (en xyz2, más escaso) hacia xyz1 con los 3 vecinos más cercanos
        d = square_distance(xyz1, xyz2)
        d, idx = d.topk(3, dim=-1, largest=False)
        w = 1.0 / (d + 1e-8)
        w = w / w.sum(-1, keepdim=True)
        interp = (index_points(feats2, idx) * w.unsqueeze(-1)).sum(2)
        x = interp if feats1 is None else torch.cat([feats1, interp], -1)
        return self.mlp(x.transpose(1, 2)).transpose(1, 2)


class PointNet2Seg(nn.Module):
    def __init__(self, in_feats, num_classes, radii=(0.1, 0.2, 0.4, 0.8), npoints=(1024, 256, 64, 16), nsample=32, dropout=0.5):
        super().__init__()
        self.sa1 = SetAbstraction(npoints[0], radii[0], nsample, in_feats, [32, 32, 64])
        self.sa2 = SetAbstraction(npoints[1], radii[1], nsample, 64, [64, 64, 128])
        self.sa3 = SetAbstraction(npoints[2], radii[2], nsample, 128, [128, 128, 256])
        self.sa4 = SetAbstraction(npoints[3], radii[3], nsample, 256, [256, 256, 512])
        self.fp4 = FeaturePropagation(768, [256, 256])
        self.fp3 = FeaturePropagation(384, [256, 256])
        self.fp2 = FeaturePropagation(320, [256, 128])
        self.fp1 = FeaturePropagation(128 + in_feats, [128, 128, 128])
        self.head = nn.Sequential(nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Conv1d(128, num_classes, 1))

    def forward(self, xyz, feats):
        l1_xyz, l1 = self.sa1(xyz, feats)
        l2_xyz, l2 = self.sa2(l1_xyz, l1)
        l3_xyz, l3 = self.sa3(l2_xyz, l2)
        l4_xyz, l4 = self.sa4(l3_xyz, l3)
        l3 = self.fp4(l3_xyz, l4_xyz, l3, l4)
        l2 = self.fp3(l2_xyz, l3_xyz, l2, l3)
        l1 = self.fp2(l1_xyz, l2_xyz, l1, l2)
        l0 = self.fp1(xyz, l1_xyz, feats, l1)
        return self.head(l0.transpose(1, 2))  # [B,num_classes,N]


if __name__ == "__main__":
    m = PointNet2Seg(7, 5)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parámetros")
    x = torch.rand(2, 4096, 3)
    print(m(x, torch.rand(2, 4096, 7)).shape)
