"""PATH-SYN model: pathway tokens (cell) x drug tokens, fused by cross-attention.

Cell side  : per-modality encoders turn the (P,3) pathway features [RNA, CNV, MUT] into P tokens (+ pathway-id
             embedding; + a learned 'targeted by drug A/B' embedding on pathways hit by each drug's targets),
             followed by a small self-attention encoder over pathways.
Drug side  : fingerprint(+target pathway vector) -> drug tokens; queries = [CLS, drugA, drugB].
Fusion     : N blocks of  self-attn(queries) -> CROSS-attn(queries -> pathway tokens) -> FFN.
Heads      : Bliss, HSA, synergy logit (pair pass) and monotherapy sensitivity (single-drug pass: [CLS, drug]).
"""
import torch
import torch.nn as nn


class CrossBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, mem, need_w=False):
        h = self.ln1(q)
        q = q + self.drop(self.sa(h, h, h, need_weights=False)[0])
        h = self.ln2(q)
        a, w = self.ca(h, mem, mem, need_weights=need_w, average_attn_weights=True)
        q = q + self.drop(a)
        q = q + self.drop(self.ffn(self.ln3(q)))
        return q, w


class PathSyn(nn.Module):
    key = "feats"          # which batch tensor holds the cell representation

    def __init__(self, n_path, drug_dim, d=256, heads=4, layers=3, dropout=0.2,
                 mods=(1, 1, 1), path_enc_layers=1):
        super().__init__()
        self.d = d
        self.register_buffer("mod_mask", torch.tensor(mods, dtype=torch.float32))
        self.mod_enc = nn.ModuleList([nn.Sequential(nn.Linear(1, d // 2), nn.GELU(), nn.Linear(d // 2, d))
                                      for _ in range(3)])
        self.path_id = nn.Embedding(n_path, d)
        self.tgt_tok = nn.Parameter(torch.randn(2, d) * 0.02)       # pathway targeted by drug A / B
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.path_enc = nn.TransformerEncoder(enc, path_enc_layers, enable_nested_tensor=False)

        self.drug_enc = nn.Sequential(nn.Linear(drug_dim, 512), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(512, d), nn.LayerNorm(d))
        self.tgt_proj = nn.Linear(n_path, d)
        self.role = nn.Embedding(3, d)                              # CLS / A / B
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([CrossBlock(d, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head_bliss, self.head_hsa = nn.Linear(d, 1), nn.Linear(d, 1)
        self.head_cls, self.head_mono = nn.Linear(d, 1), nn.Linear(d, 1)

    # -- building blocks -------------------------------------------------------------------
    def pathway_tokens(self, feats, tgts):
        tok = self.path_id.weight.unsqueeze(0).expand(feats.size(0), -1, -1)
        for m in range(3):
            if self.mod_mask[m] > 0:
                tok = tok + self.mod_enc[m](feats[..., m:m + 1])
        for r, t in enumerate(tgts):
            tok = tok + t.unsqueeze(-1) * self.tgt_tok[r]
        return self.path_enc(tok)

    def _fuse(self, feats, drugs, return_attn=False):
        """drugs: list of (x, t) -- 2 entries for a pair, 1 for a single drug."""
        B = feats.size(0)
        mem = self.pathway_tokens(feats, [t for _, t in drugs])
        q = [self.cls.expand(B, -1, -1) + self.role.weight[0]]
        for r, (x, t) in enumerate(drugs):
            q.append((self.drug_enc(x) + self.tgt_proj(t) + self.role.weight[r + 1]).unsqueeze(1))
        q, w = torch.cat(q, 1), None
        for i, blk in enumerate(self.blocks):
            q, w = blk(q, mem, need_w=return_attn and i == len(self.blocks) - 1)
        return self.norm(q).mean(1), w

    # -- public API ---------------------------------------------------------------------------
    def forward(self, b, return_attn=False):
        z, w = self._fuse(b["feats"], [(b["xa"], b["ta"]), (b["xb"], b["tb"])], return_attn)
        out = {"bliss": self.head_bliss(z).squeeze(-1), "hsa": self.head_hsa(z).squeeze(-1),
               "logit": self.head_cls(z).squeeze(-1)}
        if return_attn:
            out["attn"] = w                       # (B, 3 queries [CLS,A,B], P pathways)
        return out

    def forward_mono(self, feats, x, t):
        z, _ = self._fuse(feats, [(x, t)])
        return self.head_mono(z).squeeze(-1)


class RawMLP(nn.Module):
    """Baseline: concatenated raw gene-level omics (no pathway mapping) + drug vectors -> MLP."""
    key = "raw"

    def __init__(self, n_raw_genes, drug_dim, mods=(1, 1, 1), d=256, dropout=0.2):
        super().__init__()
        G = n_raw_genes
        keep = torch.cat([torch.full((G,), float(m)) for m in mods])
        self.register_buffer("keep", keep)
        self.drug = nn.Sequential(nn.Linear(drug_dim, d), nn.GELU())
        self.cell = nn.Sequential(nn.Linear(3 * G, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, d), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(3 * d, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, d), nn.GELU())
        self.mono_trunk = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.head_bliss, self.head_hsa = nn.Linear(d, 1), nn.Linear(d, 1)
        self.head_cls, self.head_mono = nn.Linear(d, 1), nn.Linear(d, 1)

    def forward(self, b, return_attn=False):
        c = self.cell(b["raw"] * self.keep)
        z = self.trunk(torch.cat([c, self.drug(b["xa"]) + self.drug(b["xb"]),
                                  (self.drug(b["xa"]) - self.drug(b["xb"])).abs()], -1))
        return {"bliss": self.head_bliss(z).squeeze(-1), "hsa": self.head_hsa(z).squeeze(-1),
                "logit": self.head_cls(z).squeeze(-1)}

    def forward_mono(self, raw, x, t):
        z = self.mono_trunk(torch.cat([self.cell(raw * self.keep), self.drug(x)], -1))
        return self.head_mono(z).squeeze(-1)
