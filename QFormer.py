# CPW_model_cls_qformer.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class InteractionDataset(Dataset):
    def __init__(self, fps_path, labels_path, gene_ids):
        self.fps_path = fps_path
        self.labels_path = labels_path
        self.gene_ids = gene_ids

        self.labels = np.load(labels_path, mmap_mode="c")
        self.fps = None

        assert len(self.labels) == len(self.gene_ids), (
            f"标签数量 {len(self.labels)} 与 gene_ids 数量 {len(self.gene_ids)} 不一致"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.fps is None:
            self.fps = np.load(self.fps_path, mmap_mode="c")

        fp = torch.from_numpy(self.fps[idx]).float()
        gene_id = torch.tensor(self.gene_ids[idx], dtype=torch.long)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        gene_count = torch.tensor(
            [np.log1p(len(self.gene_ids[idx]))],
            dtype=torch.float32
        )

        return fp, gene_id, gene_count, label


def fast_collate_fn(batch):
    fps, gene_ids_list, gene_counts, labels = zip(*batch)

    padded_gene_ids = pad_sequence(
        gene_ids_list,
        batch_first=True,
        padding_value=0
    )

    lengths = torch.tensor([len(g) for g in gene_ids_list])
    max_len = padded_gene_ids.size(1)

    key_padding_mask = torch.arange(max_len)[None, :] >= lengths[:, None]

    return (
        torch.stack(fps),
        padded_gene_ids,
        key_padding_mask,
        torch.stack(gene_counts),
        torch.stack(labels)
    )


class QFormerBlock(nn.Module):
    """
    A lightweight Q-Former block.

    Structure:
        1. self-attention among learnable query tokens
        2. cross-attention: query tokens attend to pathway gene embeddings
        3. feed-forward network

    This differs from full CrossAttentionClassifier:
        - CrossAttentionClassifier uses molecular embedding as the query.
        - QFormerClassifier uses a small set of learnable pathway queries
          to compress gene-level pathway embeddings into latent pathway tokens.
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout_rate=0.1,
        ffn_mult=4,
    ):
        super().__init__()

        self.query_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.query_gene_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.norm_q1 = nn.LayerNorm(embed_dim)
        self.norm_q2 = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * ffn_mult, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, query_tokens, gene_tokens, key_padding_mask):
        """
        query_tokens: [B, Q, D]
        gene_tokens:  [B, L, D]
        key_padding_mask: [B, L], True means padding.

        return:
            query_tokens: [B, Q, D]
            cross_attn_weights: [B, Q, L]
        """

        # 1. self-attention among query tokens
        q_norm = self.norm_q1(query_tokens)

        self_attn_out, _ = self.query_self_attn(
            query=q_norm,
            key=q_norm,
            value=q_norm,
            need_weights=False,
        )

        query_tokens = query_tokens + self.dropout(self_attn_out)

        # 2. query-to-gene cross-attention
        q_norm = self.norm_q2(query_tokens)

        cross_out, cross_attn_weights = self.query_gene_cross_attn(
            query=q_norm,
            key=gene_tokens,
            value=gene_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        query_tokens = query_tokens + self.dropout(cross_out)

        # 3. FFN
        q_norm = self.norm_ffn(query_tokens)
        query_tokens = query_tokens + self.ffn(q_norm)

        return query_tokens, cross_attn_weights


class QFormerClassifier(nn.Module):
    """
    Q-Former pathway encoder for compound-pathway classification.

    Core idea:
        pathway gene embeddings
            -> K learnable query tokens
            -> K latent pathway tokens
            -> pooled pathway representation

        molecule Morgan FP
            -> molecular representation

        [molecule representation, Q-Former pathway representation, gene_count]
            -> classifier

    This is an intermediate-granularity model between:
        Mean pooling DNN
        and full molecular-query CrossAttention.
    """

    def __init__(
        self,
        pretrained_scgpt_matrix,
        embed_dim=512,
        num_heads=8,
        num_queries=8,
        num_qformer_layers=2,
        dropout_rate=0.1,
        num_classes=2,
        freeze_gene_embedding=True,
        use_mean_residual=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.use_mean_residual = use_mean_residual

        # ========= Gene embedding =========
        self.gene_embedding = nn.Embedding.from_pretrained(
            pretrained_scgpt_matrix,
            freeze=freeze_gene_embedding,
            padding_idx=0,
        )

        self.gene_proj = nn.Sequential(
            nn.Linear(pretrained_scgpt_matrix.shape[1], embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # ========= Molecular encoder =========
        self.mol_proj = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ========= Learnable Q-Former queries =========
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) * 0.02
        )

        self.qformer_blocks = nn.ModuleList([
            QFormerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_qformer_layers)
        ])

        # Pool K query tokens into one pathway vector
        self.query_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        if use_mean_residual:
            self.mean_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
                nn.Sigmoid(),
            )

        # ========= Classifier =========
        # molecule embedding + Q-Former pathway embedding + gene_count
        classifier_input_dim = embed_dim * 2 + 1

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, num_classes),
        )

    def masked_mean_pooling(self, gene_tokens, key_padding_mask):
        """
        gene_tokens: [B, L, D]
        key_padding_mask: [B, L], True means padding.
        """

        valid_mask = (~key_padding_mask).float().unsqueeze(-1)  # [B, L, 1]

        summed = torch.sum(gene_tokens * valid_mask, dim=1)
        counts = valid_mask.sum(dim=1).clamp(min=1.0)

        return summed / counts

    def forward(self, mol_fp, gene_ids, key_padding_mask, gene_counts):
        """
        mol_fp: [B, 2048]
        gene_ids: [B, L]
        key_padding_mask: [B, L], True means padding.
        gene_counts: [B, 1]

        return:
            logits: [B, 2]
            attn_weights: [B, Q, L]
        """

        batch_size = mol_fp.size(0)

        # Molecular representation
        mol_embed = self.mol_proj(mol_fp)  # [B, D]

        # Gene-level pathway embeddings
        gene_vectors = self.gene_embedding(gene_ids)  # [B, L, D_scgpt]
        gene_tokens = self.gene_proj(gene_vectors)    # [B, L, D]

        # Expand learnable queries
        query_tokens = self.query_tokens.expand(
            batch_size,
            -1,
            -1,
        )  # [B, Q, D]

        last_attn_weights = None

        for block in self.qformer_blocks:
            query_tokens, last_attn_weights = block(
                query_tokens=query_tokens,
                gene_tokens=gene_tokens,
                key_padding_mask=key_padding_mask,
            )

        # Pool query tokens into pathway representation
        # [B, Q, D] -> [B, D]
        qformer_pathway = self.query_pool(query_tokens).mean(dim=1)

        # Optional residual fusion with mean pathway embedding.
        # This keeps the strong pathway-level prior and lets Q-Former act as refinement.
        if self.use_mean_residual:
            mean_pathway = self.masked_mean_pooling(
                gene_tokens,
                key_padding_mask,
            )

            gate = self.mean_gate(
                torch.cat([mean_pathway, qformer_pathway], dim=-1)
            )  # [B, 1]

            pathway_embed = (1.0 - gate) * mean_pathway + gate * qformer_pathway
        else:
            pathway_embed = qformer_pathway

        fused_features = torch.cat(
            [mol_embed, pathway_embed, gene_counts],
            dim=-1,
        )

        logits = self.classifier(fused_features)

        return logits, last_attn_weights