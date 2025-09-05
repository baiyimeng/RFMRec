import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from rfmrec import RiemannFlowMatchingRec


def create_dense_model(args):
    if args.model == "rfmrec":
        return RiemannFlowMatchingRec(args)
    else:
        raise NotImplementedError(f"model {args.model} not implemented")


class SparseDenseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb_dim = args.hidden_size
        self.args = args
        self.item_num = args.item_num
        self.sparse_embedding = self.create_item_embedding(
            pretrained=args.pretrained, freeze_emb=args.freeze_emb
        )
        self.embed_dropout = nn.Dropout(args.emb_dropout)

        if args.pre_norm:
            self.first_norm = nn.Identity()
        else:
            self.first_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.dense_model = create_dense_model(args)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                if not self.args.pretrained:
                    nn.init.xavier_normal_(module.weight)
                    if module.padding_idx is not None:
                        with torch.no_grad():
                            module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def load_pretrained_emb_weight(self):
        path = os.path.join("saved", "pretrain", self.args.dataset, "pretrain.pth")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        pretrained_emb_weight = saved["sparse_embedding.weight"]
        return pretrained_emb_weight

    def create_item_embedding(self, pretrained=False, freeze_emb=False):
        if pretrained:
            embedding = nn.Embedding.from_pretrained(
                self.load_pretrained_emb_weight(),
                padding_idx=0,
                freeze=freeze_emb,
            )
        else:
            embedding = nn.Embedding(self.item_num + 1, self.emb_dim, padding_idx=0)
        return embedding

    def calculate_cross_entropy_loss(self, seq_embs, target_ids, temperature=0.07):
        valid_index = target_ids > 0
        seq_embs = seq_embs[valid_index]
        target_ids = target_ids[valid_index]

        seq_embs = F.normalize(seq_embs, p=2, dim=-1)
        emb_weight = F.normalize(self.sparse_embedding.weight, p=2, dim=-1)

        scores = torch.matmul(seq_embs, emb_weight.t()) / temperature
        ce_loss = F.cross_entropy(scores, target_ids)
        return ce_loss

    def forward(self, input_ids, target_ids=None, train_flag=True):
        input_embs = self.sparse_embedding(input_ids)
        input_embs = input_embs * (self.emb_dim**0.5)

        input_embs = self.embed_dropout(input_embs)
        input_embs = self.first_norm(input_embs)

        input_mask = (input_ids > 0).float()

        if train_flag:
            target_embs = self.sparse_embedding(target_ids)

            target_mask = (target_ids > 0).float()

            fm_loss, seq_embs = self.dense_model(
                input_embs, target_embs, input_mask, target_mask
            )

            seq_ce_loss = self.calculate_cross_entropy_loss(seq_embs, target_ids, 0.07)
            return fm_loss, seq_ce_loss

        else:
            pred_emb, seq_emb = self.dense_model.inference(input_embs, input_mask)

            pred_emb = F.normalize(pred_emb, p=2, dim=-1)
            seq_emb = F.normalize(seq_emb, p=2, dim=-1)
            emb_weight = F.normalize(self.sparse_embedding.weight, p=2, dim=-1)

            pred_scores = torch.matmul(pred_emb, emb_weight.t())
            seq_scores = torch.matmul(seq_emb, emb_weight.t())
            return pred_scores, seq_scores
