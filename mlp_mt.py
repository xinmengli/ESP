import torch
import torch.nn.functional as F

num_fp_size = 4096 # including the extra mask tokens
num_instrument_type = 7 # including the extra mask tokens
num_lda_size = 100

class MLP_MT(torch.nn.Module):
    def __init__(self, emb_dim, output_dim, drop_ratio=0, disable_mt_lda=False, correlation_mat_rank=0, correlation_type=5,
                 mt_lda_weight=0.01, correlation_mix_residual_weight=0.3):

        super(MLP_MT, self).__init__()
        self.x_embedding1 = torch.nn.Sequential(
            torch.nn.Linear(num_fp_size, emb_dim),
            torch.nn.ReLU()
        )

        self.x_embedding2 = torch.nn.Sequential(
            torch.nn.Linear(num_instrument_type, emb_dim),
            torch.nn.ReLU()
        )

        self.pred_mlp = torch.nn.Sequential(
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim, output_dim),
        )

        self.pred_mlp_reverse = torch.nn.Sequential(
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim, output_dim),
        )

        self.gate = torch.nn.Sequential(
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim, output_dim),
            torch.nn.Sigmoid(),
        )

        self.disable_mt_lda = disable_mt_lda
        if not self.disable_mt_lda:
            self.graph_pred_mt_lda = torch.nn.Sequential(torch.nn.Linear(emb_dim, num_lda_size), torch.nn.Softmax(dim=-1))
            self.mt_lda_weight = mt_lda_weight
        self.num_tasks = output_dim
        self.correlation_mat_rank = correlation_mat_rank
        if self.correlation_mat_rank > 0:
            self.correlation_mat = torch.nn.Parameter(torch.randn([correlation_type, self.num_tasks, self.correlation_mat_rank]), requires_grad=True)
            self.correlation_belong = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, correlation_type),
                torch.nn.Softmax(dim=-1)
            )
            self.correlation_type = correlation_type
            self.correlation_mix_residual_weight = correlation_mix_residual_weight

    def forward(self, *argv, return_logits=False):
        _, _, _, _, instrument, fp, shift = argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]

        x1 = self.x_embedding1(fp)
        x2 = self.x_embedding2(instrument)
        x = x1 + x2

        pred_logit = x

        pred_val = self.pred_mlp(x)

        loss = 0.0
        if self.training:
            if not self.disable_mt_lda:
                if len(argv) >= 8:
                    lda_feature = argv[7]
                    pred_mt_lda = self.graph_pred_mt_lda(pred_logit)
                    loss_mt_lda = -(lda_feature * torch.log(pred_mt_lda + 1e-9)).sum(-1).mean()
                    loss = loss + self.mt_lda_weight * loss_mt_lda

        pred_val_reverse = torch.flip(self.pred_mlp_reverse(x), dims=[1])
        for i in range(len(shift)):
            pred_val_reverse[i, :] = pred_val_reverse[i, :].roll(shift[i].item())
            pred_val_reverse[i, shift[i]:] = 0
        gate = self.gate(x)
        pred_val = gate * pred_val + (1 - gate) * pred_val_reverse

        pred_val = F.softplus(pred_val)

        if self.correlation_mat_rank > 0:
            y_belong = self.correlation_belong(pred_logit).unsqueeze(-1)
            y = pred_val.reshape([1, -1, self.num_tasks])
            y = y @ self.correlation_mat @ self.correlation_mat.transpose(-1, -2)
            y = y.transpose(0, 1)
            y = (y * y_belong).sum(-2)
            y = F.softplus(y)
            pred_val = (1.0 - self.correlation_mix_residual_weight) * y + self.correlation_mix_residual_weight * pred_val

        if return_logits:
            return pred_val, loss, pred_logit
        else:
            return pred_val, loss


