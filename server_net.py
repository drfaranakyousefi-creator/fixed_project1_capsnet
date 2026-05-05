import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------
# Squash activation — قلب Capsule Network
# vector رو نرمالایز میکنه بین 0 و 1 بدون اینکه جهتش رو عوض کنه
# ---------------------------------------------------------------
def squash(x, dim=-1):
    norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm = norm_sq.sqrt()
    scale = norm_sq / (1.0 + norm_sq)
    return scale * x / (norm + 1e-8)


# ---------------------------------------------------------------
# Primary Capsule Layer
# input vector رو به N تا capsule تبدیل میکنه
# ---------------------------------------------------------------
class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_features, num_capsules, capsule_dim):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        # هر capsule یه linear layer داره
        self.capsules = nn.Linear(in_features, num_capsules * capsule_dim)

    def forward(self, x):
        # x: (batch, in_features)
        out = self.capsules(x)                                    # (batch, num_capsules * capsule_dim)
        out = out.view(x.size(0), self.num_capsules, self.capsule_dim)  # (batch, num_capsules, capsule_dim)
        return squash(out)                                        # (batch, num_capsules, capsule_dim)


# ---------------------------------------------------------------
# Dynamic Routing بین capsule ها
# ---------------------------------------------------------------
class RoutingLayer(nn.Module):
    def __init__(self, num_input_caps, input_dim, num_output_caps, output_dim, num_routing=3):
        super().__init__()
        self.num_routing = num_routing
        self.num_output_caps = num_output_caps
        self.output_dim = output_dim

        # W ماتریس transformation برای هر جفت (input_cap, output_cap)
        self.W = nn.Parameter(
            torch.randn(1, num_input_caps, num_output_caps, output_dim, input_dim) * 0.01
        )

    def forward(self, x):
        # x: (batch, num_input_caps, input_dim)
        batch = x.size(0)

        # x رو expand میکنیم برای ضرب ماتریسی
        x_ = x.unsqueeze(2).unsqueeze(4)   # (batch, num_input_caps, 1, input_dim, 1)
        W_ = self.W.expand(batch, -1, -1, -1, -1)  # (batch, num_input_caps, num_output_caps, output_dim, input_dim)

        # u_hat: prediction هر input capsule برای هر output capsule
        u_hat = torch.matmul(W_, x_).squeeze(-1)  # (batch, num_input_caps, num_output_caps, output_dim)

        # routing logits — اول صفر
        b = torch.zeros(batch, x.size(1), self.num_output_caps, device=x.device)

        for _ in range(self.num_routing):
            c = F.softmax(b, dim=2)                          # (batch, num_input_caps, num_output_caps)
            c_ = c.unsqueeze(-1)                             # (batch, num_input_caps, num_output_caps, 1)
            s = (c_ * u_hat).sum(dim=1)                      # (batch, num_output_caps, output_dim)
            v = squash(s)                                    # (batch, num_output_caps, output_dim)
            # routing logits رو آپدیت کن
            b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)     # (batch, num_input_caps, num_output_caps)

        return v  # (batch, num_output_caps, output_dim)


# ---------------------------------------------------------------
# Prediction Network با Capsule
# جایگزین MLP ساده قبلی
# ---------------------------------------------------------------
class prediction_net(nn.Module):
    def __init__(self, w, n_features_input, lr=0.007,
                 num_primary_caps=16, primary_dim=4,
                 num_output_caps=4, output_dim=8,
                 num_routing=3):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_size = w + n_features_input  # همون (batch, w+N) که از client میاد

        # لایه اول: فشرده‌سازی اولیه
        self.fc_in = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        ).to(self.device)

        # Primary Capsules
        self.primary_caps = PrimaryCapsLayer(
            in_features=64,
            num_capsules=num_primary_caps,
            capsule_dim=primary_dim
        ).to(self.device)

        # Routing به Output Capsules
        self.routing = RoutingLayer(
            num_input_caps=num_primary_caps,
            input_dim=primary_dim,
            num_output_caps=num_output_caps,
            output_dim=output_dim,
            num_routing=num_routing
        ).to(self.device)

        # لایه آخر: از capsule vector به یه عدد
        self.fc_out = nn.Linear(num_output_caps * output_dim, 1).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, combined_embedded, label=None, status='test'):
        combined_embedded = torch.tensor(
            combined_embedded, dtype=torch.float, device=self.device
        )
        combined_embedded.requires_grad_(True)

        # forward pass
        x = self.fc_in(combined_embedded)           # (batch, 64)
        x = self.primary_caps(x)                    # (batch, num_primary_caps, primary_dim)
        x = self.routing(x)                         # (batch, num_output_caps, output_dim)
        x = x.view(x.size(0), -1)                  # (batch, num_output_caps * output_dim)
        output = self.fc_out(x)                     # (batch, 1)

        if status == 'train':
            label = torch.tensor(label, dtype=torch.float, device=self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(output, label)
            loss.backward()
            input_grad = combined_embedded.grad.detach().cpu().tolist()
            self.optimizer.step()
            return {'grad': input_grad}

        else:  # test
            result = output.detach().cpu().tolist()
            return {'prediction': result}
