"""Adapt VAT loss to binary output"""

import contextlib
import torch
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def _kl_div_with_logit(logits_p, logits_m):
    kl_p = torch.sigmoid(logits_p)*(
            F.logsigmoid(logits_p)-F.logsigmoid(logits_m))
    kl_q = torch.sigmoid(-logits_p)*(
            F.logsigmoid(-logits_p)-F.logsigmoid(-logits_m))
    return torch.mean(kl_p)+torch.mean(kl_q)


def entropy_with_logit(logits):
    ent_p = F.logsigmoid(logits)*torch.sigmoid(logits)
    ent_q = F.logsigmoid(-logits)*torch.sigmoid(-logits)
    return -torch.mean(ent_p)-torch.mean(ent_q)


def entropy_regularization(model, x):
    with _disable_tracking_bn_stats(model):
        fx = model(x)
        ent_loss = entropy_with_logit(fx)
    return ent_loss


# no idea if it works or not
class VAT(object):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def __call__(self, model, x):
        # # shape [batch_size, 1]
        # logits_p = model(x, update_batch_stats=False).detach()

        # # prepare random unit tensor
        # d = torch.randn(x.shape).to(x.device)
        # d = _l2_normalize(d)

        # # calc adversarial direction
        # for _ in range(self.ip):
        #     # x_hat = x + self.xi*d
        #     d.requires_grad_()
        #     logits_m = model(x + self.xi*d, update_batch_stats=False)
        #     adv_distance = _kl_div_with_logit(logits_p, logits_m)
        #     adv_distance.backward()
        #     # d = _l2_normalize(d.grad)
        #     d = _l2_normalize(d.grad)
        #     model.zero_grad()

        # # calc LDS
        # r_adv = d * self.eps
        # logits_m = model(x + r_adv.detach(), update_batch_stats=False)
        # lds = _kl_div_with_logit(logits_p, logits_m)
        with _disable_tracking_bn_stats(model):
            # shape [batch_size, 1]
            logits_p = model(x).detach()

            # prepare random unit tensor
            d = torch.randn(x.shape).to(x.device)
            d = _l2_normalize(d)

            # calc adversarial direction
            for _ in range(self.ip):
                x_hat = x + self.xi*d
                x_hat.requires_grad = True
                logits_m = model(x_hat)
                adv_distance = _kl_div_with_logit(logits_p, logits_m)
                adv_distance.backward()
                # d = _l2_normalize(d.grad)
                d = _l2_normalize(x_hat.grad)

            # calc LDS
            r_adv = d * self.eps
            logits_m = model(x + r_adv)
            lds = _kl_div_with_logit(logits_p, logits_m)

        return lds
