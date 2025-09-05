import torch


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, : ks[i]].sum(dim=-1).cpu().numpy() for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k - 1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg / max_dcg).cpu().numpy())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit / log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(
        labels.clone().detach().to("cpu"), scores.clone().detach().to("cpu"), ks
    )
    hr = cal_hr(
        labels.clone().detach().to("cpu"), scores.clone().detach().to("cpu"), ks
    )
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics["HR@%d" % k] = hr_temp
        metrics["NDCG@%d" % k] = ndcg_temp
    return metrics
