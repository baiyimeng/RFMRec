import os
import copy
import numpy as np
import torch
from torch import optim

from metrics import hrs_and_ndcgs_k
from model import SparseDenseModel


def train_and_evaluate(args, train_loader, val_loader, test_loader, logger):
    """Train and evaluate the model with early stopping"""
    device = args.device

    # Initialize model and optimizer
    model = SparseDenseModel(args).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Track best metrics
    best_metrics = {
        f"{metric}": 0
        for metric in ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"]
    }
    best_epoch = -1
    bad_count = 0
    best_model = None

    # Create save directory and output file path
    save_dir = os.path.join("saved", args.model, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(
        save_dir, str(args.start_time) + args.description + ".pth"
    )

    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        fm_losses = []
        seq_ce_losses = []

        for train_batch in train_loader:
            input_ids, target_ids = [x.to(device) for x in train_batch]
            optimizer.zero_grad()

            fm_loss, seq_ce_loss = model(input_ids, target_ids, train_flag=True)

            fm_losses.append(fm_loss.item())
            seq_ce_losses.append(seq_ce_loss.item())

            loss = fm_loss + seq_ce_loss

            loss.backward()
            optimizer.step()

        # Log training losses
        avg_fm_loss = sum(fm_losses) / len(fm_losses)
        avg_seq_ce_loss = sum(seq_ce_losses) / len(seq_ce_losses)
        logger.info(
            f"Epoch {epoch}: FM loss {avg_fm_loss:.4f}, Seq CE loss {avg_seq_ce_loss:.4f}",
        )
        lr_scheduler.step()

        # Validation phase
        if epoch % args.eval_interval == 0:
            val_metrics, val_init_metrics = evaluate(
                model, val_loader, device, [5, 10, 20]
            )
            logger.info(f"Validation metrics at epoch {epoch}: {val_metrics}")
            logger.info(f"Validation init metrics at epoch {epoch}: {val_init_metrics}")

            refer_metrics = val_metrics
            # refer_metrics = val_init_metrics

            # Calculate mean of all metrics to determine overall improvement
            current_mean_score = np.mean(list(refer_metrics.values()))
            best_mean_score = np.mean(list(best_metrics.values()))

            # Update best metrics if overall performance improves
            improved = current_mean_score > best_mean_score

            if improved:
                bad_count = 0
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                # Update all best metrics
                for metric in refer_metrics:
                    best_metrics[f"{metric}"] = refer_metrics[metric]

                logger.info(f"New best metrics: {best_metrics}")
                logger.info(
                    f"Mean score improved from {best_mean_score:.4f} to {current_mean_score:.4f}"
                )
            else:
                bad_count += 1
                logger.info(
                    f"No improvement. Patience count: {bad_count}/{args.patience}"
                )
                if bad_count >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            test_metrics, test_init_metrics = evaluate(
                model, test_loader, device, [5, 10, 20]
            )
            logger.info(f"Test metrics at epoch {epoch}: {test_metrics}")
            logger.info(f"Test init metrics at epoch {epoch}: {test_init_metrics}")

    # torch.save(best_model.state_dict(), str(output_path))
    logger.info(f"Best model saved at {output_path}, from epoch {best_epoch}")

    model = best_model
    val_metrics, val_init_metrics = evaluate(model, val_loader, device, [5, 10, 20])
    logger.info(f"Best validation metrics: {val_metrics}")
    logger.info(f"Best validation init metrics: {val_init_metrics}")

    test_metrics, test_init_metrics = evaluate(model, test_loader, device, [5, 10, 20])
    logger.info(f"Best test metrics: {test_metrics}")
    logger.info(f"Best test init metrics: {test_init_metrics}")


@torch.no_grad()
def evaluate(model, loader, device, metric_ks):
    """Evaluate the model on the given dataset"""
    model.eval()
    pred_metrics_dict = {
        f"{metric}@{k}": [] for k in metric_ks for metric in ["HR", "NDCG"]
    }
    seq_metrics_dict = {
        f"{metric}@{k}": [] for k in metric_ks for metric in ["HR", "NDCG"]
    }

    for batch in loader:
        input_ids, target_ids = [x.to(device) for x in batch]
        pred_scores, seq_scores = model(input_ids, None, train_flag=False)

        pred_metrics = hrs_and_ndcgs_k(pred_scores, target_ids[:, -1:], metric_ks)
        seq_metrics = hrs_and_ndcgs_k(seq_scores, target_ids[:, -1:], metric_ks)

        for k, v in pred_metrics.items():
            pred_metrics_dict[k].extend(list(v))
        for k, v in seq_metrics.items():
            seq_metrics_dict[k].extend(list(v))

    pred_mean_metrics = {
        k: round(np.mean(v) * 100, 4) for k, v in pred_metrics_dict.items()
    }
    seq_mean_metrics = {
        k: round(np.mean(v) * 100, 4) for k, v in seq_metrics_dict.items()
    }
    return pred_mean_metrics, seq_mean_metrics
