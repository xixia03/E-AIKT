import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
class Trainer:
    def __init__(self, model, train_dataloader, optimizer, evaluator, device, args):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.alpha = args.alpha
        self.num_epochs = args.num_epochs
        self.patience = args.patience
        self.model_save_dir = args.model_save_dir
        self.T = args.temperature
        self.device = device
        self.disable_clusters = args.disable_clusters
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # Monitor loss for reduction
            factor=0.5,  # Reduce learning rate by 50%
            patience=3,  # Wait for 3 epochs without improvement
            verbose=True
        )

    def train(self):
        best_micro_f1 = 0
        best_loss = float('inf')
        cur_patience = 0
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")

            epoch_loss = self.train_epoch()
            print(f"Total Loss: {epoch_loss:.4f}")

            avg_acc, total_acc, avg_f1, micro_f1 = self.evaluator.evaluate(self.model)
            print(f"Eval Micro F1: {micro_f1:.4f}")
            print(f"average acc: {avg_acc:.4f},total acc: {total_acc:.4f}")
            self.scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                cur_patience = 0
                print("Saving model")
                self.save_model(epoch)
            else:
                cur_patience += 1
            if cur_patience >= self.patience:
                print("Early stopping triggered. Training complete.")
                break
        print("Training complete.")

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_dataloader, desc='Training', leave=True)

        for i, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            batch = self._batch_to_device(batch)
            loss = self.process_batch(batch)
            loss.backward() 

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

            wandb.log({'step_loss': loss.item()})

        return total_loss/(i+1)

    def process_batch(self, batch):
        question_embeddings = self.model(batch['question_ids'], batch['question_mask']).last_hidden_state[:, 0, :]
        desc_embeddings = self.model(batch['desc_ids'], batch['desc_mask']).last_hidden_state[:, 0, :]
        kc_embeddings = self.model(batch['kc_ids'], batch['kc_mask']).last_hidden_state[:, 0, :]

        question_kc_similarity =  F.cosine_similarity(question_embeddings.unsqueeze(1), kc_embeddings.unsqueeze(0), dim=2)
        question_desc_similarity = F.cosine_similarity(question_embeddings.unsqueeze(1), desc_embeddings.unsqueeze(0), dim=2)
        question_kc_score = torch.exp(question_kc_similarity / self.T)
        question_desc_score = torch.exp(question_desc_similarity / self.T)

        loss = self.contrastive_loss(question_kc_score, batch['kc_quest_pairs'], batch['cluster_kc_quest_pairs']) + \
            self.alpha * self.contrastive_loss(question_desc_score, batch['desc_quest_pairs'], batch['cluster_desc_quest_pairs'])

        return loss

    def contrastive_loss(self, score_matrix, direct_pair_indices, clustered_pair_indices):
        pos_mask = torch.zeros_like(score_matrix)
        neg_mask = torch.ones_like(score_matrix)

        pos_mask[direct_pair_indices[:, 1], direct_pair_indices[:, 0]] = 1
        neg_mask[direct_pair_indices[:, 1], direct_pair_indices[:, 0]] = 0

        if not self.disable_clusters:
            neg_mask[clustered_pair_indices[:, 1], clustered_pair_indices[:, 0]] = 0

        pos_score = score_matrix * pos_mask
        neg_score = score_matrix * neg_mask

        pos_score_sum = pos_score.sum(dim=-1)
        neg_score_sum = neg_score.sum(dim=-1)
        scores_sum = pos_score_sum + neg_score_sum

        scores_sum = scores_sum.clamp(min=1e-8)
        pos_score_sum = pos_score_sum.clamp(min=1e-8)
        pos_score = pos_score + (1 - pos_mask)
        element_loss = -1 * torch.log(pos_score / scores_sum.unsqueeze(-1))

        element_loss = element_loss * pos_mask
        row_loss = element_loss.sum(dim=-1) / pos_mask.sum(dim=-1).clamp(min=1e-8)
        valid_rows = pos_mask.sum(dim=-1) > 0
        row_loss = row_loss[valid_rows]
        cl_loss = row_loss.mean()

        return cl_loss
    def _batch_to_device(self, batch):
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device)
        return batch

    def save_model(self, epoch):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        model_dir = self.model_save_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_save_path = os.path.join(model_dir, f'bert_finetuned_epoch_{epoch + 1}.bin')
        torch.save(model_to_save.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
