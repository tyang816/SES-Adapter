import argparse
import warnings
import torch
import os
import sys
sys.path.append(os.getcwd())
import wandb
import random
import json
import re
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import logging
from torchmetrics.classification import Accuracy
from torchmetrics.regression import SpearmanCorrCoef
from accelerate import Accelerator
from accelerate.utils import set_seed
from time import strftime, localtime
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer, T5Tokenizer, T5EncoderModel, AutoTokenizer
from src.utils.data_utils import BatchSampler
from src.models.adapter import AdapterModel

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

DATASET_TO_NUM_LABELS = {
    "DeepLocBinary": 2, "DeepLocMulti": 10, 
    "DeepSol": 2, "DeepSoluE": 2,
    "MetalIonBinding": 2, "Thermostability": 1
}
DATASET_TO_TASK = {
    "DeepLocBinary": "single_label_classification", 
    "DeepLocMulti": "single_label_classification",
    "DeepSol": "single_label_classification", 
    "DeepSoluE": "single_label_classification",
    "MetalIonBinding": "single_label_classification", 
    "Thermostability": "regression",
}
# valid and test metrics
DATASET_TO_METRICS = {
    "DeepLocBinary": ("accuracy", Accuracy(task="multiclass", num_classes=2)),
    "DeepLocMulti": ("accuracy", Accuracy(task="multiclass", num_classes=10)),
    "DeepSol": ("accuracy", Accuracy(task="multiclass", num_classes=2)),
    "DeepSoluE": ("accuracy", Accuracy(task="multiclass", num_classes=2)),
    "MetalIonBinding": ("accuracy", Accuracy(task="multiclass", num_classes=2)),
    "Thermostability": ("spearman_corr", SpearmanCorrCoef())
}
DATASET_TO_MONITOR = {
    "DeepLocBinary": "accuracy",
    "DeepLocMulti": "accuracy",
    "DeepSol": "accuracy",
    "DeepSoluE": "accuracy",
    "MetalIonBinding": "accuracy",
    "Thermostability": "spearman_corr"
}
DATASET_TO_NORMALIZE = {
    "DeepLocBinary": None,
    "DeepLocMulti": None,
    "DeepSol": None,
    "DeepSoluE": None,
    "MetalIonBinding": None,
    "Thermostability": "min_max"
}

def min_max_normalize_dataset(train_dataset, val_dataset, test_dataset):
    labels = [e["label"] for e in train_dataset]
    min_label, max_label = min(labels), max(labels)
    for e in train_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    for e in val_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    for e in test_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    return train_dataset, val_dataset, test_dataset


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=1, reduction='mean', device="cuda"):
        super(MultiClassFocalLossWithAlpha, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes, dtype=torch.float32)
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


def train(args, model, plm_model, accelerator, metrics, train_loader, val_loader, test_loader, 
          loss_fn, optimizer, device):
    best_val_loss, best_val_metric_score = float("inf"), -float("inf")
    val_loss_list, val_metric_list = [], []
    metric_name, metric = metrics
    metric = metric.to(device)
    path = os.path.join(args.ckpt_dir, args.model_name)
    global_steps = 0
    for epoch in range(args.max_train_epochs):
        print(f"---------- Epoch {epoch} ----------")
        # train
        model.train()
        epoch_train_loss = 0
        epoch_iterator = tqdm(train_loader)
        for batch in epoch_iterator:
            with accelerator.accumulate(model):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                label = batch["label"]
                logits = model(plm_model, batch)
                loss = loss_fn(logits, label)
                epoch_train_loss += loss.item() * len(label)                
                global_steps += 1
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                epoch_iterator.set_postfix(train_loss=loss.item())
                if args.wandb:
                    wandb.log({"train/loss": loss.item(), "train/epoch": epoch}, step=global_steps)
                    
        train_loss = epoch_train_loss / len(train_loader.dataset)
        print(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f}')
        
        # eval every epoch
        model.eval()
        with torch.no_grad():
            val_loss, val_metric_score = eval_loop(model, plm_model, metric, val_loader, loss_fn, device)
            val_metric_list.append(val_metric_score)
            val_loss_list.append(val_loss)
            if args.wandb:
                wandb.log({"valid/loss": val_loss, f"valid/{metric_name}": val_metric_score, "valid/epoch": epoch})
        print(f'EPOCH {epoch} VAL loss: {val_loss:.4f} {metric_name}: {val_metric_score:.4f}')
    
        # early stopping
        if args.monitor == "loss":
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), path)
                print(f'>>> BEST at epcoh {epoch}, loss: {best_val_loss:.4f}, {metric_name}: {val_metric_score:.4f}')
                print(f'>>> Save model to {path}')
            
            if len(val_loss_list) - val_loss_list.index(min(val_loss_list)) > args.patience:
                print(f'>>> Early stopping at epoch {epoch}')
                break
        else:
            if val_metric_score > best_val_metric_score:
                best_val_metric_score = val_metric_score
                torch.save(model.state_dict(), path)
                print(f'>>> BEST at epcoh {epoch}, loss: {val_loss:.4f}, {metric_name}: {best_val_metric_score:.4f}')
                print(f'>>> Save model to {path}')
            
            if len(val_metric_list) - val_metric_list.index(max(val_metric_list)) > args.patience:
                print(f'>>> Early stopping at epoch {epoch}')
                break
    
    print(f"TESTING: loading from {path}")
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        test_loss, test_metric_score = eval_loop(model, plm_model, metric, test_loader, loss_fn, device)
        if args.wandb:
            wandb.log({"test/loss": test_loss, f"test/{metric_name}": test_metric_score})
    print(f'TEST loss: {test_loss:.4f} {metric_name}: {test_metric_score:.4f}')


def eval_loop(model, plm_model, metric, dataloader, loss_fn, device=None):
    total_loss = 0
    epoch_iterator = tqdm(dataloader)
    
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch["label"]
        logits = model(plm_model, batch)
        loss = loss_fn(logits, label)
        total_loss += loss.item() * len(label)
        metric_socre = metric(logits.squeeze(), label).item()
        epoch_iterator.set_postfix(eval_loss=loss.item(), eval_metric=metric_socre)
    
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_metric_score = metric.compute()
    metric.reset()
    return epoch_loss, epoch_metric_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--num_labels', type=int, default=None, help='number of labels')
    parser.add_argument('--pooling_method', type=str, default='attention1d', help='pooling method')
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help='pooling dropout')
    
    # dataset
    parser.add_argument('--dataset', type=str, default=None, required=True, help='dataset name')
    parser.add_argument('--pdb_type', type=str, default='ef', help='pdb type')
    parser.add_argument('--train_file', type=str, default=None, help='train file')
    parser.add_argument('--val_file', type=str, default=None, help='val file')
    parser.add_argument('--test_file', type=str, default=None, help='test file')
    
    # train model
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--max_batch_token', type=int, default=10000, help='max number of token per batch')
    parser.add_argument('--max_train_epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--monitor', type=str, default=None, help='monitor metric')
    parser.add_argument('--use_foldseek', action='store_true', help='use foldseek')
    parser.add_argument('--use_ss8', action='store_true', help='use ss8')
    parser.add_argument('--loss_fn', type=str, default='cross_entropy', choices=['cross_entropy', 'focal_loss'], help='loss function')
    
    # save model
    parser.add_argument('--model_name', type=str, default=None, help='model name')
    parser.add_argument('--ckpt_root', default="ckpt", help='root directory to save trained models')
    parser.add_argument('--ckpt_dir', default=None, help='directory to save trained models')
    
    # wandb log
    parser.add_argument('--wandb', action='store_true', help='use wandb to log')
    parser.add_argument('--wandb_project', type=str, default='LocSeek')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    # init wandb
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"Adapter-{args.dataset}"
        if args.model_name is None:
            args.model_name = f"{args.wandb_run_name}.pt"
        
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
        )
    
    # create checkpoint directory
    if args.ckpt_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.ckpt_dir = os.path.join(args.ckpt_root, current_date)
    else:
        args.ckpt_dir = os.path.join(args.ckpt_root, args.ckpt_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # build tokenizer and protein language model
    if "esm" in args.plm_model:
        tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        plm_model = EsmModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "bert" in args.plm_model:
        tokenizer = BertTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = BertModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "prot_t5" in args.plm_model:
        tokenizer = T5Tokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    elif "ankh" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    
    args.vocab_size = plm_model.config.vocab_size
    if args.train_file is None:
        args.train_file = f"dataset/{args.dataset}/{args.pdb_type}_train.json"
    if args.val_file is None:
        args.val_file = f"dataset/{args.dataset}/{args.pdb_type}_val.json"
    if args.test_file is None:
        args.test_file = f"dataset/{args.dataset}/{args.pdb_type}_test.json"
    if args.num_labels is None:
        args.num_labels = DATASET_TO_NUM_LABELS[args.dataset]
    
    # load adapter model
    model = AdapterModel(args)
    model.to(device)

    def param_num(model):
        total = sum([param.numel() for param in model.parameters() if param.requires_grad])
        num_M = total/1e6
        if num_M >= 1000:
            return "Number of parameter: %.2fB" % (num_M/1e3)
        else:
            return "Number of parameter: %.2fM" % (num_M)
    print(param_num(model))
    
    # process dataset
    def load_dataset(file):
        dataset, token_num = [], []
        for l in open(file):
            data = json.loads(l)
            dataset.append(data)
            if args.max_seq_len is not None:
                token_num.append(min(len(data["aa_seq"]), args.max_seq_len))
            else:
                token_num.append(len(data["aa_seq"]))
        return dataset, token_num

    train_dataset, train_token_num = load_dataset(args.train_file)
    val_dataset, val_token_num = load_dataset(args.val_file)
    test_dataset, test_token_num = load_dataset(args.test_file)
    if DATASET_TO_NORMALIZE[args.dataset] == "min_max":
        train_dataset, val_dataset, test_dataset = min_max_normalize_dataset(train_dataset, val_dataset, test_dataset)
    
    print(">>> trainset: ", len(train_dataset))
    print(">>> valset: ", len(val_dataset))
    print(">>> testset: ", len(test_dataset))
    print("---------- Smple 3 data point from trainset ----------")
    
    for i in random.sample(range(len(train_dataset)), 2):
        print(">>> ", train_dataset[i])
    
    def collate_fn(examples):
        aa_seqs, foldseek_seqs, ss8_seqs, labels = [], [], [], []
        for e in examples:
            aa_seq, foldseek_seq, ss8_seq = e["aa_seq"], e["foldseek_seq"], e["ss8_seq"]
            
            if args.max_seq_len is not None:
                aa_seq = e["aa_seq"][:args.max_seq_len]
                foldseek_seq = e["foldseek_seq"][:args.max_seq_len]
                ss8_seq = e["ss8_seq"][:args.max_seq_len]
            if 'prot_bert' in args.plm_model or "prot_t5" in args.plm_model:
                aa_seq = " ".join(list(aa_seq))
                aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
                foldseek_seq = " ".join(list(foldseek_seq))
                ss8_seq = " ".join(list(ss8_seq))
            elif 'ankh' in args.plm_model:
                aa_seq = list(aa_seq)
                foldseek_seq = list(foldseek_seq)
                ss8_seq = list(ss8_seq)
            
            aa_seqs.append(aa_seq)
            foldseek_seqs.append(foldseek_seq)
            ss8_seqs.append(ss8_seq)
            labels.append(e["label"])
        
        if 'ankh' in args.plm_model:
            aa_inputs = tokenizer.batch_encode_plus(aa_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
            foldseek_input_ids = tokenizer.batch_encode_plus(foldseek_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")["input_ids"]
            ss8_input_ids = tokenizer.batch_encode_plus(ss8_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")["input_ids"]
        else:
            aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
            foldseek_input_ids = tokenizer(foldseek_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            ss8_input_ids = tokenizer(ss8_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        aa_input_ids = aa_inputs["input_ids"]
        attention_mask = aa_inputs["attention_mask"]
        
        if 'classification' in DATASET_TO_TASK[args.dataset]:
            labels = torch.as_tensor(labels, dtype=torch.long)
        elif 'regression' in DATASET_TO_TASK[args.dataset]:
            labels = torch.as_tensor(labels, dtype=torch.float)
        
        return {
            "aa_input_ids": aa_input_ids, 
            "attention_mask": attention_mask, 
            "foldseek_input_ids": foldseek_input_ids, 
            "ss8_input_ids": ss8_input_ids,
            "label": labels
            }
        
    # metrics, optimizer, dataloader
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metrics = DATASET_TO_METRICS[args.dataset]
    if args.monitor is None:
        args.monitor = DATASET_TO_MONITOR[args.dataset]
    if DATASET_TO_TASK[args.dataset] == "single_label_classification":
        if args.loss_fn == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss()
        elif args.loss_fn == "focal_loss":
            train_labels = [e["label"] for e in train_dataset]
            alpha = [len(train_labels) / train_labels.count(i) for i in range(args.num_labels)]
            print(">>> alpha: ", alpha)
            loss_fn = MultiClassFocalLossWithAlpha(num_classes=args.num_labels, alpha=alpha, device=device)
    elif DATASET_TO_TASK[args.dataset] == "regression":
        loss_fn = nn.MSELoss()
    
    train_loader = DataLoader(
        train_dataset, num_workers=args.num_workers, collate_fn=collate_fn,
        batch_sampler=BatchSampler(train_token_num, args.max_batch_token)
        )
    val_loader = DataLoader(
        val_dataset, num_workers=args.num_workers, collate_fn=collate_fn,
        batch_sampler=BatchSampler(val_token_num, args.max_batch_token, False)
        )
    test_loader = DataLoader(
        test_dataset, num_workers=args.num_workers, collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, args.max_batch_token, False)
        )
    
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    
    print("---------- Start Training ----------")
    train(args, model, plm_model, accelerator, metrics, train_loader, val_loader, test_loader, loss_fn, optimizer, device)
    
    if args.wandb:
        wandb.finish()