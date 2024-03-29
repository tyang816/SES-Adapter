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
from accelerate import Accelerator
from accelerate.utils import set_seed
from time import strftime, localtime
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer, T5Tokenizer, T5EncoderModel
from src.utils.data_utils import BatchSampler
from src.models.locseek import LocSeekModel

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

dataset_to_label = {
    "MetalIonBinding": 2, "deeploc-1_multi": 10, "deepsol": 2,
    "deeploc-1_binary": 2
}

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=1, reduction='mean', device="cuda"):
        super(MultiClassFocalLossWithAlpha, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes, dtype=torch.float32)
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


def train(args, model, plm_model, accelerator, metrics, train_loader, val_loader, test_loader, 
          loss_fn, optimizer, device):
    best_val_loss, best_val_acc = float("inf"), 0
    val_loss_list, val_acc_list = [], []
    path = os.path.join(args.ckpt_dir, args.model_name)
    global_steps = 0
    for epoch in range(args.max_train_epochs):
        print(f"---------- Epoch {epoch} ----------")
        # train
        model.train()
        epoch_train_loss, epoch_train_acc = 0, 0
        epoch_iterator = tqdm(train_loader)
        for batch in epoch_iterator:
            with accelerator.accumulate(model):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                label = batch["label"]
                logits = model(plm_model, batch)
                loss = loss_fn(logits, label)
                epoch_train_loss += loss.item() * len(label)
                acc = metrics(logits, label).item()
                epoch_train_acc += acc * len(label)
                
                global_steps += 1
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                epoch_iterator.set_postfix(train_loss=loss.item(), train_acc=acc)
                if args.wandb:
                    wandb.log({"train/train_loss": loss.item(), "train/train_acc": acc, "train/epoch": epoch}, step=global_steps)
                
                # eval every n steps
                if global_steps % args.eval_every_n_steps == 0 and args.eval_every_n_steps > 0:
                    model.eval()
                    with torch.no_grad():
                        val_loss, val_acc = eval_loop(model, plm_model, metrics, val_loader, loss_fn, device)
                        val_acc_list.append(val_acc)
                        val_loss_list.append(val_loss)
                        if args.wandb:
                            wandb.log({"valid/val_loss": val_loss, "valid/val_acc": val_acc, "valid/epoch": epoch}, step=global_steps)
                    print(f'>>> Steps {global_steps} VAL loss: {val_loss:.4f} acc: {val_acc:.4f}')
                    model.train()
                    
                    # early stopping
                    if args.monitor == "val_acc":
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            torch.save(model.state_dict(), path)
                            print(f'>>> BEST at steps {global_steps}, loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f}')
                            print(f'>>> Save model to {path}')
                            
                        if len(val_acc_list) - val_acc_list.index(max(val_acc_list)) > args.patience:
                            print(f'>>> Early stopping at steps {global_steps}')
                            break
                    
                    if args.monitor == "val_loss":
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.state_dict(), path)
                            print(f'>>> BEST at steps {global_steps}, loss: {best_val_loss:.4f}, acc: {val_acc:.4f}')
                            print(f'>>> Save model to {path}')
                        
                        if len(val_loss_list) - val_loss_list.index(min(val_loss_list)) > args.patience:
                            print(f'>>> Early stopping at steps {global_steps}')
                            break
                    
        train_loss = epoch_train_loss / len(train_loader.dataset)
        train_acc = epoch_train_acc / len(train_loader.dataset)
        print(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f} acc: {train_acc:.4f}')
        
        # eval every epoch
        if args.eval_every_n_steps < 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_acc = eval_loop(model, plm_model, metrics, val_loader, loss_fn, device)
                val_acc_list.append(val_acc)
                val_loss_list.append(val_loss)
                if args.wandb:
                    wandb.log({"valid/val_loss": val_loss, "valid/val_acc": val_acc, "valid/epoch": epoch})
            print(f'EPOCH {epoch} VAL loss: {val_loss:.4f} acc: {val_acc:.4f}')
        
            # early stopping
            if args.monitor == "val_acc":
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), path)
                    print(f'>>> BEST at epcoh {epoch}, loss: {val_loss:.4f}, acc: {best_val_acc:.4f}')
                    print(f'>>> Save model to {path}')
                
                if len(val_acc_list) - val_acc_list.index(max(val_acc_list)) > args.patience:
                    print(f'>>> Early stopping at epoch {epoch}')
                    break
            
            if args.monitor == "val_loss":
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), path)
                    print(f'>>> BEST at epcoh {epoch}, loss: {best_val_loss:.4f}, acc: {val_acc:.4f}')
                    print(f'>>> Save model to {path}')
                
                if len(val_loss_list) - val_loss_list.index(min(val_loss_list)) > args.patience:
                    print(f'>>> Early stopping at epoch {epoch}')
                    break
        
    print(f"TESTING: loading from {path}")
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        test_loss, test_acc = eval_loop(model, plm_model, metrics, test_loader, loss_fn, device)
        if args.wandb:
            wandb.log({"test/test_loss": test_loss, "test/test_acc": test_acc})
    print(f'TEST loss: {test_loss:.4f} acc: {test_acc:.4f}')


def eval_loop(model, plm_model, metrics, dataloader, loss_fn, device=None):
    total_loss, total_acc = 0, 0
    epoch_iterator = tqdm(dataloader)
    
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch["label"]
        logits = model(plm_model, batch)
        loss = loss_fn(logits, label)
        total_loss += loss.item() * len(label)
        acc = metrics(logits, label).item()
        total_acc += acc * len(label)
        epoch_iterator.set_postfix(eval_loss=loss.item(), eval_acc=acc)
    
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = total_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc


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
    parser.add_argument('--monitor', type=str, default='val_acc', choices=['val_acc','val_loss'], help='monitor metric')
    parser.add_argument('--eval_every_n_steps', type=int, default=-1, help='evaluate every n steps')
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
    parser.add_argument("--wandb_entity", type=str, default="matwings")
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    # init wandb
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"LocSeek"
        if args.model_name is None:
            args.model_name = f"{args.wandb_run_name}.pt"
        
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, 
            entity=args.wandb_entity, config=vars(args)
        )
    
    # create checkpoint directory
    if args.ckpt_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.ckpt_dir = os.path.join(args.ckpt_root, current_date)
    else:
        args.ckpt_dir = os.path.join(args.ckpt_root, args.ckpt_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # load protein language model
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # build tokenizer
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
    
    args.vocab_size = plm_model.config.vocab_size
    if args.num_labels is None:
        args.num_labels = dataset_to_label[args.train_file.split("/")[1]]
    
    # load locseek model
    model = LocSeekModel(args)
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
            
            aa_seqs.append(aa_seq)
            foldseek_seqs.append(foldseek_seq)
            ss8_seqs.append(ss8_seq)
            labels.append(e["label"])
        
        aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
        aa_input_ids = aa_inputs["input_ids"]
        attention_mask = aa_inputs["attention_mask"]
        
        foldseek_input_ids = tokenizer(foldseek_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        ss8_input_ids = tokenizer(ss8_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        return {
            "aa_input_ids": aa_input_ids, 
            "attention_mask": attention_mask, 
            "foldseek_input_ids": foldseek_input_ids, 
            "ss8_input_ids": ss8_input_ids,
            "label": torch.as_tensor(labels, dtype=torch.long)
            }
        
    # metrics, optimizer, dataloader
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    metrics = Accuracy(task="multiclass", num_classes=args.num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.loss_fn == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_fn == "focal_loss":
        train_labels = [e["label"] for e in train_dataset]
        alpha = [len(train_labels) / train_labels.count(i) for i in range(args.num_labels)]
        print(">>> alpha: ", alpha)
        loss_fn = MultiClassFocalLossWithAlpha(num_classes=args.num_labels, alpha=alpha, device=device)
    
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