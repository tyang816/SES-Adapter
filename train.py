import argparse
import warnings
import torch
import os
import sys
sys.path.append(os.getcwd())
import wandb
import random
import json
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import logging
from torchmetrics.classification import Accuracy
from accelerate import Accelerator
from accelerate.utils import set_seed
from time import strftime, localtime
from transformers import EsmTokenizer, EsmModel
from src.utils.data_utils import BatchSampler
from src.models.locseek import LocSeekModel

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def train(args, model, plm_model, accelerator, metrics, train_loader, val_loader, test_loader, 
          optimizer, device):
    best_val_loss, best_val_acc = float("inf"), 0
    val_loss_list, val_acc_list = [], []
    loss_fn = nn.CrossEntropyLoss()
    path = os.path.join(args.ckpt_dir, args.model_name)
    for epoch in range(args.max_train_epochs):
        print(f"---------- Epoch {epoch} ----------")
        model.train()
        train_loss, train_acc = loop(
            model, plm_model, accelerator, metrics, train_loader, loss_fn, epoch,
            optimizer, use_wandb=args.wandb, device=device
            )
        print(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f} acc: {train_acc:.4f}')
        
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = loop(
                model, plm_model, accelerator, metrics, val_loader, loss_fn, epoch, 
                use_wandb=args.wandb, device=device
                )
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            if args.wandb:
                wandb.log({"valid/val_loss": val_loss, "valid/val_acc": val_acc, "valid/epoch": epoch})
        print(f'EPOCH {epoch} VAL loss: {val_loss:.4f} acc: {val_acc:.4f}')
        
        if args.monitor == "val_acc":
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), path)
                print(f'>>> BEST at epcoh {epoch}, loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f}')
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
        loss, acc = loop(
            model, plm_model, accelerator, metrics, test_loader, loss_fn, 
            use_wandb=args.wandb, device=device
            )
        if args.wandb:
            wandb.log({"test/test_loss": loss, "test/test_acc": acc})
    print(f'TEST loss: {loss:.4f} acc: {acc:.4f}')

    
def loop(model, plm_model, accelerator, metrics, dataloader, loss_fn, 
         epoch=0, optimizer=None, use_wandb=False, device=None):
    
    total_loss, total_acc = 0, 0
    iter_num = len(dataloader)
    global_steps = epoch * len(dataloader)
    epoch_iterator = tqdm(dataloader)
    
    for batch in epoch_iterator:
        with accelerator.accumulate(model):
            for k, v in batch.items():
                batch[k] = v.to(device)
            label = batch["label"]
            logits = model(plm_model, batch)
            loss = loss_fn(logits, label)
            total_loss += loss.item()
            acc = metrics(logits, label).item()
            total_acc += acc
            
            global_steps += 1
            
            if optimizer:
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                epoch_iterator.set_postfix(train_loss=loss.item(), train_acc=acc)
                if use_wandb:
                    wandb.log({"train/train_loss": loss.item(), "train/train_acc": acc, "train/epoch": epoch}, step=global_steps)
            else:
                epoch_iterator.set_postfix(eval_loss=loss.item(), eval_acc=acc)
    
    epoch_loss = total_loss / iter_num
    epoch_acc = total_acc / iter_num
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--num_labels', type=int, default=2, help='number of labels')
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
    parser.add_argument('--use_foldseek', action='store_true', help='use foldseek')
    parser.add_argument('--use_ss8', action='store_true', help='use ss8')
    
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
    plm_model = EsmModel.from_pretrained(args.plm_model).to(device).eval()
    for param in plm_model.parameters():
        param.requires_grad = False
    if args.hidden_size is None:
        if "esm2" in args.plm_model:
            args.hidden_size = plm_model.config.hidden_size
        elif "prot_t5" in args.plm_model:
            args.hidden_size = plm_model.config.d_model
        else:
            raise ValueError("hidden_size is not provided")
    
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
    
    # build tokenizer
    aa_tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
    foldseek_tokenizer = EsmTokenizer(vocab_file="src/vocab/foldseek.txt")
    ss8_tokenizer = EsmTokenizer(vocab_file="src/vocab/ss8.txt")
    
    def collate_fn(examples):
        aa_seq, foldseek_seq, ss8_seq, label = [], [], [], []
        for e in examples:
            if args.max_seq_len is not None:
                aa_seq.append(e["aa_seq"][:args.max_seq_len])
                foldseek_seq.append(e["foldseek_seq"][:args.max_seq_len])
                ss8_seq.append(e["ss8_seq"][:args.max_seq_len])
            else:
                aa_seq.append(e["aa_seq"])
                foldseek_seq.append(e["foldseek_seq"])
                ss8_seq.append(e["ss8_seq"])
            
            label.append(e["label"])
        
        aa_inputs = aa_tokenizer(aa_seq, return_tensors="pt", padding=True, truncation=True)
        aa_input_ids = aa_inputs["input_ids"]
        attention_mask = aa_inputs["attention_mask"]
        
        foldseek_input_ids = foldseek_tokenizer(foldseek_seq, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        ss8_input_ids = ss8_tokenizer(ss8_seq, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        return {
            "aa_input_ids": aa_input_ids, 
            "attention_mask": attention_mask, 
            "foldseek_input_ids": foldseek_input_ids, 
            "ss8_input_ids": ss8_input_ids,
            "label": torch.as_tensor(label, dtype=torch.long)
            }
        
    # metrics, optimizer, dataloader
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    metrics = Accuracy(task="multiclass", num_classes=args.num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
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
    train(args, model, plm_model, accelerator, metrics, train_loader, val_loader, test_loader, optimizer, device)
    
    if args.wandb:
        wandb.finish()