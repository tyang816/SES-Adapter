import argparse
import torch
import re
import json
import warnings
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import Accuracy, MulticlassMatthewsCorrCoef, F1Score, Recall
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer, T5Tokenizer, T5Model
from transformers import logging
from torch.utils.data import DataLoader
from src.utils.data_utils import BatchSampler
from src.models.locseek import LocSeekModel

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def evaluate(model, plm_model, metrics, dataloader, loss_fn, device=None):
    total_loss = 0
    epoch_iterator = tqdm(dataloader)
    metrics_result_dict = {}
    for k in metrics:
        metrics_result_dict[k] = 0
    
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch["label"]
        logits = model(plm_model, batch)
        loss = loss_fn(logits, label)
        total_loss += loss.item() * len(label)
        for k, v in metrics.items():
            metrics_result_dict[k] += v(logits, label).item() * len(label)
        epoch_iterator.set_postfix(eval_loss=loss.item())
    
    epoch_loss = total_loss / len(dataloader.dataset)
    for k in metrics_result_dict:
        metrics_result_dict[k] /= len(dataloader.dataset)
        print(f"{k}: {metrics_result_dict[k]}")
    return epoch_loss, metrics_result_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--num_labels', type=int, default=2, help='number of labels')
    parser.add_argument('--pooling_method', type=str, default='attention1d', help='pooling method')
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help='pooling dropout')
    
    
    parser.add_argument('--test_file', type=str, default=None, help='test file')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--max_batch_token', type=int, default=10000, help='max number of token per batch')
    parser.add_argument('--use_foldseek', action='store_true', help='use foldseek')
    parser.add_argument('--use_ss8', action='store_true', help='use ss8')
    
    
    # model path
    parser.add_argument('--model_name', type=str, default=None, help='model name')
    parser.add_argument('--ckpt_root', default="ckpt", help='root directory to save trained models')
    parser.add_argument('--ckpt_dir', default=None, help='directory to save trained models')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # build tokenizer
    if "esm" in args.plm_model:
        aa_tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        foldseek_tokenizer = EsmTokenizer(vocab_file="src/vocab/esm/foldseek.txt")
        ss8_tokenizer = EsmTokenizer(vocab_file="src/vocab/esm/ss8.txt")
        plm_model = EsmModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "bert" in args.plm_model:
        aa_tokenizer = BertTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        foldseek_tokenizer = BertTokenizer(vocab_file="src/vocab/prot_bert/foldseek.txt", do_lower_case=False)
        ss8_tokenizer = BertTokenizer(vocab_file="src/vocab/prot_bert/ss8.txt", do_lower_case=False)
        plm_model = BertModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "prot_t5" in args.plm_model:
        aa_tokenizer = T5Tokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        foldseek_tokenizer = T5Tokenizer(vocab_file="src/vocab/prot_t5/foldseek.txt", do_lower_case=False)
        ss8_tokenizer = T5Tokenizer(vocab_file="src/vocab/prot_t5/ss8.txt", do_lower_case=False)
        plm_model = T5Model.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    
    # load locseek model
    model = LocSeekModel(args)
    model_path = f"{args.ckpt_root}/{args.ckpt_dir}/{args.model_name}"
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    def param_num(model):
        total = sum([param.numel() for param in model.parameters() if param.requires_grad])
        num_M = total/1e6
        if num_M >= 1000:
            return "Number of parameter: %.2fB" % (num_M/1e3)
        else:
            return "Number of parameter: %.2fM" % (num_M)
    print(param_num(model))
     
    def collate_fn(examples):
        aa_seqs, foldseek_seqs, ss8_seqs, labels = [], [], [], []
        for e in examples:
            aa_seq, foldseek_seq, ss8_seq = e['aa_seq'], e['foldseek_seq'], e['ss8_seq']
            
            if args.max_seq_len is not None:
                aa_seq = aa_seq[:args.max_seq_len]
                foldseek_seq = foldseek_seq[:args.max_seq_len]
                ss8_seq = ss8_seq[:args.max_seq_len]

            if 'prot_bert' in args.plm_model or "prot_t5" in args.plm_model:
                aa_seq = " ".join(list(aa_seq))
                aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
                foldseek_seq = " ".join(list(foldseek_seq))
                ss8_seq = " ".join(list(ss8_seq))
            
            aa_seqs.append(e["aa_seq"])
            foldseek_seqs.append(e["foldseek_seq"])
            ss8_seqs.append(e["ss8_seq"])
            
            labels.append(e["label"])
        
        aa_inputs = aa_tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
        aa_input_ids = aa_inputs["input_ids"]
        attention_mask = aa_inputs["attention_mask"]
        
        foldseek_input_ids = foldseek_tokenizer(foldseek_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        ss8_input_ids = ss8_tokenizer(ss8_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        return {
            "aa_input_ids": aa_input_ids, 
            "attention_mask": attention_mask, 
            "foldseek_input_ids": foldseek_input_ids, 
            "ss8_input_ids": ss8_input_ids,
            "label": torch.as_tensor(labels, dtype=torch.long)
            }
        
    loss_fn = nn.CrossEntropyLoss()
    metrics = {
        "acc": Accuracy(task="multiclass", num_classes=args.num_labels).to(device),
        "mcc": MulticlassMatthewsCorrCoef(num_classes=args.num_labels).to(device),
        "f1": F1Score(task="multiclass", num_classes=args.num_labels).to(device),
        "recall": Recall(task="multiclass", average='macro', num_classes=args.num_labels).to(device)
    }
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

    test_dataset, test_token_num = load_dataset(args.test_file)
    test_loader = DataLoader(
        test_dataset, num_workers=args.num_workers, collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, args.max_batch_token, False)
        )

    print("---------- Start Eval ----------")
    with torch.no_grad():
        evaluate(model, plm_model, metrics, test_loader, loss_fn, device)