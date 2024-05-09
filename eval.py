import argparse
import torch
import re
import json
import warnings
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import Accuracy, MulticlassMatthewsCorrCoef, F1Score, Recall
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer
from transformers import logging
from torch.utils.data import DataLoader
from src.utils.data_utils import BatchSampler
from src.models.adapter import AdapterModel

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def evaluate(model, plm_model, metrics, dataloader, loss_fn, device=None):
    total_loss = 0
    epoch_iterator = tqdm(dataloader)
    
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch["label"]
        logits = model(plm_model, batch)
        loss = loss_fn(logits, label)
        total_loss += loss.item() * len(label)
        for k, v in metrics.items():
            metrics[k].update(logits, label)
        epoch_iterator.set_postfix(eval_loss=loss.item())
    
    epoch_loss = total_loss / len(dataloader.dataset)
    for k, v in metrics.items():
        metrics[k] = v.compute()
        print(f"{k}: {metrics[k]}")
    return epoch_loss, metrics

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
    
    # dataset
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
    
    # load adapter model
    model = AdapterModel(args)
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
        
        if args.problem_type == 'regression':
            labels = torch.as_tensor(labels, dtype=torch.float)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)
        
        return {
            "aa_input_ids": aa_input_ids, 
            "attention_mask": attention_mask, 
            "foldseek_input_ids": foldseek_input_ids, 
            "ss8_input_ids": ss8_input_ids,
            "label": labels
            }
        
    loss_fn = nn.CrossEntropyLoss()
    accuary = Accuracy(task="multiclass", num_classes=args.num_labels, average=None).to(device)
    f1 = F1Score(task="multiclass", num_classes=args.num_labels, average=None).to(device)
    recall = Recall(task="multiclass", num_classes=args.num_labels, average=None).to(device)
    mcc = MulticlassMatthewsCorrCoef(num_classes=args.num_labels).to(device)
    metrics = {"accuary": accuary, "f1": f1, "recall": recall, "mcc": mcc}
    
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