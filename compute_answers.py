import torch
import pandas as pd
import numpy as np
import json
import transformers
from tqdm import tqdm
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import BertForQuestionAnswering, BertModel

# CONFIGS
OUTPUT_FILE_PATH = "predictions.txt"
HF_BASE_MODEL_NAME = "prajjwal1/bert-medium"
MODEL_WEIGHTS_PATH = 'bert-medium-final.pt'

BATCH_SIZE = 16
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 1. Loading the json into a pandas df
def load_json(json_path) -> pd.DataFrame:
    with open(json_path, 'r') as file:
        raw = json.load(file)['data']
    data = []
    for topic in raw:
        for paragraph in topic['paragraphs']:
            for question in paragraph['qas']:
                data.append((question['id'], paragraph['context'], question['question']))
    return pd.DataFrame(data, columns=['id', 'context', 'question'])


# 2. Tokenization
def tokenize(df: pd.DataFrame) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL_NAME)
    tok = []
    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        t = tokenizer(
            row['question'],
            row['context'],
            max_length=512,
            truncation='only_second',
            padding='max_length',
            return_offsets_mapping=True,
        )
        tok.append({
            'input_ids': torch.tensor(t['input_ids']),
            'token_type_ids': torch.tensor(t['token_type_ids']),
            'attention_mask': torch.tensor(t['attention_mask']),
            'offset_mapping': torch.tensor(t['offset_mapping']),
        })
    return pd.concat((df, pd.DataFrame(tok)), axis=1)


# 3. Model loading
class CustomBertForQuestionAnswering(BertForQuestionAnswering):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(config.hidden_size, config.num_labels),
        )


def load_model() -> BertForQuestionAnswering:
    model = CustomBertForQuestionAnswering.from_pretrained(HF_BASE_MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    return model


# 4. PyTorch Dataset
class QADataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {
            'input_ids': item['input_ids'],
            'token_type_ids': item['token_type_ids'],
            'attention_mask': item['attention_mask'],
        }

    @staticmethod
    def load(ds, batch_size) -> DataLoader:
        return DataLoader(QADataset(ds), batch_size=batch_size, shuffle=False)


# 5. Compute answers
def compute_answers(model, loader) -> pd.DataFrame:
    # Store the answers' TOK indices
    answer_start_tok = []
    answer_end_tok = []

    # Get outputs from the model
    model.to(device)
    model.eval()

    # Note: comments not included for brevity. See training.ipynb for the explanation
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), leave=False):
            args = dict(
                input_ids=batch['input_ids'].to(device, non_blocking=True),
                attention_mask=batch['attention_mask'].to(device, non_blocking=True),
                token_type_ids=batch['token_type_ids'].to(device, non_blocking=True),
            )
            outputs = model(**args)

            mask = ((args['token_type_ids'] - 1) * torch.inf).nan_to_num(0)
            start_logits = outputs['start_logits'] + mask
            end_logits = outputs['end_logits'] + mask
            start_tok_indices = torch.argmax(start_logits, dim=1).tolist()
            end_tok_indices = torch.argmax(end_logits, dim=1).tolist()

            for k in range(len(start_tok_indices)):
                sti, eti = start_tok_indices[k], end_tok_indices[k]
                if sti <= eti:
                    answer_start_tok.append(sti)
                    answer_end_tok.append(eti)
                    continue
                valid_logits = {}
                candidate_sti = start_logits[k].argsort(descending=True)[:20]
                candidate_eti = end_logits[k].argsort(descending=True)[:20]
                for i in candidate_sti:
                    for j in candidate_eti:
                        if i <= j:
                            valid_logits[(i, j)] = (start_logits[k, i] + end_logits[k, j]).item()
                if len(valid_logits) > 0:
                    sti, eti = max(valid_logits, key=valid_logits.get)
                answer_start_tok.append(sti)
                answer_end_tok.append(eti)

    # Convert the TOK indices into TEXT using the context
    df = loader.dataset.df.reset_index()
    answer_text_pred = []
    for idx, row in df.iterrows():
        om = row['offset_mapping']
        tst, ten = answer_start_tok[idx], answer_end_tok[idx]  # Tok start, tok end
        cst, cen = om[tst, 0], om[ten, 1]  # Char start, Char end
        answer_text_pred.append(row['context'][cst:cen])
    answer_text_pred = pd.Series(answer_text_pred, name='answer_text_pred')

    return pd.concat([df, answer_text_pred], axis=1)


# 4+5. Handle possible torch memory errors
def compute_answers_wrapper(model, df) -> pd.DataFrame:
    bs = BATCH_SIZE
    while bs >= 1:
        try:
            res = compute_answers(model, QADataset.load(df, bs))
            return res
        except RuntimeError as e:
            msg = str(e).lower()
            if 'memory' in msg or 'cuda' in msg:
                bs = bs // 2
                print(f"  ⚠️ Not enough memory, retrying with batch size: {bs}")
            else:
                raise e


# 6. Save the answers into a json
def answers_to_file(df: pd.DataFrame):
    raw_out = {}
    df.apply(lambda row: raw_out.update({row['id']: row['answer_text_pred']}), axis=1)
    with open(OUTPUT_FILE_PATH, 'w+') as fout:
        fout.write(json.dumps(raw_out))


def main():
    # Parse the command line, taking in input the
    parser = argparse.ArgumentParser(description="Compute answers from a squad dataset.")
    parser.add_argument('json_path', type=str, help='path to a SQuAD dataset (.json)')
    args = parser.parse_args()
    squad_path = args.json_path

    transformers.logging.set_verbosity_error()
    print()

    # Load the dataset
    print(f"🚚 Loading {squad_path}")
    df = load_json(squad_path)

    # Tokenization
    print("✂️ Tokenizing")
    df = tokenize(df)

    # Model loading
    print("⬇️ Loading model")
    model = load_model()

    # Computing answers
    print("🤖 Computing answers")
    res = compute_answers_wrapper(model, df)

    # Saving to file
    print(f"🖨️ Saving answers in {OUTPUT_FILE_PATH}")
    answers_to_file(res)

    print("✨✨✨  DONE  ✨✨✨")


if __name__ == '__main__':
    main()
