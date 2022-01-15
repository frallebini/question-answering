from __future__ import annotations
import json
import torch
import transformers
from random import randint


def read_squad(path: str) -> tuple[list(str), list(str), list(str), list(dict)]:
    """
    Reads the SQuaD 1.1 training set (a .json file) and, for each question, 
    stores its context, text (i.e. the question itself), id, and answer info 
    (i.e. answer start index and answer text) into a list.
    """
    with open(path, 'rb') as f:
      squad = json.load(f)

    contexts = []
    questions = []
    ids = []
    answers = []

    for topic in squad['data']:
        for paragraph in topic['paragraphs']:
            context = paragraph['context']
            for question_data in paragraph['qas']:
                question = question_data['question']
                id = question_data['id']
                # check that no question in the training set has more than one answer
                assert len(question_data['answers']) == 1
                answer = question_data['answers'][0]

                contexts.append(context)
                questions.append(question)
                ids.append(id)
                answers.append(answer)

    return contexts, questions, ids, answers


def add_end_idx(answers: list(dict), contexts: list(str)) -> None:
    """
    Adds a field to each dictionary in `answers` denoting the index of the 
    last character of the answer text within its context, so as to identify
    where the answer ends.
    """
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']  
        start_idx = answer['answer_start']  
        end_idx = start_idx + len(gold_text) - 1

        assert answer['text'] == context[start_idx : end_idx + 1]  # sanity check

        answer['answer_end'] = end_idx


def encode(contexts: list(str), 
           questions: list(str), 
           tokenizer: transformers.BertTokenizer) -> transformers.BatchEncoding:
    """
    Creates BERT context–question encodings, i.e. context token indices + [SEP] + question token indices.
    """
    encodings = tokenizer(contexts, questions, padding=True, truncation=True)
    return encodings


def add_token_positions_and_ids(encodings: transformers.BatchEncoding, 
                                answers: list(str), 
                                ids: list(str),
                                tokenizer: transformers.BertTokenizer) -> None:
    """
    Adds three fields to the `BatchEncoding` object returned by `encode` (which is basically
    a standard Python dictionary):
    - The index of the first token of the answer.
    - The index of the last token of the answer.
    - The ID of the answer.
    """
    start_positions = []
    end_positions = []
    
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        if start_positions[-1] is None:  # the answer passage has been completely truncated
            start_positions[-1] = tokenizer.model_max_length
        shift = 1
        while end_positions[-1] is None:  # the answer passage has been patially truncated
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1

    encodings.update({
        'start_positions': start_positions, 
        'end_positions': end_positions, 
        'ids': ids
    })


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key != 'ids'}

    def __len__(self):
        return len(self.encodings.input_ids)


def build_dataset(fname: str, tokenizer: transformers.BertTokenizer = None) -> SquadDataset:
    if not tokenizer:
        tokenizer = transformers.DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    contexts, questions, ids, answers = read_squad(fname)
    add_end_idx(answers, contexts)
    encodings = encode(contexts, questions, tokenizer)
    add_token_positions_and_ids(encodings, answers, ids, tokenizer)
    dataset = SquadDataset(encodings)
    
    return dataset


if __name__ == '__main__':
    dataset = build_dataset('training_set.json')
    print(dataset[0])
