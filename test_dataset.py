from __future__ import annotations
import json
import transformers
import pandas as pd
from dataset import SquadDataset


class Test_SquadDataset(SquadDataset):
    """
    PyTorch wrapper for SQuaD 1.1. (for testing purposes, therefore with no 
    answers).
    """

    @classmethod
    def _data_from_json(cls, path: str) -> pd.DataFrame:
        """
        Reads the SQuaD 1.1 training set (a .json file) and stores each question
        and related information into a Pandas dataframe.
        """
        with open(path, 'rb') as f:
            squad = json.load(f)

        raw_data = []

        for topic in squad['data']:
            title = topic['title']
            for paragraph in topic['paragraphs']:
                context = paragraph['context']
                for question_data in paragraph['qas']:
                    question = question_data['question']
                    id = question_data['id']

                    raw_data.append((id,
                                     title,
                                     context,
                                     question))

        data = pd.DataFrame.from_records(raw_data, columns=['id',
                                                            'title',
                                                            'context',
                                                            'question'])

        return data

    @classmethod
    def _encode(cls, 
                data: pd.DataFrame, 
                tokenizer: transformers.BertTokenizerFast 
                ) -> transformers.BatchEncoding:
        """
        Creates BERT context-question encodings, i.e. context token indices 
        (a.k.a. input IDs) + [SEP] + question token indices.
        """
        contexts = list(data['context'])
        questions = list(data['question'])
        encodings = tokenizer(contexts, questions, padding=True, truncation=True)

        return encodings
