from __future__ import annotations
import json
import torch
import transformers
import pandas as pd
import math


class SquadDataset(torch.utils.data.Dataset):
    """
    Represents an instance of the SQuAD 1.1 Dataset
    """

    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: transformers.BertTokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encodings = None
        self.updated = False

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self.update_encoding()

        return {
            k: torch.tensor(v[index])
            for k, v in self.encodings.items() if k != 'ids'
        }

    def __len__(self) -> int:
        self.update_encoding()

        return len(self.encodings.input_ids)

    @classmethod
    def _data_from_json(cls, path: str) -> pd.DataFrame:
        """
        Reads the SQuaD 1.1 training set (a .json file) and stores each question and
        related information into a Pandas dataframe.
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
                    # check that no question in the training set has more than one
                    # answer
                    assert len(question_data['answers']) == 1
                    answer = question_data['answers'][0]
                    answer_text = answer['text']
                    answer_start = answer['answer_start']

                    answer_end = answer_start + len(answer_text) - 1
                    assert answer_text == context[answer_start: answer_end + 1]

                    raw_data.append((id,
                                     title,
                                     context,
                                     question,
                                     answer_text,
                                     answer_start,
                                     answer_end))

        data = pd.DataFrame.from_records(raw_data, columns=['id',
                                                            'title',
                                                            'context',
                                                            'question',
                                                            'answer',
                                                            'answer_start',
                                                            'answer_end'])

        return data

    @classmethod
    def from_json(cls, path: str, tokenizer: transformers.BertTokenizer) -> SquadDataset:
        """
        Instantiates a SquadDataset from a JSON file
        """
        return cls(cls._data_from_json(path), tokenizer)

    @classmethod
    def _train_val_split_data(cls, data: pd.DataFrame, train_ratio=0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the given dataframe into training and validation data.
        Specifically, the first `ceil(n_samples * train_ratio)` samples are
        reserved for training and the remainder for validation.

        If, by doing so, the split occurs between samples with the same `title`,
        then the smaller subset of such samples is moved from training to validation
        data or viceversa. As a result, training and validation data do not share
        any `title`.
        """
        n_samples = data.shape[0]
        n_train = math.ceil(n_samples * train_ratio)
        n_val = n_samples - n_train

        train_data = data.iloc[:n_train]
        val_data = data.iloc[n_train:].reset_index(drop=True)

        # since samples are still ordered, the title of the first sample in the
        # validation set is the one that might have been split between the two
        # sets
        shared_title = val_data.iloc[0]['title']
        if shared_title not in train_data['title']:
            # we got lucky: the split occurs precisely where a title ends and the
            # new one begins
            return train_data, val_data

        shared_title_mask = lambda df: df['title'] == shared_title
        shared_title_count = lambda df: df[shared_title_mask(df)].shape[0]

        n_shared_title_train = shared_title_count(train_data)
        n_shared_title_var = shared_title_count(val_data)

        # if the training set contains less samples with the shared title then the
        # validation set, then move those samples to the validation set
        if n_shared_title_train < n_shared_title_var:
            train_portion = train_data[shared_title_mask(train_data)]
            val_data = pd.concat((train_portion, val_data), ignore_index=True)
            train_data = train_data.drop(train_portion.index)
            assert val_data.shape[0] == n_val + train_portion.shape[0]
            assert train_data.shape[0] == n_train - train_portion.shape[0]
        else:  # otherwise do the opposite
            val_portion = val_data[shared_title_mask(val_data)]
            train_data = pd.concat((train_data, val_portion), ignore_index=True)
            val_data = val_data.drop(val_portion.index)
            assert train_data.shape[0] == n_train + val_portion.shape[0]
            assert val_data.shape[0] == n_val - val_portion.shape[0]

        assert set(train_data['title']).intersection(val_data['title']) == {}

        return train_data, val_data

    def train_val_split(self, train_ratio=0.75) -> tuple[SquadDataset, SquadDataset]:
        """
        Splits the current SquadDataset into two SquadDatasets.
        One for training and the other for validation purposes.
        """
        train_data, val_data = self._train_val_split_data(self.data, train_ratio)
        return SquadDataset(train_data, self.tokenizer), SquadDataset(val_data, self.tokenizer)

    def update_encoding(self, force=False):
        """
        Updates the encodings if necessary. Encodings are produced in a lazy way.
        """
        if self.encodings is None or self.updated or force:
            self.encodings = self._encode(self.data, self.tokenizer)

    @classmethod
    def _encode(cls, data: pd.DataFrame, tokenizer: transformers.BertTokenizer) -> transformers.BatchEncoding:
        """
        Creates BERT context-question encodings, i.e. context token indices (a.k.a.
        input IDs) + [SEP] + question token indices.

        Encodings will contain three more fields to the `BatchEncoding` object returned by `encode` (which
        is basically a standard Python dictionary):
        - The index of the first token of the answer.
        - The index of the last token of the answer.
        - The ID of the answer.
        """
        contexts = list(data['context'])
        questions = list(data['question'])
        encodings = tokenizer(contexts, questions, padding=True, truncation=True)

        start_positions = []
        end_positions = []
        answer_starts = list(data['answer_start'])
        answer_ends = list(data['answer_end'])
        ids = list(data['id'])

        for i in range(len(answer_starts)):
            start_positions.append(encodings.char_to_token(i, answer_starts[i]))
            end_positions.append(encodings.char_to_token(i, answer_ends[i]))

            if start_positions[-1] is None:
                # the answer passage has been completely truncated
                start_positions[-1] = tokenizer.model_max_length
            shift = 1
            while end_positions[-1] is None:
                # the answer passage has been partially truncated
                end_positions[-1] = encodings.char_to_token(
                    i, answer_ends[i] - shift)
                shift += 1

        encodings.update({
            'start_positions': start_positions,
            'end_positions': end_positions,
            'ids': ids
        })

        return encodings

    def set_data(self, data: pd.DataFrame):
        """
        Changes the current dataframe of the SquadDataset object.
        """
        self.data = data
        self.updated = True
