from config import conf
from dataset import SquadDataset
from transformers import DistilBertTokenizerFast


if __name__ == '__main__':
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased')
    dataset = SquadDataset.from_json(conf['DATASET_FILE'], tokenizer)
    train_dataset, val_dataset = dataset.train_val_split(conf['TRAIN_RATIO'])
