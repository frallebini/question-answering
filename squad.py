from __future__ import annotations
import json
from collections import OrderedDict
from tqdm import tqdm
from transformers.data.processors.squad import SquadExample


def get_examples_from_dataset(fname: str, fwrite=False) -> list[SquadExample]:
    """
    Reads the training set of SQuaD 1.1 and creates a `SquadExample` object for each question.

    Args:
        fname: name of the file containing the training set.
        fwrite: whether to store the examples in a .json file (for visual inspection).
    """
    with open(fname) as f:
        squad = json.load(f)
    
    examples = []
    fcontent = OrderedDict()
    for topic in tqdm(squad['data']):
        title = topic['title']
        for paragraph in topic['paragraphs']:
            context = paragraph['context']
            for question_data in paragraph['qas']:  
                question = question_data['question']
                id = question_data['id']
                answers = question_data['answers']
                assert len(answers) == 1  # check that no question in the training set has more than one answer
                answer_start = answers[0]['answer_start']
                answer_text = answers[0]['text']
                
                example = SquadV1Example(
                    qas_id=id,
                    question_text=question,
                    context_text=context,
                    answer_text=answer_text,
                    start_position_character=answer_start,
                    title=title)
                examples.append(example)
                if fwrite: fcontent[example.qas_id] = example.as_dict()
    
    if fwrite:
        name, ext = fname.split('.')
        with open(f'{name}_rearranged.{ext}', 'w') as f:
            json.dump(fcontent, f, indent=4)

    return examples


class SquadV1Example(SquadExample):
    def as_dict(self) -> dict:
        return {
            'title': self.title,
            'context': self.context_text,
            'question': self.question_text,
            'answer_start': self.start_position,
            'answer_text': self.answer_text
        }
            

if __name__ == '__main__':
    examples = get_examples_from_dataset('training_set.json')