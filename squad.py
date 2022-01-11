import json
from collections import OrderedDict


def rearrange_squad(fname: str) -> None:
    """
    Reads the training set of SQuaD 1.1 and creates a .json file
    storing one sample for each question, where each sample has shape
    ```
    id (int): {
        title (str): ...
        context (str): ...
        question (str): ...
        answer_start (int): ...
        answer_text (str): ...
    }
    ```
    """
    with open(fname) as f:
        squad = json.load(f)
    
    samples = OrderedDict()
    for topic in squad['data']:
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
                samples[id] = {
                    'title': title,
                    'context': context,
                    'question': question,
                    'answer_start': answer_start,
                    'answer_text': answer_text
                }
    
    name, ext = fname.split('.')
    with open(f'{name}_rearranged.{ext}', 'w') as f:
        json.dump(samples, f, indent=4)


if __name__ == '__main__':
    rearrange_squad('training_set.json')