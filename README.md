# Question Answering

**Natural Language Processing Project**

*Alma Mater Studiorum Università di Bologna AY 2021–2022*

Question Answering is the task of selecting the span of text inside a passage that answers a given reading-comprehension question. Our task is to build a model performing question answering on the SQuAD1.1 dataset, which, as opposed to the 2.0 version, \emph{does not} contain unanswerable questions, namely questions whose answer cannot be extracted anywhere in the corresponding passage.

### Authors

Francesco Ballerini
[francesco.ballerini3@studio.unibo.it](mailto:francesco.ballerini3@studio.unibo.it)

Emmanuele Bollino
[emmanuele.bollino@studio.unibo.it](mailto:emmanuele.bollino@studio.unibo.it)

Tommaso Giannuli
[tommaso.giannuli@studio.unibo.it](mailto:tommaso.giannuli@studio.unibo.it)

Manuel Mariani
[manuel.mariani2@studio.unibo.it](mailto:manuel.mariani2@studio.unibo.it)

## Run

* Place the [model file](https://liveunibo-my.sharepoint.com/:u:/g/personal/manuel_mariani2_studio_unibo_it/EVL8c-Sr-3NIkN41HmFOyvMBEkUMIvAIJ7Nv3xaJGr1fUg?e=mqHqg1) inside the root folder of the project. You can download our final model using a Unibo account [from here](https://liveunibo-my.sharepoint.com/:u:/g/personal/manuel_mariani2_studio_unibo_it/EVL8c-Sr-3NIkN41HmFOyvMBEkUMIvAIJ7Nv3xaJGr1fUg?e=mqHqg1).

* Run the script [compute_answers.py](compute_answers.py) to execute the inference on the provided dataset and generate the answers file.
```sh
python3 compute_answers.py <path_to_json_file>
```
> :warning: **Heavy process:** Inference will run automatically on a GPU if available. Even if it is possibile to run it using a CPU, it is highly discouraged.

* Run the [evaluation script](evaluate.py) to evaluate the results.
```sh
python3 evaluate.py <path_to_ground_truth> predictions.txt
```
Of course, if the predictions path is changed, the evaluation command will become
```sh
python3 evaluate.py <path_to_ground_truth> <path_to_predictions_file>
```
