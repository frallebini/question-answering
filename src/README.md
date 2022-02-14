# How to run

1. Place the [model file](https://liveunibo-my.sharepoint.com/:u:/g/personal/manuel_mariani2_studio_unibo_it/EVL8c-Sr-3NIkN41HmFOyvMBEkUMIvAIJ7Nv3xaJGr1fUg?e=mqHqg1) inside the root folder of the project.

2. Install the requirements:
    ```sh
    pip install -r requirements.txt
    ```

3. Run [`compute_answers.py`](compute_answers.py) to perform inference on a .json file formatted as the [training set](training_set.json) (but without answers); a file `predictions.txt` containing the answers will be generated:
    ```sh
    python3 compute_answers.py <path_to_json_file>
    ```
    > :warning: **Heavy process:** inference will run automatically on a GPU, if available. Although it can be run on a CPU, it is highly discouraged.

4. Run the [evaluation script](evaluate.py) to evaluate the results:
    ```sh
    python3 evaluate.py <path_to_ground_truth> predictions.txt
    ```
