To run the project, follow these steps:

1. Install the required packages from `requirements.txt` using the following command:

    ```
    pip install -r requirements.txt
    ```

2. To train the model, execute the following command:

    ```
    python src/train.py
    ```

3. To run inference on the model, execute the following command:

    ```
    python src/inference.py
    ```

Note that the `train.py` script generates the model weights and saves them to the `results` directory. The `inference.py` script loads the trained model weights and generates sentence-context pairs the input words.

You can modify the input data by editing the `word_context_pairs_1k_words_10contexts.csv` file in the `data` directory. 
