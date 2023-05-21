Description: This GitHub repository hosts a natural language processing project utilizing a Transformer-based model to generate concise and unambiguous sentence-contexts for given keywords.

Motivation: The project aims to address the challenge of creating meaningful sentence-contexts from limited input information (a single keyword as query), which is crucial for various natural language understanding and generation applications.

Use-cases:

Enhancing search engine results by providing relevant sentence-contexts for user queries.
Improving content summarization tools by generating accurate and meaningful contexts for keywords.
Assisting personal assistants and chatbots in providing more informative responses based on user inputs.
Facilitating text analytics and information retrieval tasks by creating concise and clear sentence-context representations for keywords.


To run the project, follow these steps:

1. Install the required packages from `requirements.txt` using the following command:

    ```
    pip install -r requirements.txt
    ```

2. To train the model, execute the following command:

    ```
    python src/main.py
    ```

3. To run inference on the model, execute the following command:

    ```
    python src/inference.py
    ```

Note that the `train.py` script generates the model weights and saves them to the `results` directory. The `inference.py` script loads the trained model weights and generates sentence-context pairs the input words.

You can modify the input data by editing the `word_context_pairs_1k_words_10contexts.csv` file in the `data` directory. 

Here's the text with the added information:

Pretrained model checkpoints and training data can be accessed via the following Google Drive link: https://drive.google.com/drive/u/1/folders/1Zm41qpdWjUBmN2792YzUxfPYIbefC5yB

In an evaluation of three models, T5-small, T5-base, and GPT-2, we compared their performance using the BLEU and METEOR metrics, as well as the number of parameters involved. Table \ref{tab:performance_comparison} shows the comparison for 10k samples, revealing that the T5-base model performed better in terms of both BLEU and METEOR scores. For 1M samples, as shown in Table \ref{tab:performance_comparison_large}, the T5-base model still outperformed the other models in terms of BLEU and METEOR scores. However, it is worth noting that the T5-small model had fewer parameters than both the T5-base and GPT