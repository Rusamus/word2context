The project directory structure is as follows:

.
├── data
│   └── word_context_pairs_1k_words_10contexts.csv
├── results
└── src
    ├── context_reverso.py
    ├── dataset.py
    ├── main.py
    ├── model.py
    ├── train.py
    ├── utils.py
    ├── word_context_pairs.py
    ├── .gitignore
    └── requirements.txt

To run the project, first make sure you have installed the required packages from requirements.txt. You can install them using the following command:

pip install -r requirements.txt

After installing the required packages, you can run the main.py script as follows:

python src/main.py

To run the inference script, simply execute the following command:

python src/inference.py
