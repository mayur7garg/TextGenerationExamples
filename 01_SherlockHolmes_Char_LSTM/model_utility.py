import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.notebook import trange

def read_book_data(book_dir: Path, file_pat: str = "*.txt", seq_len: int = 512):
    X = []
    y = []

    for book_file_path in book_dir.rglob(file_pat):
        with book_file_path.open('r', encoding = 'utf-8') as book_file:
            book_data = book_file.read()
            book_data = re.sub("\n", " ", book_data)
            book_data = re.sub("[ ]+", " ", book_data)
            char_len = len(book_data)

            for i in range(0, char_len - seq_len):
                X.append(book_data[i : i + seq_len])
                y.append(book_data[i + seq_len])

    return X, y

def get_train_val_data(book_dir: Path, file_pat: str = "*.txt", seq_len: int = 512, val_size: float = 0.05, random_state: int = 7):
    X, y = read_book_data(book_dir, file_pat, seq_len)
    return train_test_split(X, y, test_size = val_size, random_state = random_state)

def generate_text(model, vocab, input_text: str,  chars_to_predict: int = 256):
    pred_output = ''

    for _ in trange(chars_to_predict, desc = "Predicting chars", unit = " char"):
        pred = model.predict([input_text], verbose = False)
        pred_char_id = pred.argmax()
        pred_char = vocab[pred_char_id]
        pred_output += pred_char
        input_text = input_text[1:] + pred_char

    return pred_output