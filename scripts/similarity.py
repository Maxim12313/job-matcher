from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity


# https://www.geeksforgeeks.org/sentence-similarity-using-bert-transformer/
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    sentence1 = "I like coding in Python."
    sentence2 = "Python is my favorite programming language."

    tokens1 = tokenizer(sentence1, padding=True, truncation=True, return_tensors="pt")
    tokens2 = tokenizer(sentence2, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        out1 = model(**tokens1)
        out2 = model(**tokens2)
        # select CLS token, all batches and all dimensions
        embed1 = out1.last_hidden_state[:, 0, :]
        embed2 = out2.last_hidden_state[:, 0, :]
        res = cosine_similarity(embed1, embed2)
        print(res)
