import pandas as pd
import numpy as np
import transformers 
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import zipfile 
import os

test_dir = "preds"


def prepare_data(lang):
    
    # taking the training data
    train_data, scores = [], []
    data = pd.read_csv("{}/{}_train.csv".format(lang, lang))
    cell_value_test = data.iloc[0, data.columns.get_loc('Text')]
    print(repr(cell_value_test))
    split_text = data["Text"].str.split("\n", expand=True)
    
    # taking the dev with labels data
    dev_data = pd.read_csv("{}/{}_dev_with_labels.csv".format(lang, lang))
    cell_value_dev = dev_data.iloc[0, dev_data.columns.get_loc('Text')]
    print(repr(cell_value_dev))
    split_text_dev = dev_data["Text"].str.split("\n", expand=True)
    
    train_data.extend(split_text[0])
    train_data.extend(split_text[1])
    train_data.extend(split_text_dev[0])
    train_data.extend(split_text_dev[1])
    
    scores.append(data["Score"].to_list())
    scores.append(dev_data["Score"].to_list())

    print("Total number of sentences for {} language : {}".format(lang, len(train_data)))
    
    return train_data, scores

def train(train_data, lang):

    model_name = 'distilroberta-base'
    word_embedding_model = models.Transformer(model_name, max_seq_length=32)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_data = [InputExample(texts = [s,s]) for s in train_data]
    trainloader = DataLoader(train_data, batch_size = 64, shuffle = True)
    
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(trainloader, train_loss)],
        epochs=15,
        show_progress_bar=True
    )
    model.save("output/simcse_model_{}".format(lang))

def calculate_cosine_similarity(embeddings):
    return cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]

def run_inference(lang):

    model_file = "output/simcse_model_{}".format(lang)
    test_model = SentenceTransformer(model_file)
    val_data = pd.read_csv("{}/{}_test.csv".format(lang, lang))
    cell_value_val = val_data.iloc[0,val_data.columns.get_loc('Text')]
    print(repr(cell_value_val))
    val_ids = val_data["PairID"].to_list()
    split_text = val_data["Text"].str.split("\n", expand=True)
    sentence_pairs = zip(split_text[0], split_text[1])
    sentence_pairs = list(sentence_pairs)

    sentence_embeddings = [test_model.encode(sentences) for sentences in sentence_pairs]

    val_preds = [calculate_cosine_similarity(embeddings) for embeddings in sentence_embeddings]
    results = pd.DataFrame({'PairID' : val_ids, 'Pred_Score' : val_preds})
    
    filename = "pred_{}_a.zip".format(lang)
    # results = results.drop(columns = ["Unnamed: 0"], inplace = True)  
    results.to_csv(os.path.join(test_dir, "pred_{}_a.csv".format(lang)), index = False)
    
    
    with zipfile.ZipFile(filename, "w") as zf:
        zf.write("pred_{}_a.csv".format(lang))


if __name__ == "__main__":
    

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    lang = "tel"    
    train_data, scores = prepare_data(lang)
    train(train_data, lang)
    run_inference(lang)


