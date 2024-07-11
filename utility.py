from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utility import * 
data_dir = "dataset"
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder as enc , LabelEncoder as le






def get_lines(filename):
    """  
    input - filename
    output- file text as a list
    """
    with open(filename,"r") as f:
        return f.readlines()
    

def preprocess_text_with_line_number(filename):
    """
    returns list of dict of abstracts

    """
    input_lines = get_lines(filename)

    abstract_lines = ""
    abstract_samples = []

    for line in input_lines:
        if(line.startswith("###")):
            abstract_id = line
            abstract_lines = ""

        elif(line.isspace()):
            abstract_line_split = abstract_lines.splitlines()

            for abstract_line_num,abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t")
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1].lower()
                line_data["line_num"] = abstract_line_num
                line_data["total_lines"] = len(abstract_line_split)-1
                abstract_samples.append(line_data)

        else:
            abstract_lines += line

    return abstract_samples

def calculate_results(y_true, y_pred):

  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results


def preprocess_for_model(filename):
    train_samples = preprocess_text_with_line_number(filename + "train.txt")
    test_samples = preprocess_text_with_line_number(filename + "test.txt")
    val_samples = preprocess_text_with_line_number(filename + "dev.txt")
    train_df = pd.DataFrame(train_samples)
    test_df = pd.DataFrame(test_samples)
    val_df = pd.DataFrame(val_samples)

    train_sentences = train_df["text"].tolist()
    test_sentences = test_df["text"].tolist()
    val_sentences = val_df["text"].tolist()

    encoder = enc(sparse=False)
    train_labels = encoder.fit_transform(train_df["target"].to_numpy().reshape(-1,1))
    test_labels = encoder.transform(test_df["target"].to_numpy().reshape(-1,1))
    val_labels = encoder.transform(val_df["target"].to_numpy().reshape(-1,1))

    train_chars = [split_chars(sentence) for sentence in train_sentences]
    test_chars = [split_chars(sentence) for sentence in test_sentences]
    val_chars = [split_chars(sentence) for sentence in val_sentences]

    train_line_num_one_hot = tf.one_hot(train_df["line_num"].to_numpy(),depth = 15)
    val_line_num_one_hot = tf.one_hot(val_df["line_num"].to_numpy(),depth = 15)
    test_line_num_one_hot = tf.one_hot(test_df["line_num"].to_numpy(),depth = 15)
    train_line_total_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(),depth = 20)
    val_line_total_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(),depth = 20)
    test_line_total_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(),depth = 20)

    training_data = tf.data.Dataset.from_tensor_slices((train_line_num_one_hot,train_line_total_one_hot,train_sentences,train_chars))
    training_labels = tf.data.Dataset.from_tensor_slices(train_labels)
    training_dataset = tf.data.Dataset.zip((training_data,training_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

    testing_data = tf.data.Dataset.from_tensor_slices((test_line_num_one_hot,test_line_total_one_hot,test_sentences,test_chars))
    testing_labels = tf.data.Dataset.from_tensor_slices(test_labels)
    testing_dataset = tf.data.Dataset.zip((testing_data,testing_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

    validation_data = tf.data.Dataset.from_tensor_slices((val_line_num_one_hot,val_line_total_one_hot,val_sentences,val_chars))
    validation_labels = tf.data.Dataset.from_tensor_slices(val_labels)
    validation_dataset = tf.data.Dataset.zip((validation_data,validation_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

    return training_dataset,testing_dataset,validation_dataset


def split_chars(text):
    return " ".join(list(text))