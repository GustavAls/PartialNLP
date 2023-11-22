import os.path

import pandas as pd
import pickle
from datasets import load_dataset
from datasets import Dataset as HuggingFaceDataset
import numpy as np
def read_sentences(path):
    dataframe = pd.read_csv(path, sep = '\t')
    sentence_index_to_sentence = dict(zip(dataframe['sentence_index'], dataframe['sentence']))
    return sentence_index_to_sentence

def read_labels(path):
    dataframe = pd.read_csv(path, sep='|')
    phrase_id_to_label = dict(zip(dataframe['phrase ids'], dataframe['sentiment values']))
    return phrase_id_to_label
def read_phrases(path):
    dataframe = pd.read_csv(path, sep='|', header=None)
    phrase_to_phrase_id = dict(zip(dataframe[0], dataframe[1]))
    return phrase_to_phrase_id

def read_splits(path):
    dataframe = pd.read_csv(path, sep=',')
    sentence_index_to_split =dict(zip(dataframe['sentence_index'], dataframe['splitset_label']))
    return sentence_index_to_split

def float_label_to_class(float_label):
    if float_label <= 0.2:
        return 0
    elif 0.2 < float_label <=0.4:
        return 1
    elif 0.4 < float_label <= 0.6:
        return 2
    elif 0.6 < float_label <= 0.8:
        return 3
    elif 0.8 < float_label <= 1.0:
        return 4


def make_dataset():

    phrases_path = r"C:\Users\45292\Downloads\stanfordSentimentTreebank\stanfordSentimentTreebank\dictionary.txt"
    label_path = r"C:\Users\45292\Downloads\stanfordSentimentTreebank\stanfordSentimentTreebank\sentiment_labels.txt"
    split_path =  r"C:\Users\45292\Downloads\stanfordSentimentTreebank\stanfordSentimentTreebank\datasetSplit.txt"
    sentences_path = r"C:\Users\45292\Downloads\stanfordSentimentTreebank\stanfordSentimentTreebank\datasetSentences.txt"
    output_path = r'C:\Users\45292\Documents\Master\SentimentClassification\SST2\Runs'
    sentence_index_to_sentence = read_sentences(sentences_path)
    phrase_id_to_label = read_labels(label_path)
    phrase_to_phrase_id = read_phrases(phrases_path)
    sentence_index_to_split = read_splits(split_path)

    train_val_sentences = [index for index, split in sentence_index_to_split.items() if split != 2]
    test_sentences = [index for index, split in sentence_index_to_split.items() if split ==2]
    num_samples = len(train_val_sentences)
    for run in range(5):
        out_path = os.path.join(output_path, f"run_{run}")
        os.mkdir(out_path)
        num_val = num_samples // 5
        valset = list(np.random.choice(train_val_sentences, size = (num_val, )))
        trainset = list(set(train_val_sentences) - set(valset))

        labels, sentences = [], []
        dataframe = pd.DataFrame()
        for index in trainset:
            sentence = sentence_index_to_sentence[index]
            phrase_id = phrase_to_phrase_id.get(sentence, None)
            if phrase_id is not None:
                label = phrase_id_to_label[phrase_id]
                labels.append(float_label_to_class(label))
                sentences.append(sentence)

        dataframe['sentences'] = sentences
        dataframe['labels'] = labels

        dataframe.to_csv(os.path.join(out_path, 'train.csv'))
        labels, sentences = [], []
        dataframe = pd.DataFrame()
        for index in valset:
            sentence = sentence_index_to_sentence[index]
            phrase_id = phrase_to_phrase_id.get(sentence, None)
            if phrase_id is not None:
                label = phrase_id_to_label[phrase_id]
                labels.append(float_label_to_class(label))
                sentences.append(sentence)

        dataframe['sentences'] = sentences
        dataframe['labels'] = labels
        dataframe.to_csv(os.path.join(out_path, 'val.csv'))

        labels, sentences = [], []
        dataframe = pd.DataFrame()
        for index in test_sentences:
            sentence = sentence_index_to_sentence[index]
            phrase_id = phrase_to_phrase_id.get(sentence, None)
            if phrase_id is not None:
                label = phrase_id_to_label[phrase_id]
                labels.append(float_label_to_class(label))
                sentences.append(sentence)

        dataframe['sentences'] = sentences
        dataframe['labels'] = labels
        dataframe.to_csv(os.path.join(out_path, 'test.csv'))


if __name__ == '__main__':
    make_dataset()

#
# import os
# import sys
# import shutil
# import argparse
# import tempfile
# import urllib.request
# import zipfile
# from argparse import Namespace
# TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
# TASK2PATH = {"CoLA": 'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
#              "SST": 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
#              "QQP": 'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
#              "STS": 'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
#              "MNLI": 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
#              "QNLI": 'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
#              "RTE": 'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
#              "WNLI": 'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
#              "diagnostic": 'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'}
#
# MRPC_TRAIN = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
# MRPC_TEST = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'
#
#
# def download_and_extract(task, data_dir):
#     print("Downloading and extracting %s..." % task)
#     if task == "MNLI":
#         print(
#             "\tNote (12/10/20): This script no longer downloads SNLI. You will need to manually download and format the data to use SNLI.")
#     data_file = "%s.zip" % task
#     urllib.request.urlretrieve(TASK2PATH[task], data_file)
#     with zipfile.ZipFile(data_file) as zip_ref:
#         zip_ref.extractall(data_dir)
#     os.remove(data_file)
#     print("\tCompleted!")
#
#
# def format_mrpc(data_dir, path_to_data):
#     print("Processing MRPC...")
#     mrpc_dir = os.path.join(data_dir, "MRPC")
#     if not os.path.isdir(mrpc_dir):
#         os.mkdir(mrpc_dir)
#     if path_to_data:
#         mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
#         mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
#     else:
#         try:
#             mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
#             mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
#             URLLIB.urlretrieve(MRPC_TRAIN, mrpc_train_file)
#             URLLIB.urlretrieve(MRPC_TEST, mrpc_test_file)
#         except urllib.error.HTTPError:
#             print("Error downloading MRPC")
#             return
#     assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
#     assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file
#
#     with io.open(mrpc_test_file, encoding='utf-8') as data_fh, \
#             io.open(os.path.join(mrpc_dir, "test.tsv"), 'w', encoding='utf-8') as test_fh:
#         header = data_fh.readline()
#         test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
#         for idx, row in enumerate(data_fh):
#             label, id1, id2, s1, s2 = row.strip().split('\t')
#             test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))
#
#     try:
#         URLLIB.urlretrieve(TASK2PATH["MRPC"], os.path.join(mrpc_dir, "dev_ids.tsv"))
#     except KeyError or urllib.error.HTTPError:
#         print("\tError downloading standard development IDs for MRPC. You will need to manually split your data.")
#         return
#
#     dev_ids = []
#     with io.open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding='utf-8') as ids_fh:
#         for row in ids_fh:
#             dev_ids.append(row.strip().split('\t'))
#
#     with io.open(mrpc_train_file, encoding='utf-8') as data_fh, \
#             io.open(os.path.join(mrpc_dir, "train.tsv"), 'w', encoding='utf-8') as train_fh, \
#             io.open(os.path.join(mrpc_dir, "dev.tsv"), 'w', encoding='utf-8') as dev_fh:
#         header = data_fh.readline()
#         train_fh.write(header)
#         dev_fh.write(header)
#         for row in data_fh:
#             label, id1, id2, s1, s2 = row.strip().split('\t')
#             if [id1, id2] in dev_ids:
#                 dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
#             else:
#                 train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
#
#     print("\tCompleted!")
#
#
# def download_diagnostic(data_dir):
#     print("Downloading and extracting diagnostic...")
#     if not os.path.isdir(os.path.join(data_dir, "diagnostic")):
#         os.mkdir(os.path.join(data_dir, "diagnostic"))
#     data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
#     urllib.request.urlretrieve(TASK2PATH["diagnostic"], data_file)
#     print("\tCompleted!")
#     return
#
#
# def get_tasks(task_names):
#     task_names = task_names.split(',')
#     if "all" in task_names:
#         tasks = TASKS
#     else:
#         tasks = []
#         for task_name in task_names:
#             assert task_name in TASKS, "Task %s not found!" % task_name
#             tasks.append(task_name)
#     return tasks
#
#
# def main(arguments):
#
#     args = {'data_dir': r'C:\Users\45292\Documents\Master\SentimentClassification\SST2',
#             'tasks': 'SST'}
#     args = Namespace(**args)
#     if not os.path.isdir(args.data_dir):
#         os.mkdir(args.data_dir)
#     tasks = get_tasks(args.tasks)
#
#     for task in tasks:
#         if task == 'MRPC':
#             format_mrpc(args.data_dir, args.path_to_mrpc)
#         elif task == 'diagnostic':
#             download_diagnostic(args.data_dir)
#         else:
#             download_and_extract(task, args.data_dir)
#
#
# if __name__ == '__main__':
#     main(2)
#
