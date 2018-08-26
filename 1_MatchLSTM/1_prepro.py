import os
import json
import spacy
import random
import numpy as np
import tensorflow as tf

from collections import Counter
from tqdm import tqdm
from tensorflow.train import Example, Features, Feature, BytesList, Int64List

"""
将原始数据加工成tfrecord的形式
"""

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def find_word_spans(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def get_examples(filename, word_counter):
    samples = []
    eval_samples = {}

    with open(filename, 'r') as f:
        source = json.load(f)['data']
        for article in tqdm(source):
            for para in article['paragraphs']:
                context = para['context']
                context_tokens = word_tokenize(para['context'])
                context_word_spans = find_word_spans(context, context_tokens)

                for token in context_tokens:
                    word_counter[token] += len(para['qas'])

                context = para['context']
                for qa in para['qas']:
                    question = qa['question']
                    question_tokens = word_tokenize(question)

                    y1s, y2s = [], []
                    answer_texts = []
                    for ans in qa['answers']:
                        answer = ans['text']
                        answer_start = ans['answer_start']
                        answer_end = answer_start + len(answer)
                        answer_index = []
                        for idx, span in enumerate(context_word_spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_index.append(idx)
                        y1s.append(answer_index[0])
                        y2s.append(answer_index[-1])
                    samples.append({
                        "context_tokens": context_tokens,
                        "question_tokens": question_tokens,
                        "y1s": y1s, "y2s": y2s, "id": len(samples) + 1
                    })
                    eval_samples[str(len(samples))] = {
                        "context": context,
                        "spans": context_word_spans,
                        "answers": answer_texts,
                        "uuid": qa['id']
                    }
    random.shuffle(samples)
    print("{} questions in total".format(len(samples)))
    return samples, eval_samples


def get_embedding(emb_file, counter, vec_size=None):

    embedding_dict = {}

    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            array = line.split()
            word = ''.join(array[0:-vec_size])
            if word in counter:
                embedding_dict[word] = list(map(float, array[-vec_size:]))

    null = '--NULL--'
    oov = '--OOV--'  # out of vocabulary
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[null] = 0
    token2idx_dict[oov] = 1
    embedding_dict[null] = [0. for _ in range(vec_size)]
    embedding_dict[oov] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_feature(examples, word2idx_dict, para_limit, ques_limit, ans_limit, out_file):

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1  # OOV

    with tf.python_io.TFRecordWriter(out_file) as writer:
        for example in tqdm(examples):
            if len(example['context_tokens']) > para_limit or \
                    len(example['question_tokens']) > ques_limit or \
                    (example['y2s'][0] - example['y1s'][0]) > ans_limit:
                continue
            context_idxs = np.zeros([para_limit], dtype=np.int32)
            ques_idxs = np.zeros([ques_limit], dtype=np.int32)
            y1 = np.zeros([para_limit], dtype=np.float32)
            y2 = np.zeros([para_limit], dtype=np.float32)

            for i, token in enumerate(example["context_tokens"]):
                context_idxs[i] = _get_word(token)
            for i, token in enumerate(example["question_tokens"]):
                ques_idxs[i] = _get_word(token)
            start, end = example['y1s'][-1], example['y2s'][-1]  # 只取一个正确答案
            y1[start], y2[end] = 1.0, 1.0

            record = Example(features=Features(feature={
                "context_idxs": Feature(bytes_list=BytesList(value=[context_idxs.tostring()])),
                "ques_idxs": Feature(bytes_list=BytesList(value=[ques_idxs.tostring()])),
                "y1": Feature(bytes_list=BytesList(value=[y1.tostring()])),
                "y2": Feature(bytes_list=BytesList(value=[y1.tostring()])),
                "id": Feature(int64_list=Int64List(value=[example['id']]))
            }))
            writer.write(record.SerializeToString())


if __name__ == '__main__':

    def save(filename, obj):
        with open(filename, 'w') as f:
            json.dump(obj, f)

    source_dir = '/home/limeng/data/'
    target_dir = 'data/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    train_file = source_dir + 'squad/train-v1.1.json'
    dev_file = source_dir + 'squad/dev-v1.1.json'
    train_eval_file = target_dir + 'train_eval.json'
    dev_eval_file = target_dir + 'dev_eval.json'

    word_counter = Counter()  # 好像只有word embedding
    train_examples, train_eval = get_examples(train_file, word_counter)
    dev_examples, dev_eval = get_examples(dev_file, word_counter)
    save(train_eval_file, train_eval)
    save(dev_eval_file, dev_eval)

    embedding_file = source_dir + '/glove/glove.840B.300d.txt'
    word_emb_save_file = target_dir + 'word_emb.json'
    word_dictionary = target_dir + 'word_dictionary.json'
    word_emb_dim = 300

    word_emb_mat, word2idx_dict = get_embedding(embedding_file, word_counter, word_emb_dim)
    save(word_emb_save_file, word_emb_mat)
    save(word_dictionary, word2idx_dict)

    para_limit = 400
    ques_limit = 50
    ans_limit = 30
    train_record_file = target_dir + 'train.tfrecords'
    dev_record_file = target_dir + 'dev.tfrecords'
    build_feature(train_examples, word2idx_dict, para_limit, ques_limit, ans_limit, train_record_file)
    build_feature(dev_examples, word2idx_dict, para_limit, ques_limit, ans_limit, dev_record_file)
