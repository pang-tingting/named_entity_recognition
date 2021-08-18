from os.path import join
from codecs import open
import os
import pickle as pkl

word_vocab_path = "./datas/word_vocab.pkl"   #句子中包含的词的字典
tag_vocab_path = "./datas/tag_vocab.pkl"    #标签字典

def build_corpus(split, make_vocab=True, data_dir="./datas"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+"_data"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line.count('\n') != len(line):
                # print(line)
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        if os.path.exists(word_vocab_path):  # 提前定义好了词典，直接拿
            word2id = pkl.load(open(word_vocab_path, 'rb'))
        else:  # 根据训练集提取词典
            word2id = build_map(word_lists)
            pkl.dump(word2id, open(word_vocab_path, 'wb'))
        if os.path.exists(tag_vocab_path):  # 提前定义好了词典，直接拿
            tag2id = pkl.load(open(tag_vocab_path, 'rb'))
        else:  # 根据训练集提取词典
            tag2id = build_map(tag_lists)
            pkl.dump(tag2id, open(tag_vocab_path, 'wb'))
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps
