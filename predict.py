from utils import load_model, get_entity, extend_maps
from data import build_corpus
import  argparse
import pickle as pkl
import numpy as np

#选择模型
parser = argparse.ArgumentParser(description='Named Enitity Recognition')
parser.add_argument('--model', type=str, required=True, help='choose a model: hmm,crf,bilstm,bilstm_crf')
args = parser.parse_args()

word_vocab_path = "./datas/word_vocab.pkl"   #句子中包含的词的字典
tag_vocab_path = "./datas/tag_vocab.pkl"    #标签字典

def main():
    word2id = pkl.load(open(word_vocab_path,'rb'))
    tag2id = pkl.load(open(tag_vocab_path,'rb'))
    str = input("请输入要识别的句子：")
    if str == '' or str.isspace():   #输入为空
        print('See you next time!')
    else:
        model = load_model("./ckpts/"+args.model+".pkl")  #加载模型
        word_list = [one for one in str]        #将句子变成一个字一个字的字典
        if args.model == "hmm":
            tag = model.decoding(word_list,word2id,tag2id)   #预测标签
        elif args.model == "crf":
            tag = model.predict_one(word_list)
        elif args.model == "bilstm":
            tag_list = word_list
            bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
            lstm_pre, target_tag_list = model.test(word_list, tag_list,
                                              bilstm_word2id, bilstm_tag2id)
            tag = []
            for item in lstm_pre:
                tag.append(item[0])
            #预测标签
        elif args.model == "bilstm_crf":
            print("1")
            # 预测标签
        print("\n预测标签: {}".format(tag))
        PER, LOC, ORG = get_entity(tag,word_list)   #整理标签为pre、loc、org的词语
        print('人名: {}\n地名: {}\n组织机构名: {}'.format(PER, LOC, ORG))     #输出

if __name__ == "__main__":
    main()