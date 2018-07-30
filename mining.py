from sudachipy import tokenizer
from sudachipy import dictionary
from sudachipy import config
import csv
import json
from sklearn.feature_extraction.text import CountVectorizer
import math
import os


class Respondent():
    def __init__(self,evaluation,type,reason,discription):
        self.evaluation = evaluation
        self.type = type
        self.reason = reason
        self.discription = discription
        self.decomposition = []
        self.wordlist = []
        self.TF = {}
        self.TFIDF = {}
        self.importance = 0

    def morphological_decomposition(self,tokenizer_obj,mode):
        self.decomposition = [m.surface() for m in tokenizer_obj.tokenize(mode,self.discription[1:])]

    def crop_lyrics(self,tokenizer_obj,mode,lyrics):
        tmp = []
        for text in self.decomposition:
            normalize = tokenizer_obj.tokenize(mode,text)[0].normalized_form()
            hinsi = tokenizer_obj.tokenize(mode,normalize)[0].part_of_speech()[0]
            if hinsi in lyrics:
                text = tokenizer_obj.tokenize(mode,normalize)[0].dictionary_form()
                tmp.append(text)
        self.wordlist += tmp

    def tf(self):
        uniq_tf = list(set(self.wordlist))
        num_tf = 0
        for word in uniq_tf:
            num_tf = self.wordlist.count(word)
            self.TF[word] = num_tf/len(self.wordlist)

    def tfidf(self,IDF):
        uniq_tfidf = list(set(self.wordlist))
        for word in uniq_tfidf:
            self.TFIDF[word] = self.TF[word] * IDF[word]

    def calc_importance(self):
        sum_tfidf = 0
        for v in self.TFIDF.values():
            sum_tfidf += v
        if self.wordlist:
            self.importance = sum_tfidf / len(self.wordlist)
        else:
            self.importance = -1


class Field():
    def __init__(self,year,month,region,field,evaluation,type,reason,discription):
        self.year = year
        self.month = month
        self.region = region
        self.field = field
        self.evaluation = evaluation
        self.type = type
        self.reason = reason
        self.discription = discription
        self.respondents = []
        self.decomposition = []
        self.wordlist = []
        self.all_wordlist = []
        self.bagofwords = []
        self.all_bagofwords = []
        self.IDF = {}
        self.TFIDF = {}
        self.summary = {}

    def append_respondent(self,respondent):
        self.respondents.append(respondent)

    def morphological_decomposition(self,tokenizer_obj,mode):
        self.decomposition = [m.surface() for m in tokenizer_obj.tokenize(mode,self.discription[1:])]

    def crop_lyrics(self,tokenizer_obj,mode,lyrics):
        for respondent in self.respondents:
            respondent.crop_lyrics(tokenizer_obj,mode,lyrics)
        tmp = []
        for text in self.decomposition:
            normalize = tokenizer_obj.tokenize(mode,text)[0].normalized_form()
            hinsi = tokenizer_obj.tokenize(mode,normalize)[0].part_of_speech()[0]
            if hinsi in lyrics:
                text = tokenizer_obj.tokenize(mode,normalize)[0].dictionary_form()
                tmp.append(text)
        self.wordlist += tmp

        for respondent in self.respondents:
            self.all_wordlist += respondent.wordlist
    def bagofword(self):
        count_vectorizer = CountVectorizer()
        if self.all_wordlist:
            feature_vectors = count_vectorizer.fit_transform(self.all_wordlist)
            print(feature_vectors.toarray())
            self.all_bagofwords = feature_vectors
        if self.wordlist:
            feature_vectors = count_vectorizer.fit_transform(self.wordlist)
            print(feature_vectors.toarray())
            self.bagofwords = feature_vectors


    def tf(self):
        for respondent in self.respondents:
            respondent.tf()

    def idf(self):
        uniq_df = list(set(self.all_wordlist))
        for word in uniq_df:
            num_df = 0
            for respondent in self.respondents:
                if word in respondent.wordlist:
                    num_df += 1
            self.IDF[word] = math.log2(len(self.respondents)/num_df) + 1


    def tfidf(self):
        self.tf()
        self.idf()
        for respondent in self.respondents:
            respondent.tfidf(self.IDF)
            #print(respondent.TFIDF)

    def calc_importance(self):
        for respondent in self.respondents:
            respondent.calc_importance()
            print(respondent.importance)

    def create_summary(self):
        nijumaru = {}
        maru = {}
        sikaku = {}
        sankaku = {}
        batu = {}

        for respondent in self.respondents:
            if respondent.evaluation == "◎":
                nijumaru[respondent.reason] = []

            if respondent.evaluation == "○":
                maru[respondent.reason] = []

            if respondent.evaluation == "□":
                sikaku[respondent.reason] = []

            if respondent.evaluation == "▲":
                sankaku[respondent.reason] = []

            if respondent.evaluation == "×":
                batu[respondent.reason] = []

        for respondent in self.respondents:
            if respondent.evaluation == "◎":
                nijumaru[respondent.reason].append(respondent)

            if respondent.evaluation == "○":
                maru[respondent.reason].append(respondent)

            if respondent.evaluation == "□":
                sikaku[respondent.reason].append(respondent)

            if respondent.evaluation == "▲":
                sankaku[respondent.reason].append(respondent)

            if respondent.evaluation == "×":
                batu[respondent.reason].append(respondent)

        if nijumaru:
            for k, v in nijumaru.items():
                #print(k,v)
                tmp = max(v, key = lambda x:x.importance)
                self.summary[tmp.evaluation + tmp.reason] = tmp.discription

        if maru:
            for k, v in maru.items():
                tmp = max(v, key = lambda x:x.importance)
                self.summary[tmp.evaluation + tmp.reason] = tmp.discription

        if sikaku:
            for k, v in sikaku.items():
                tmp = max(v, key = lambda x:x.importance)
                self.summary[tmp.evaluation + tmp.reason] = tmp.discription

        if sankaku:
            for k, v in sankaku.items():
                tmp = max(v, key = lambda x:x.importance)
                self.summary[tmp.evaluation + tmp.reason] = tmp.discription

        if batu:
            for k, v in batu.items():
                tmp = max(v, key = lambda x:x.importance)
                self.summary[tmp.evaluation + tmp.reason] = tmp.discription




with open(config.SETTINGFILE, "r",encoding="utf-8") as f:
    settings = json.load(f)
tokenizer_obj = dictionary.Dictionary(settings).create()


mode = tokenizer.Tokenizer.SplitMode.C
textlist = []
wordlist = []
fields = []
path = "景気ウォッチャー/"
listdir = os.listdir(path)
#include cvs data
for file in listdir:
    filename = path + file
    f = open(filename,'r')

    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if "関連" in row[0]:
            for i in range(len(row[0])):
                if row[0][i] == "(":
                    region = row[0][i:].strip(")")
            fields.append(Field(file[:2],file[2:].strip(".csv"),region,row[0],row[2],row[3],row[4],row[5]))
            fields[-1].append_respondent(Respondent(row[2],row[3],row[4],row[5]))
        if row[0] == "" and row[1] == "" and row[2] != "":
            fields[-1].append_respondent(Respondent(row[2],row[3],row[4],row[5]))
    #print(fields[0].respondents[0].discription)

for field in fields:
    field.morphological_decomposition(tokenizer_obj,mode)
    for respondent in field.respondents:
        respondent.morphological_decomposition(tokenizer_obj,mode)
        #textlist.append([m.surface() for m in tokenizer_obj.tokenize(mode,text[1:])])

#print(textlist)
filename = csv
for field in fields:
    field.crop_lyrics(tokenizer_obj,mode,["名詞","形容詞"])
    print(field.wordlist)
    field.tfidf()
    print(field.TFIDF)
    field.calc_importance()
    field.bagofword()
f = open('summary.csv','w')
writer = csv.writer(f)

for field in fields:
    field.create_summary()
    for k, v in field.summary.items():
        writer.writerow([field.year, field.month, field.region, field.field, k, v])

f.close()


"""
count_vectorizer = CountVectorizer()
feature_vectors = count_vectorizer.fit_transform(wordlist)
print(feature_vectors.toarray())
"""


#dct = gensim.corpora.Dictionary(wordlist.values())
