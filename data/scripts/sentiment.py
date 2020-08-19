import abc
import argparse
import glob
import os
import re

import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from senticnet.senticnet import SenticNet


class SemtimentAnalysis(object):

    def __init__(self):
        self._negations = set(['not', 'n\'t', 'less', 'no', 'never',
                         'nothing', 'nowhere', 'hardly', 'barely',
                         'scarcely', 'nobody', 'none'])
        self._non_base  = set(['VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNS', 'NNPS'])
        self._stopwords = stopwords.words('english')

    @abc.abstractmethod
    def score(self):
        pass

    def _pos_short(self, pos):
        """Convert NLTK POS tags to SWN's POS tags."""
        if pos in set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            return 'v'
        elif pos in set(['JJ', 'JJR', 'JJS']):
            return 'a'
        elif pos in set(['RB', 'RBR', 'RBS']):
            return 'r'
        elif pos in set(['NNS', 'NN', 'NNP', 'NNPS']):
            return 'n'
        else:
            return 'a'

    def _preprocess(self, text:str):
        '''
        Shape:
        ------
        output: token size is m sentences * n words.
        '''
        if type(text) is not str:
            raise TypeError("argument text must be str.")
        if len(text) == 0:
            return []

        text   = text.lower()
        tokens = list()
        sentences = sent_tokenize(text)
        for sentence in sentences:
            sentence = re.sub(r'[^a-z｜\s|\'|0-9]', '', sentence)
            words = word_tokenize(sentence)
            words = [WordNetLemmatizer().lemmatize(w, pos=self._pos_short(p)) for w, p in pos_tag(words)]
            tokens.append(words)
        return tokens


class SenticNet5(SemtimentAnalysis):

    def __init__(self):
        super().__init__()
        self.sentiment_labels = ['polarity', 'pleasantness', 'attention',
                                 'sensitivity', 'aptitude']
        self._max_phrase_len = 4

    def score(self, text:str):
        scores = dict(polarity=0, pleasantness=0, attention=0,
                      sensitivity=0, aptitude=0)
        sentences = self._preprocess(text)
        for sentence in sentences:
            negative = False
            self._combine_phrase(sentence)
            for word in sentence:
                if word in self._negations:
                    negative = not negative
                    continue
                try:
                    sentics = SenticNet().sentics(word)
                except KeyError:
                    continue
                for sentiment in self.sentiment_labels:
                    if sentiment == 'polarity':
                        polar = float(SenticNet().polarity_intense(word)) / len(sentence)
                        scores[sentiment] -= polar if negative else -polar
                    else:
                        score = float(sentics[sentiment])  / len(sentence)
                        scores[sentiment] -= score if negative else -score
        return scores

    def _combine_phrase(self, words):
        i = 0
        while i < len(words):
            phrase, l = self._query_phrase(words[i:])
            if l > 1:
                words[i] = phrase
                words = words[:i+1] + words[i+l:]
            i += 1
        return words

    def _query_phrase(self, words:list):
        if len(words) == 0:
            return '', 0
        phrase, length = words[0], 1
        for i in range(1, min(len(words), self._max_phrase_len)):
            next_phrase = '_'.join(words[:i+1])
            try:
                SenticNet().polarity_intense(next_phrase)
            except KeyError:
                pass
            else:
                phrase, length = next_phrase, i+1
        return phrase, length


class LMFinance(SemtimentAnalysis):

    def __init__(self, filepath:str):
        super().__init__()
        self.sentiment_labels = ['positive', 'negative', 'uncertainty', 'litigious',
                                 'weakmodal', 'strongmodal', 'constraining']
        self._lexicon = dict()
        for sentiment in self.sentiment_labels:
            f = open(os.path.join(filepath, sentiment+'.txt'))
            self._lexicon[sentiment] = [w.strip('\n') for w in f.readlines()]
            f.close()

    def score(self, text:str):
        scores = dict(positive=0, negative=0, uncertainty=0, litigious=0,
                      weakmodal=0, strongmodal=0, constraining=0)
        sentences = self._preprocess(text)
        for sentence in sentences:
            negative = False
            for word in sentence:
                if word in self._negations:
                    negative = not negative
                    continue
                for sentiment in self.sentiment_labels:
                    if word in self._lexicon[sentiment]:
                        score = 1 / len(sentence)
                        scores[sentiment] -= score if negative else -score
        return scores


def parse_args():
    parser = argparse.ArgumentParser(description= 'batch calculate news sentiments')
    parser.add_argument('-i', '--data_dir', required=True, type=str, help='direction of data file')
    parser.add_argument('-o', '--save_dir', required=True, type=str, help='direction of output file')
    parser.add_argument('--lexicon', required=True, type=str, help='sentiment lexicion {SenticNet5|LMFinance}')
    parser.add_argument('--lexicon_dir', type=str, help='lexicion file direction')
    return parser.parse_args()


# python -u ./data/scripts/sentiment.py -i ./data/raw/news -o ./data/processed/sentiments/LMFinance --lexicon SenticNet5 --lexicon_dir ./data/lexicon/LMFinance
if __name__ == "__main__":
    params = parse_args()

    if os.path.isdir(params.data_dir):
        csvlist = glob.glob(os.path.join(params.data_dir, '*.csv'))
    elif os.path.isfile(params.data_dir):
        csvlist = [params.data_dir]
    else:
        raise KeyError('unknown data direction')

    os.makedirs(params.save_dir, exist_ok=True)

    print('load data from %s, save to %s.' % (params.data_dir, params.save_dir))

    if params.lexicon == 'SenticNet5':
        lexicon = SenticNet5()
    elif params.lexicon == 'LMFinance':
        lexicon = LMFinance(params.lexicon_dir)
    else:
        raise KeyError("unknown sentiment lexicon")

    for i, csvfile in enumerate(csvlist):
        news = pd.read_csv(csvfile)
        data = pd.DataFrame(columns=['date', *lexicon.sentiment_labels], index=news.index, dtype='float')
        data.date = news.date
        for index, content in zip(news.index, news.contents):
            score = lexicon.score(content)
            for sentiment in score:
                data.loc[index, sentiment] = score[sentiment] 
        data = data.dropna(axis=0)
        data = data.groupby('date').agg('sum')
        filename = os.path.basename(csvfile)
        data.to_csv(os.path.join(params.save_dir, filename), float_format='%.5f')
        print('[%d/%d] %s was preprocessed.' % (i+1, len(csvlist), filename))
        
    print('All files have been preprocessed.')
