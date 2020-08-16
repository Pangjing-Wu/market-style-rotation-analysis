from senticnet.senticnet import SenticNet

sentiment_labels = ['polarity', 'pleasantness', 'attention',
                    'sensitivity', 'aptitude']

negations = set(['not', 'n\'t', 'less', 'no', 'never',
                 'nothing', 'nowhere', 'hardly', 'barely',
                 'scarcely', 'nobody', 'none'])


def sentiment_score(sentence:list):
    scores = [0 for _ in sentiment_labels]
    negative = False
    for word in sentence:
        if word in negations:
            negative = not negative
            continue
        try:
            sentics = SenticNet().sentics(word)
        # if the word not in SenticNet will raise KeyError.
        except KeyError:
            continue
        else:
            if negative:
                scores[0] -= eval(SenticNet().polarity_intense(word))
                for i in range(1,len(scores)):
                    scores[i] -= eval(sentics[sentiment_labels[i]])
            else:
                scores[0] += eval(SenticNet().polarity_intense(word))
                for i in range(1,len(scores)):
                    scores[i] += eval(sentics[sentiment_labels[i]])
    # calculate sentence average sentiment
    if len(sentence):
        scores = [score / len(sentence) for score in scores]
    return scores

print(sentiment_score(['surprise']))