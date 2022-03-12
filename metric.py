import sys
from collections import Counter
import argparse
from nltk import ngrams
import re
import os
import nltk
import numpy as np
from stop_words import get_stop_words

from nltk.translate import bleu_score as nltkbleu

def bleu_corpus(hypothesis, references):
    from nltk.translate.bleu_score import corpus_bleu
    hypothesis = hypothesis.copy()
    references = references.copy()
    hypothesis = [hyp.split() for hyp in hypothesis]
    references = [[ref.split()] for ref in references]
    # hypothesis = [normalize_answer(hyp).split(" ") for hyp in hypothesis]
    # references = [[normalize_answer(ref).split(" ")] for ref in references]
    b1 = corpus_bleu(references, hypothesis, weights=(1.0/1.0,), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b2 = corpus_bleu(references, hypothesis, weights=(1.0/2.0, 1.0/2.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b3 = corpus_bleu(references, hypothesis, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b4 = corpus_bleu(references, hypothesis, weights=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    return (b1, b2, b3, b4)

def bleu_metric(hypothesis, references):
    return bleu_corpus(hypothesis, references)


def distinct_metric(hypothesis):
    '''
    compute distinct metric
    :param hypothesis: list of str
    :return:
    '''
    unigram_counter, bigram_counter = Counter(), Counter()
    for hypo in hypothesis:
        tokens = hypo.split()
        unigram_counter.update(tokens)
        bigram_counter.update(ngrams(tokens, 2))

    distinct_1 = len(unigram_counter) / sum(unigram_counter.values())
    distinct_2 = len(bigram_counter) / sum(bigram_counter.values())
    return distinct_1, distinct_2

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_answer_new(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))).split(' ')


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _f1_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for _, _, f1 in scores)

def _recall_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(recall for _, recall, _ in scores)

def _precision_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(precision for precision, _, _ in scores)


def f1_metric(hypothesis, references):
    '''
    calculate f1 metric
    :param hypothesis: list of str
    :param references: list of str
    :return:
    '''
    f1 = []
    for hyp, ref in zip(hypothesis, references):
        _f1 = _f1_score(hyp, [ref])
        f1.append(_f1)
    return np.mean(f1)


def recall_2at1(score_list, k=1):
    num_correct = 0
    num_total = len(score_list)
    for scores in score_list:
        ranking_index = np.argsort(-np.array(scores[0:2]))
        # Message at index 0 is always correct next message in our test data
        if 0 in ranking_index[:k]:
            num_correct += 1
    return float(num_correct) / num_total


def recall_at_k(score_list, k=1):
    num_correct = 0
    num_total = len(score_list)
    for scores in score_list:
        ranking_index = np.argsort(-np.array(scores))
        # Message at index 0 is always correct next message in our test data
        if 0 in ranking_index[:k]:
            num_correct += 1
    return float(num_correct) / num_total


def recall_at_k_new(labels, scores, k=1, doc_num=10):
    scores = scores.reshape(-1, doc_num)  # [batch, doc_num]
    labels = labels.reshape(-1, doc_num)  # # [batch, doc_num]
    sorted, indices = np.sort(scores, 1), np.argsort(-scores, 1)
    count_nonzero = 0
    recall = 0
    for i in range(indices.shape[0]):
        num_rel = np.sum(labels[i])
        if num_rel == 0: continue
        rel = 0
        for j in range(k):
            if labels[i, indices[i, j]] == 1:
                rel += 1
        recall += float(rel) / float(num_rel)
        count_nonzero += 1
    return float(recall) / count_nonzero


def precision_at_k(labels, scores, k=1, doc_num=10):
    scores = scores.reshape(-1, doc_num)  # [batch, doc_num]
    labels = labels.reshape(-1, doc_num)  # [batch, doc_num]

    sorted, indices = np.sort(scores, 1), np.argsort(-scores, 1)
    count_nonzero = 0
    precision = 0
    for i in range(indices.shape[0]):
        num_rel = np.sum(labels[i])
        if num_rel == 0: continue
        rel = 0
        for j in range(k):
            if labels[i, indices[i, j]] == 1:
                rel += 1
        precision += float(rel) / float(k)
        count_nonzero += 1
    return precision / count_nonzero


def MAP(target, logits, k=10):
    """
    Compute mean average precision.
    :param target: 2d array [batch_size x num_clicks_per_query] true
    :param logits: 2d array [batch_size x num_clicks_per_query] pred
    :return: mean average precision [a float value]
    """
    assert logits.shape == target.shape

    target = target.reshape(-1, k)
    logits = logits.reshape(-1, k)

    sorted, indices = np.sort(logits, 1)[::-1], np.argsort(-logits, 1)
    count_nonzero = 0
    map_sum = 0
    for i in range(indices.shape[0]):
        average_precision = 0
        num_rel = 0
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                num_rel += 1
                average_precision += float(num_rel) / (j + 1)
        if num_rel == 0: continue
        average_precision = average_precision / num_rel
        # print("average_precision: ", average_precision)
        map_sum += average_precision
        count_nonzero += 1
    # return map_sum / indices.shape[0]
    return float(map_sum) / count_nonzero


def MRR(target, logits, k=10):
    """
    Compute mean reciprocal rank.
    :param target: 2d array [batch_size x rel_docs_per_query]
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :return: mean reciprocal rank [a float value]
    """
    assert logits.shape == target.shape
    target = target.reshape(-1, k)
    logits = logits.reshape(-1, k)

    sorted, indices = np.sort(logits, 1)[::-1], np.argsort(-logits, 1)
    count_nonzero = 0
    reciprocal_rank = 0
    for i in range(indices.shape[0]):
        flag = 0
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                reciprocal_rank += float(1.0) / (j + 1)
                flag = 1
                break
        if flag: count_nonzero += 1

    # return reciprocal_rank / indices.shape[0]
    return float(reciprocal_rank) / count_nonzero


def NDCG(target, logits, k):
    """
    Compute normalized discounted cumulative gain.
    :param target: 2d array [batch_size x rel_docs_per_query]
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :return: mean average precision [a float value]
    """
    assert logits.shape == target.shape
    target = target.reshape(-1, k)
    logits = logits.reshape(-1, k)

    assert logits.shape[1] >= k, 'NDCG@K cannot be computed, invalid value of K.'

    sorted, indices = np.sort(logits, 1)[::-1], np.argsort(-logits, 1)
    NDCG = 0
    for i in range(indices.shape[0]):
        DCG_ref = 0
        num_rel_docs = np.count_nonzero(target[i])
        for j in range(indices.shape[1]):
            if j == k:
                break
            if target[i, indices[i, j]] == 1:
                DCG_ref += float(1.0) / np.log2(j + 2)
        DCG_gt = 0
        for j in range(num_rel_docs):
            if j == k:
                break
            DCG_gt += float(1.0) / np.log2(j + 2)
        NDCG += DCG_ref / DCG_gt

    return float(NDCG) / indices.shape[0]

def METEOR_metric(hypothesis, references):
    from nlgeval import NLGEval
    # from metrics import f1_metric, distinct_metric
    nlgeval = NLGEval(metrics_to_omit=[
        'CIDEr',
        'SkipThoughtCS',
        'EmbeddingAverageCosineSimilairty',
        'VectorExtremaCosineSimilarity',
        'GreedyMatchingScore',
        'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
        'ROUGE_L'
    ])
    metrics_dict = nlgeval.compute_metrics([references], hypothesis)
    return metrics_dict['METEOR']

def ROUGE_metric(hypothesis, references):
    from nlgeval import NLGEval
    # from metrics import f1_metric, distinct_metric
    nlgeval = NLGEval(metrics_to_omit=[
        'CIDEr',
        'SkipThoughtCS',
        'EmbeddingAverageCosineSimilairty',
        'VectorExtremaCosineSimilarity',
        'GreedyMatchingScore',
        'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
        'METEOR'
    ])
    metrics_dict = nlgeval.compute_metrics([references], hypothesis)
    return metrics_dict['ROUGE_L']



def rouge_metric(hypothesis, references, use_stemmer=True, average=True):
    from language_evaluation import rouge_scorer

    rouge_types = ["rouge1", "rouge2", "rougeL"]
    # rouge_types = ["rouge1", "rouge2"]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer, normalize_answer_new)
    scores = {rouge_type: [] for rouge_type in rouge_types}
    for predict, answer in zip(hypothesis, references):
        # TODO : support multi-reference
        score = scorer.score(answer, predict)
        for key, value in score.items():
            scores[key].append(value.fmeasure)

    # Averaging
    for key in scores.keys():
        if average:
            scores[key] = np.mean(np.array(scores[key]))
        else:
            scores[key] = np.array(scores[key])

    return scores['rouge1'], scores['rouge2'], scores['rougeL']


if __name__=='__main__':
    hyp_path='/home/futc/2021work2/dump_analysis/'+'11'+'/0step_result'
    ref_path='/home/futc/persona/ref'
    hyp_list=[]
    ref_list=[]
    f1=open(hyp_path,mode='r',encoding='utf-8')
    f2=open(ref_path,mode='r',encoding='utf-8')
    # for line1,line2 in zip(f1,f2):
    #     hyp_list.append(line1)
    #     ref_list.append(line2)
    for line in f1.readlines():
        hyp_list.append(line.strip('[unused0]').strip('[unused1]'))
    for line in f2.readlines():
        ref_list.append(line)
        
    ref_list=ref_list[:len(hyp_list)]
    print(hyp_path)
    #print(len(ref_list))
    print(bleu_metric(hyp_list,ref_list))
    print(rouge_metric(hyp_list,ref_list))
    print(distinct_metric(hyp_list))
    print(METEOR_metric(hyp_list,ref_list))
    #print(f1_metric(hyp_list,ref_list))
    
    
    
