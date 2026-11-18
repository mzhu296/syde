"""SYDE 572 Assignment 4: Naive Bayes sentiment classification (required part)."""
import math
from collections import Counter

DATA = 'SYDE572 assn4 file/all_sentiment_shuffled.txt'
ALPHA = 1.0  # Laplace/add-alpha smoothing


def read_documents(doc_file):
    docs, labels = [], []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels


def train_nb(documents, labels, alpha=ALPHA):
    """Returns model: log priors, per-class word counts, totals, vocab size."""
    n = len(labels)
    class_counts = Counter(labels)
    word_counts = {c: Counter() for c in class_counts}
    for doc, y in zip(documents, labels):
        word_counts[y].update(doc)
    vocab = set()
    for c in word_counts:
        vocab.update(word_counts[c])
    return {
        'classes': sorted(class_counts),
        'log_prior': {c: math.log(class_counts[c] / n) for c in class_counts},
        'counts': word_counts,
        'totals': {c: sum(word_counts[c].values()) for c in class_counts},
        'V': len(vocab),
        'alpha': alpha,
    }


def score_doc_label(document, label, model):
    """log P(label) + sum_w log P(w | label), add-alpha smoothed.
    Unseen words get alpha / (N_c + alpha*V) (never crashes)."""
    a, V = model['alpha'], model['V']
    cnt, tot = model['counts'][label], model['totals'][label]
    denom = tot + a * (V + 1)  # +1 slot for unknown words
    s = model['log_prior'][label]
    for w in document:
        s += math.log((cnt[w] + a) / denom)
    return s


def classify_nb(document, model):
    return max(model['classes'], key=lambda c: score_doc_label(document, c, model))


def classify_documents(docs, model):
    return [classify_nb(d, model) for d in docs]


def accuracy(true_labels, guessed_labels):
    return sum(t == g for t, g in zip(true_labels, guessed_labels)) / len(true_labels)


if __name__ == '__main__':
    all_docs, all_labels = read_documents(DATA)
    sp = int(0.8 * len(all_docs))
    train_docs, train_labels = all_docs[:sp], all_labels[:sp]
    eval_docs, eval_labels = all_docs[sp:], all_labels[sp:]
    print(f'train={len(train_docs)} eval={len(eval_docs)}', Counter(train_labels))

    model = train_nb(train_docs, train_labels)
    print('vocab size', model['V'])

    print('\nSanity check 1 (length-1 documents, probabilities):')
    for w in ['great', 'bad']:
        for c in model['classes']:
            print(f'  P([{w}], {c}) = {math.exp(score_doc_label([w], c, model)):.6f}')
    print('Sanity check 2:', {c: round(score_doc_label(['a', 'top-quality', 'performance'], c, model), 3)
                              for c in model['classes']})
    for d in (['great'], ['bad'], ['i', 'love', 'it'], ['waste', 'of', 'money']):
        print('  classify', d, '->', classify_nb(d, model))

    preds = classify_documents(eval_docs, model)
    print(f'\nTest accuracy: {accuracy(eval_labels, preds):.4f}')
    for a in (0.1, 0.5, 1.0, 2.0):
        m = train_nb(train_docs, train_labels, a)
        print(f'  alpha={a}: {accuracy(eval_labels, classify_documents(eval_docs, m)):.4f}')

    # Error analysis: wrong predictions with the largest confidence margin, short docs
    errs = []
    for d, y, p in zip(eval_docs, eval_labels, preds):
        if y != p:
            sc = {c: score_doc_label(d, c, model) for c in model['classes']}
            errs.append((sc[p] - sc[y], len(d), y, p, d))
    print(f'\n{len(errs)} errors. Most confident wrong, short docs (<=40 tokens):')
    for margin, n, y, p, d in sorted([e for e in errs if e[1] <= 40], reverse=True)[:6]:
        print(f'  margin={margin:.2f} len={n} true={y} pred={p}: {" ".join(d)}')
    print('Most confident wrong, any length:')
    for margin, n, y, p, d in sorted(errs, reverse=True)[:3]:
        print(f'  margin={margin:.2f} len={n} true={y} pred={p}: {" ".join(d[:60])} ...')
