"""SYDE 572 Assignment 4: optional tasks. Run with the syde572 conda env."""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from nb import DATA, train_nb, classify_documents, accuracy, score_doc_label

np.random.seed(0)


def read_full(path):
    docs, pol, topic = [], [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            w = line.strip().split()
            topic.append(w[0]); pol.append(w[1]); docs.append(w[3:])
    return docs, pol, topic


docs, pol, topic = read_full(DATA)
sp = int(0.8 * len(docs))
tr_d, ev_d = docs[:sp], docs[sp:]
tr_p, ev_p = pol[:sp], pol[sp:]
tr_t, ev_t = topic[:sp], topic[sp:]


def nb_acc(trd, trl, evd, evl, alpha=1.0):
    m = train_nb(trd, trl, alpha)
    return accuracy(evl, classify_documents(evd, m))


# ---- A. six-category topic classifier
print('== A. Topic classifier')
m = train_nb(tr_d, tr_t)
pt = classify_documents(ev_d, m)
print('topic accuracy', round(accuracy(ev_t, pt), 4))
cats = sorted(set(topic))
cm = pd.crosstab(pd.Series(ev_t, name='true'), pd.Series(pt, name='pred')).reindex(index=cats, columns=cats, fill_value=0)
print(cm)

# ---- B. learning curve
print('== B. Learning curve')
fracs = np.arange(0.1, 1.01, 0.1)
rng = np.random.default_rng(0)
curve = []
for fr in fracs:
    n = int(fr * len(tr_d))
    accs = []
    for _ in range(5):  # 5 random subsets per size
        idx = rng.choice(len(tr_d), n, replace=False)
        accs.append(nb_acc([tr_d[i] for i in idx], [tr_p[i] for i in idx], ev_d, ev_p))
    curve.append((np.mean(accs), np.std(accs)))
    print(f'{fr:.0%}: {curve[-1][0]:.4f} +/- {curve[-1][1]:.4f}')
mu, sd = np.array(curve).T
plt.figure(figsize=(5.5, 3.6))
plt.errorbar(fracs * 100, mu, yerr=sd, marker='o', capsize=3)
plt.xlabel('Training set size (% of training data)'); plt.ylabel('Test accuracy')
plt.title('Learning curve (mean ± std, 5 subsets)'); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig('output/learning_curve.png', dpi=160); plt.close()

# ---- C. informative features
print('== C. Informative features')
mod = train_nb(tr_d, tr_p, 50.0)  # high smoothing, as suggested


def feature_scores(model, pos='pos', neg='neg', min_count=0):
    a, V = model['alpha'], model['V']
    out = {}
    vocab = set(model['counts'][pos]) | set(model['counts'][neg])
    for w in vocab:
        if model['counts'][pos][w] + model['counts'][neg][w] < min_count: continue
        lp = math.log((model['counts'][pos][w] + a) / (model['totals'][pos] + a * (V + 1)))
        ln = math.log((model['counts'][neg][w] + a) / (model['totals'][neg] + a * (V + 1)))
        out[w] = lp - ln
    return out


def print_top_nb(k, model):
    s = feature_scores(model)
    r = sorted(s, key=s.get)
    print('top positive:', ', '.join(reversed(r[-k:])))
    print('top negative:', ', '.join(r[:k]))
    return s


s = print_top_nb(20, mod)
r = sorted(s, key=s.get)
top = r[:15] + r[-15:]
fig, ax = plt.subplots(figsize=(6, 6))
ax.barh(range(len(top)), [s[w] for w in top], color=['#c0392b'] * 15 + ['#2471a3'] * 15)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top)
ax.set_xlabel('log P(w|pos) − log P(w|neg)  (α = 50)'); ax.set_title('Most informative words')
plt.tight_layout(); plt.savefig('output/informative_words.png', dpi=160); plt.close()
# scatter: size ~ frequency, x = polarity score
cnt = Counter(w for d in tr_d for w in d)
sel = [w for w in r[:40] + r[-40:]]
fig, ax = plt.subplots(figsize=(8, 4.5))
rs = np.random.default_rng(1)
ax.scatter([s[w] for w in sel], rs.random(len(sel)), s=[cnt[w] / 6 for w in sel],
           c=[s[w] for w in sel], cmap='coolwarm_r', alpha=.5)
for w in sel: ax.annotate(w, (s[w], rs.random() if False else 0), alpha=0)
for w, y in zip(sel, rs.random(len(sel))):
    ax.text(s[w], y, w, fontsize=7, ha='center', va='center')
ax.set_yticks([]); ax.set_xlabel('polarity score (neg ← → pos)')
ax.set_title('Top polar words (marker size ∝ frequency)')
plt.tight_layout(); plt.savefig('output/word_scatter.png', dpi=160); plt.close()
print('by topic (positive words, alpha=50):')
for t in cats:
    mt = train_nb([d for d, tt in zip(tr_d, tr_t) if tt == t], [p for p, tt in zip(tr_p, tr_t) if tt == t], 20.0)
    st = feature_scores(mt, min_count=5); rt = sorted(st, key=st.get)
    print(f'  {t}: pos {", ".join(reversed(rt[-6:]))} | neg {", ".join(rt[:6])}')

# ---- D. Gaussian NB on iris
print('== D. Gaussian NB (iris)')
iris = pd.read_csv('../assignment3_syde572/data/iris.data', header=None,
                   names=['sl', 'sw', 'pl', 'pw', 'species'])
X, Y = iris.iloc[:, :4].to_numpy(), iris['species'].to_numpy()
n = len(Y)
ev = np.zeros(n, bool); ev[np.random.choice(n, 40, replace=False)] = True
Xtr, Ytr, Xev, Yev = X[~ev], Y[~ev], X[ev], Y[ev]


def train_gnb(X, y):
    return {c: (X[y == c].mean(0), X[y == c].std(0), math.log((y == c).mean())) for c in np.unique(y)}


def score_gnb(x, c, mdl):
    mu, sg, lp = mdl[c]
    return lp + np.sum(-0.5 * np.log(2 * np.pi * sg ** 2) - (x - mu) ** 2 / (2 * sg ** 2))


g = train_gnb(Xtr, Ytr)
pg = np.array([max(g, key=lambda c: score_gnb(x, c, g)) for x in Xev])
print('own GaussianNB acc', (pg == Yev).mean())
from sklearn.naive_bayes import GaussianNB
print('sklearn GaussianNB acc', GaussianNB().fit(Xtr, Ytr).score(Xev, Yev))

# ---- E. scikit-learn classifiers
print('== E. scikit-learn')
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest


def train_sklearn(d, l, clf, k='all'):
    return make_pipeline(TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, token_pattern=None),
                         SelectKBest(k=k), clf).fit(d, l)


svm = train_sklearn(tr_d, tr_p, LinearSVC())
print('LinearSVC acc', round(accuracy(ev_p, svm.predict(ev_d)), 4))
mlp = train_sklearn(tr_d, tr_p, MLPClassifier(max_iter=50, random_state=0), k=1000)
print('MLP (k=1000) acc', round(accuracy(ev_p, mlp.predict(ev_d)), 4))

# ---- F. improved features
print('== F. Improved features')
NEG = {'not', "n't", 'no', 'never', 'cannot', 'nothing', 'neither', 'nor'}
PUNCT = {'.', ',', '!', '?', ';', ':'}


def negate(doc):
    out, flag = [], False
    for w in doc:
        if w in PUNCT: flag = False; out.append(w); continue
        out.append('NOT_' + w if flag else w)
        if w in NEG: flag = True
    return out


def bigrams(doc): return doc + [a + '_' + b for a, b in zip(doc, doc[1:])]


feats = {'baseline': lambda d: d, 'negation': negate, 'bigrams': bigrams,
         'negation+bigrams': lambda d: bigrams(negate(d))}
for name, f in feats.items():
    a = max(nb_acc([f(d) for d in tr_d], tr_p, [f(d) for d in ev_d], ev_p, al) for al in (0.5, 1.0))
    print(f'{name}: {a:.4f} (best alpha in {{0.5,1}})')
    for al in (0.5, 1.0):
        print(f'   alpha={al}: {nb_acc([f(d) for d in tr_d], tr_p, [f(d) for d in ev_d], ev_p, al):.4f}')

# ---- G. domain sensitivity
print('== G. Domain sensitivity')


def domain(t):
    ix = [i for i, x in enumerate(topic) if x == t]
    k = int(.8 * len(ix))
    return ([docs[i] for i in ix[:k]], [pol[i] for i in ix[:k]], [docs[i] for i in ix[k:]], [pol[i] for i in ix[k:]])


D = {t: domain(t) for t in cats}
M = pd.DataFrame({te: {tr: nb_acc(D[tr][0], D[tr][1], D[te][2], D[te][3]) for tr in cats} for te in cats})
M.index.name = 'train \\ test'
print(M.round(3))
print('camera->books', round(M.loc['camera', 'books'], 4), 'books->books', round(M.loc['books', 'books'], 4))
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(M.values, cmap='viridis', vmin=.5, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(cats, rotation=45); ax.set_yticks(range(6)); ax.set_yticklabels(cats)
for i in range(6):
    for j in range(6): ax.text(j, i, f'{M.values[i, j]:.2f}', ha='center', va='center', color='w', fontsize=8)
ax.set_xlabel('test topic'); ax.set_ylabel('train topic'); plt.colorbar(im); plt.tight_layout()
plt.savefig('output/domain_matrix.png', dpi=160); plt.close()
