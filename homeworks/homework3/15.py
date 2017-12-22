import nltk
import numpy as np
from time import time
from nltk.tree import Tree, ProbabilisticTree as PTree


def parse_treebank(treebank_file):
    with open(treebank_file, 'r') as f:
        s = ''
        for l in f:
            if len(l) > 1:
                s += l
            elif s:
                try:
                    t = Tree.fromstring(s)
                except ValueError:
                    continue
                finally:
                    s = ''
                yield t
            else:
                s = ''


def is_tree_valid(tree):
    if not isinstance(tree, Tree):
        return False
    if tree.label() != 'S':
        return False
    if len(tree) < 2:
        return False
    return True


def train_grammar(trees):
    prods = []
    for tree in trees: 
        tree.collapse_unary(collapsePOS=True)
        tree.chomsky_normal_form()
        prods += tree.productions()
    return nltk.grammar.induce_pcfg(nltk.Nonterminal('S'), prods)


class PCKYParser(object):
    def __init__(self, grammar):
        if not isinstance(grammar, nltk.CFG):
            raise TypeError("g")
        # prepare lexicals and productions for parsing
        lexicals, productions = {}, {}
        for prod in grammar.productions():
            if len(prod.rhs()) == 1:  # lexical rule
                lexicals.setdefault(
                    prod.rhs()[0], []).append((prod.lhs(), prod.prob()))
            elif len(prod.rhs()) == 2:  # production rule
                l, r = prod.rhs()
                productions.setdefault(
                    l, {}).setdefault(
                    r, []).append((prod.lhs(), prod.prob()))
            else:
                print('non-CNF rule found:\n\t', prod)
        self.productions = productions
        self.lexicals = lexicals
        
    def parse(self, sent, n=1):
        k = len(sent)
        # init
        cky = np.empty((k,k), dtype=object)
        for i in range(k):
            for j in range(k):
                cky[i, j] = []
        # lexical rules
        for i, word in enumerate(sent):
            for l in self.lexicals.get(word, []):
                cky[i, i].append((l[0], l[1], word))
                # cky[i, i].append(PTree(prod.lhs(), [word], prob=prod.prob()))
        # production rules
        for col in range(1, k):
            for row in range(col-1, -1, -1):
                ways_to_split = col - row  # "distance" from diag
                for w in range(ways_to_split):
                    left = ways_to_split - w
                    down = 1 + w
                    for i, lt in enumerate(cky[row, col-left]):  # left tree
                        for j, dt in enumerate(cky[row+down, col]):  # right tree
                            try:
                                rules = self.productions[lt[0]][dt[0]]
                            except KeyError:
                                continue
                            for r in rules:
                                # store only the location of children in cky, build tree later
                                cky[row, col].append(
                                    (r[0], 
                                     lt[1] * dt[1] * r[1],
                                     (row, col-left, i, row+down, col, j)))
                                # cky[row, col].append(PTree(r['lhs'] , [lt, dt], prob=p))
        # collect sentences from parsed trees, sort them by probability
        sents = sorted(
            [tree for tree in cky[0, k-1] if tree[0] == nltk.Nonterminal('S')],
            # [tree for tree in cky[0, k-1] if tree.label() == nltk.Nonterminal('S')],
            key=lambda tree: tree[1]
        )
        # build and yield the most probable n trees
        def build(tree):
            lab = tree[0]  # label
            p = tree[1]  # probability
            c = tree[2]  # children
            if type(c) is tuple:
                # magic
                ch = [build(cky[c[0], c[1]][c[2]]), build(cky[c[3], c[4]][c[5]])]
            elif type(c) is str:
                ch = [c]
            return PTree(lab, ch, prob=p)
        for sent in sents[:n]:
            yield build(sent)


def constituents(tree):
    consts = []
    if isinstance(tree, Tree) and isinstance(tree[0], Tree):
        consts.append(' '.join(tree.leaves()))
        for subtree in tree:
            c = constituents(subtree)
            if type(c) is str:
                consts.append(c)
            else:
                consts += c
    else:
        consts = tree[0]
    return consts


def parseval_tree(gold, cand):
    gold_consts = constituents(gold)
    cand_consts = constituents(cand)
    correct_cand = [c for c in cand_consts if c in gold_consts]
    correct_gold = [c for c in gold_consts if c in cand_consts]
    precision = len(correct_cand) / len(cand_consts)
    recall = len(correct_gold) / len(gold_consts)
    F = 2 * (precision * recall) / (precision + recall)
    return precision, recall, F


def parseval(gold_trees, parser):
    ps, rs, Fs = [], [], []
    # get values for each sentence
    for i, gold_tree in enumerate(gold_trees):
        sentence = gold_tree.leaves()
        if len(sentence) > 15: continue
        try:
            print('parsing {} of 805, length {}...'.format(i+1, len(sentence)))
            print(' '.join(sentence))
            t = time()
            cand_tree = next(parser.parse(sentence, n=1))
            print('\tparsed in {} s'.format(time()-t))
            p, r, F = parseval_tree(gold_tree, cand_tree)
        except StopIteration:
            p, r, F = 0.0, 0.0, 0.0
        ps.append(p)
        rs.append(r)
        Fs.append(F)
    # return averages
    return [sum(v) / len(v) for v in [ps, rs, Fs]]


if __name__ == '__main__':
    grammar = train_grammar(filter(is_tree_valid, parse_treebank('en_lines-ud-train.s')))
    pparser = PCKYParser(grammar)
    t = next(pparser.parse('The SQL Server must be running on the same computer as the Access project .'.split(), n=1))
    print(t)
