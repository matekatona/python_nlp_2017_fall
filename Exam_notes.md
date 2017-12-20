# I. Python topics

## 1. Introduction

* What is Jupyter?
    * web application, documents with live code, visualizations etc.
    * JSON, `.ipynb`
    * convert to HTML, PDF, LateX etc.
    * render images, tables, graphs, LateX equations
    * organized into cells
* cell types, cell magic
    * types: code (Python, R, Lua, etc.), raw text, markdown
    * magic: special commands for cells: `%%command`, e.g. `%%timeit`
* kernel
    * unique per notebook
    * shared namespace for all cells
    * execution order arbitrary
* short history of Python
    * 1989 - Guido van Rossum (BDFL), hobby
    * 1994 - 1.0
    * 2000 - 2.0 (GC, unicode)
    * 2008 - 3.0 (backward incompatible)
    * 2020 - python2 (2.7) EOL pp from 2015
* Python community, PEP, Pythonista
    * Python Sofware Foundation, OS
    * PEPs, community, standard library, PyPI, pip

## 2. Functions and generators

* args, kwargs
    * first *pos, then **kw
    * with kw position is irrelevant
* default arguments
    * only after non-default, can be overriden pos or kw
* lambda functions
    * nameless, disposable functions: `labda args: retval`, e.g. for `sorted` keys
* generators, yield statement
    * mem

## 3. Type system

* static vs. dynamic typing
    * types checked in run-time (static in compile time)
    * variables are names, can be reassigned
* built-in types (numeric, boolean), operators
    * int, float, complex, True, False, NoneType, None, +=, and, not, or
    * implicit conversion onyl btwn numeric types: less data loss
    * /: float division
    * //: floor (integer) division
    * 1 < x < 2
    * % remainder (mod)
    * ** power
* mutability
    * lists are mutable, no new object
    * strings are immutable, always new object
    * onyl one None, True, False object


## 4. Sequence types
* list vs. tuple
    * mutable, not mutable
* operators
    * in, not in, +, *, [], index, count
* advanced indexing
    * [i] - ith
    * [i:j] - from ith to jth OPEN because size calc
    * [i:j:k] - from ith to jth in k-size steps
    * [-i] - ith from back
    * [j, i, -k] - blabla
  
* extra: time complexity of basic operations (lookup, insert, remove, append etc.)
    * list lookup O(n), set/dict lookup O(1)
* set operations
    * union, intersection, operator overloading for &, +, -, /, etc.
    * subset, superset

## 5. Strings

* immutable sequence of unicode codepoints
* character encodings: Unicode vs. UTF-8, encoding, decoding
    * Unicode: characters to code points (a -> U+0061)
    * UTF-8: code <-> byte array
    * variable length (ascii: 1 byte, others: 2 or more bytes)
* extra: Python 2 vs. Python 3 encodings
    * python2: `str` is bytestring, `unicode` is unicode, `encode`, `decode`
    * pyhton3 `str` is unicode, no `unicode`, `bytes` for bytestrings
* common string operations
    * lower, upper, title 
    * split, (r|l)strip
    * startswith, endswith, istitle, isupper, islower, isspace, isdigit
    * join
* string formatting (at least two solutions)
    * str.format(), mini-language
    * % op, tuple args
    * f-strings since 3.6, variable names

## 6. Object oriented programming I.

* data attributes, methods, class attirbutes
    * everything is an attribute
    * some attributes are regular data attributes, some are methods (callable attribute)
    * explicit instance argument -> self
    * run-time assignment possible
    * dunder attributes
    * underscore, two leading -> name mangling
    * class attributes, staticmethod, etc
* inheritance, `super`
    * methods are always inherited
    * data isn't -> call `__init__`
    * super builtin -> returns the base class
    * python2 super(class, instance)
* duck typing
    * no interfaces (`abc` module)
    * type checking in runtime -> TypeError
    * virtual function -> NotImplementedError
* magic methods, operator overloading
    * dunders
    * operator overloading
    * sequence, map, setgetattr

## 7. Object oriented programming II.
* assignment, shallow copy, deep copy
    * = only references to sam object
    * copy only first layer (lits of lists)
    * deepycopy recursicely everything
* object introspection
    * dir
* class decorators, static methods, class methods
    * decorator: function wrapper
    * static: no self, not bound to instance
    * class: cls not self, class instance, factory methods
* properties
    * setgetdel methods as attribute

## 8. List comprehension

* basic list comprehension (you should be able to write it on paper)
    * [expr for elem in sequence if condition]
    * if-else as first expression
* generator expressions
    * same with () -> not evaluated
    * not in memory, only once
* extra: iteration protocol, writing an iterator class
    * `__iter__` returns iterable(self)
    * `__next__` return something
    * StopIteration after some iterations
* set and dict comprehension
    * set: {unique for a in b}
    * sict: {key: word for c in d}
* yield keyword
    * generator function: execution back to the caller between iterations
    * memory efficiency again

## 9. Exception handling

* basic keywords
    * raise, try, except (as), else, finally
    * Exception, empty raise
* defining exception classes
    * inherit from Exception
    * try-error is pythonic (ducktyping)

## 10. Context managers

* basic usage
    * with something as sg: indent
    * manual resource management (open().close())
* defining context managers
    * `__enter__`, `__exit__`
    * exit(type, value, traceback)

## 11. Decorators

* functions are objects (callable), so can be passed and returned
    * `__call__` method
* nested functions, closure, access
* What are decorators?
    * funciton with function argument, returns wrapped funciton
* `@wraps` decorator
    * copies function metadata to wrapper
* decorators with parameters
    * 3 level deep, return decorator without parameters -> factory
* extra: classes as decorators
    * init takes the function
    * call calls the function and does the wrapping

## 12. Functional Python

* map, filter, reduce
    * generators
    * map applies function on sequence elements
    * filter only returns those for which function returns true
    * reduce inlcudes an accumulator, apllies 2arg func on element and accumulator

## 13. Packaging

* Why is it good to package your code?
    * NAMESPACES!!!!
    * `__init__.py`
    * setup.py
    * distribution
    * why organize code into functions/classes?
    * layers of abstraction

## 14. `numpy/scipy`
* main object (`ndarray`)
    * `shape`, `dtype`
        * tuple of dimension lengths, class
    * dimension (`len(shape)`)
    * elementwise operations (`exp`, `sin`, `+ - * /`, `pow` or `**`)
    * indexing
    * `reshape` -> inplace
* matrix operations: `dot`, `transpose`
* reductions: `sum` along a given axis
* broadcasting
    * the `None` index
    * row-wise normalization
        * elementwise prod, sum along axis, sqrt
    * one-liner for a complex grid
        * add two 1D vectors broadcasted in different 2nd dimensions
* some matrix constructions
    * `ones`, `zeros`, `eye`, `diag`
* scipy
    * sparse matrices
        * memory, speed
        * scipy.sparse
        * optimized for different sparsities
        * coo: storage
        * lil, dok: incremental
        * csr, csc: fast row/col operations
        * todense, toarray
    * solving linear equations, svd (sparse and dense)
        * linalg.solve(Mat, vec)
        * linalg.lstsq(Mat, vec) (overdetermined)
        * sparse.linalg.spsolve, diags, svds, etc.
        

# II. NLP topics

## 1. Tagging
* POS-tags: (__P__art-__O__f-__S__peech)
    * definition
    * Universal Tagset 
        * DET - determiner: a, the
        * NOUN - noun: dog, cat
        * ADJ - adjective: red, blue
        * VERB - verb: see, write
        * PRON - pronoun: me, us, 
* Tagging in general: definitions and terminology (from lecture notebook)
    * tokens: words for now
    * corpus: list of tokens, "sample text"
    * tokenization: splitting raw text into tokens
    * tagset: finite set of tag symbols, linguists
    * labeled corpus: (token, tag) pairs
    * tagging: assign tags to an unlabeled corpus
* statistical - no rules only previously correctly labeled data:
    * training: setting up the algorithm based on pyld
    * train set: correctly labeled data
    * prediction: tagging new text with the algorithm (inference)
    * test/unseen data: data with correct labels, but labels are not shown to the algorithm
    * evaluation: compare predicted tags with correct tags
    * annotating: manually labeling data by humans -> gold
    * gold data: seriously labeled data
    * silver data: lesser quality labeled data (might be automatic)
    * test on train: evaluate algorithm on train set
* NP-chunking: definition, examples
    * find __N__oun __P__hrases in sentences
    * single agent
    * NP is someone/something that
        * does something
        * the action is performed on
        * involved
    * shallow parsing: not the whole, just some parts
* NER: definition, examples
    * __N__amed __E__ntity __R__ecognition
    * names of things
    * persons, places, companies
* Naive ways
    * non-unique POS-tag:
        * depends on context, surroundings
        * many words with several tags: work, talk, walk
    * capital letters in NER
        * the United States, von Etwas
        * sentences start with capitals: Biking in the rain is fun.
    * no complete list exists
* Supervised learning, labeled data in general
    * train on data+label pairs, then predict and evaluate
    * object recognition in images, every other thing
    * problems with correctly labeled data:
        * quality <-> quantity
        * errors
* Sequential data: windows approach, problem with long-distance relationships
    * order of words is very important!
    * unordered window around WOI -> supervised learning
    * width is usually not enough because LONG-DISTANCE RELATIONSHIPS
* Simple POS-tagger: most seen pos tag in a given context of words
    * all words in V vocabulary
    * all tags in L labels
    * POS tag determined by preceding/following words (and their tags)
    * OOV -> not in train, 
    * data sparsity -> word not seen with these words

## 2. An HMM POS-tagger

* what is POS-tagging
    * above
* __H__idden __M__arkov __M__odel
    * the POS tag is a hidden parameter of the words
    * search for the tag sequence, that generates the words with the highest probability
    * restricitons:
        - window
    * assumption:
        - P(tag+word sequence) = P(current tag given the previous tags) * P(word given the tag)
    * estimation (TRAINING):
        - P(l) = #tagseq / #prevtagseq
        - P(w) = #wordtag / #tag
    * prediciton e.g. with viterbi
* The Viterbi algorithm (k=2,3)
    - k s the window
    - step funciton:
        + find the most probable ending tags
        + find the next tag using the previous most probable sequence
        + pi(k,prevtag,tag) = max on prevprevtag pi(k-1,prevprevtag,prevtag)*P(tagseq ppt,pt,t)\*P(wordtag)
        + start with only wordtag prob, then build up
        + then prev viterbi times wordtag times tagseq
        + the most probable last k-1 tags are always known, with probability
        + store the current most probable prevprevtag (from third iteration in k=3)
        + the last iteration will yiels the most probable last k-1 tags
        + the rest can be backtracked
* Backtracking
    - argmax

## 3. Evaluation

* labeled corpus
    - gold, silver, hoomans, etc
* train/test set, seen/unseen data
    - blabla
* word accuracy: #correctlabels / #words
* sentence accuracy: #sentencesswithalllabelscorrect / #sentences
* OOV: #OOVwithcorrectlabels / #OOVs
* Binary classification metrics (TP/TN/FP/FN table, precision/recall, Fscore)
    - True/False Positive/Negative
    - precision: TP/(TP+FP)
        + correct predictions from all predicitons
        + number of good predictions from all predictions
    - accuracy: (TP+TN)/all
        + correct moves from everything
    - recall: TP/(TP+FN)
        + correct predictions from all where should predict
        + number of good predictions from all events, where it should predict
    - Fscore: harmonic mean of prec and recall
        + 2* (prec*rec)/(prec+rec)

## 4. Morphology: theory

- Basic concepts: morphemes, tags, analysis and generation
    + morpheme: minimal meaning-bearing unit
    + tags: like in POS, but for morphemes
    + analysis: determine morphemes and their tags
    + generation: reproduce surface form from morpheme tags
- Why do we need morphology: tasks
    + spell checking
    + lemmatization: finding word stem
    + information retreival (what, where, how, with what)
    + first building block for syntax and semantics
- Phenomena:
    - free and bound morphems:
        + free is root/stem/lemma: fox
        + bound e.g. affxes: small__est__
    - affix types:
        + derivational: change POS (sad-ness, modern-ize) 
        + inflectional: syntactic function (talk-ed, dog-s)
    - affix placement:
        + pre
        + suf
        + circum
        + in
    - compounding, clitics
        + more than one stem: seahorse
        + don't, we'll
    - non-concatenative morphology: 
        + STEM MODIFICATION
        + reduplication: iddy-biddy
        + templates: for, far, fur
        + ablaut: foot, feet
    - language types
        - isolating: everything is separate - mandarin
        - analytic: some inflection, but word order - english
        - synthetic: much inflection, high morpheme-per-word
            + fusional: single suffix encode many gramatical stuff
            + agglunative: concatenative, slots
            + polysynthetic: sentence-words
- Computational morphology:
    + RULES
    - components: 
        - lexicon: list of morphemes with basic info like POS
        - morphotactics: ordering and interactions (basic)
        - orthography/phonetics: mapping from morphemes to sounds
    - usually with finite state transducers
    - deep learning not very applicable
    - analysis, generation
        + with FST both directions are posibble, but ambigous

## 5. Finite state morphology

- Finite state:
    - FSA - input, states, state changes 
    - FST - all of the above, plus output
    - mathematical description (5-tuple)
        - Q: the set of states
        - Sigma: the input alphabet
        - q0: the initial state
        - F: the accepting states
        - Gamma (FST): output alphabet
        - delta(q, w): the state transition function - Q x Sigma (x Gamma (FST)) x Q
        - ---------------------------
        - accepted symbol sequences: L language (L' output language), regular languages
        - memoryless, efficient, fast
    - Operations on FSTs: 
        - inversion: swap upper and lower tapes (not IO)
        - composition: FSTs are functions, funcitons composition
            - Morphology = Ortography(MorphoTactics(Lexicon(x)))
        - projection: one of the languages (are preserved)
- Morphology:
    - FSTs as morphological analyzers
        + lex, MT, Orto as FSTs
        + cascade or composition
        + lower side (in): surface form
        + upper side (out): morpheme struct (char + tag)
    - analysis / generation
        + inversion allows both
        + apply up (low->upp): analysis
        + apply down (upp->low): generation
        + ambiguity everywhere
    - backtracking
        + backtrack possible candidates from the accepting state(s)
        + list all possible candidates
    - lexc
        + XFST, foma, formalisms
        + MT, Orto as RE -> RULES
        + lexc is designed sor lexicons, but rules also
        + LEXICON
            + entrys are morphemes
                + characters, both sides
                + transductions-> upper:lower
            + and continuation classes -> other LEXICONs
            
- Related tasks:
    - spell checking:
        + lower projection: all words in language
    - lemmatization:
        + morphological anlysis gets stem, delete tags
    - tokenization:
        + circular analyzer??? lower projection???
    - how a morphological FST can be used for these tasks

## 6. Syntax: theory

- Basics: definition, grammaticality
    + morphology: rules that govern the structure of words
    + syntax: rules that govern the structure of __sentences__
    + grammatical, if obeys the rules (meaning not relevant)
- Concepts: 
    - POS: categories with similar grammatical stuff
        + open: CNN, lájkolni
        + closed: determiners, prepositions
    - grammatical relations (predicate, arguments, adjuncts)
        + predicate: center of sentence, main verb
        + arguments: of the predicate - subject, object (dir, indir)
        + adjuncts: optional info, not related to the predicate (where, how, etc.)
    - subcategorization (valency, transitivity)
        + limits on arguments
        + transitivity is only one
            * intrans: no obj
            * trans: direct subj
            * ditrans: dir and indir subj
    - constituency (tests)
        + phrase
        + group of words as unit, one grammatical role
            + NounPhrase: the tall man
            + PrepositionalPhrase: on the house's roof
            + AdjectivalPhrase: pretty fucking awesome
        - SUBSTITUTION
        - TOPICALIZATION: move to front
- Phrase-structure grammars (__C__ontext __F__ree __G__rammars):
    - Components: 
        + production rules
            * phrase -> other phrases
        + terminals
            * part of language: dog
        + nonterminals
            * pos tag or phrase: Noun, NounPhrase
    - Derivation
        + start with __S__tart symbol, then apply rules, until all terminals
        + the parse tree
            * root is the start symbol
            * nodes are nonterminals
            * edges are rules
            * leaves are terminals
        + Chomsky Normal Form
            * only two types of rules:
                - Nonterminal -> Nonterminal Nonterminal
                - Nonterminal -> temrinal
            * binary trees
            * for algorithms
- Chomsky Normal Form and its algorithm
    * rhs too long -> introduce new nonterminal
    * unit productions -> delete, pull up
- Dependency Grammar:
    - Difference from PSG
        + PSG 
            + no subjectobject
            + no relationships
            + english centric
        + DG 
            + direct relationship encoding
            + semantic view
            + free word order
    - The dependency tree
        + nodes are words
        + edges are grammatical relations
        + root is virtual, outside of sentence

## 7. Parsing and PCFG

- Parsing
    + tree from sentence
    - Top-down and bottom-up parsing
        + TD: start from S, rules until terminals (as generating), until whole sent
        + BU: backward rules, connect subgraphs with rules, until single S tree
    - Challenges: 
        + nondeterminism:
            + several rule expansions, only one is right: Verb -> ad | kap
        + ambiguity:
            + more than one correct parses
            + global: whole sentence
            + local: standalone constituent is, but not in the sentence
            + e.g.:
                * i saw him with glasses
                * i saw big apples and pears
                * work (Noun, Verb)
    - Dynamic programming:
        * smaller subproblems
        * solve only once, store value
    - CKY algorithm
        + only works on CNF
        * parse trees in matrix
        * subtrees reused

- PCFG
    - Motivation and mathematical formulation
        + multiple parses, decide which one is best
        + product of rule probs
    - Training and testing: treebanks and PARSEVAL
        + treebank is golden parse trees
        + normalized corpus freq
            * rule occurences / lhs occurences
        + PARSEVAL
            * word span
            * labeled, unlabeled
            * precision: correct constituents / parsed constituents
            * recall: correct constituents / actual contituents
            * F: 2(prec*rec)/(prec+rec)
    - Problems and solutions: 
        + subcategorization: NOT SUPPORTED
            * doesn't care about meaning, just number of occurences
            + lexicalization
                * annotate constituents -> more rules
                * NounPhrase after eat is ok (object)
                * NounPhrase after sleep is nok (no object)
                * introduces sparsity -> lot of unprobable rules
        + independence
            * wrong assumption of rule applicability
            * context sensitive!
            + annotation:
                * annotate constituents with grammatical features (singular, posessive)
                * propagate up
                * accept only if these match too

## 8. Classical machine translation

- Translation divergences: systematic, idiosynchretic and lexical divergences; examples
- The classical model
    - Translation steps and approaches; the Vauquois triangle
    - Direct translation, transfer, interlingua. Advantages and disadvantages of each
- Problems with the classical model

## 9. Statistical machine translation

- Mathematical description of statistical MT:
    - the noisy channel model
    - language and translation models; fidelity and fluency
- Training and decoding (parallel corpora, decoding as search)
- Evaluation:
    - Manual evaluation (what and how)
    - Overview of BLEU (modified precision, brevity penalty, micro average)

## 10. Alignment in machine translation

- Word alignment; examples and arity
- Phrase acquisition for phrase-based MT
- IBM Model 1: algorithm and mathematical formulation

These have to be cut up, but I haven't yet decided how.

## 11. Word meaning
- Representations: distributional and ontology-based approaches, pros and cons for each.
- Simple word relationships: synonymy, hipernymy.
- Common lexical resources: WordNet, FrameNet.
- Measuring word similarity: approaches and issues.

## 12. Relation Extraction, Sentiment Analysis

 - Relation extraction
   - task definition, versions, examples
   - rule-based approach, pros and cons
   - supervised learning approach, pros and cons
   - semi-supervised methods
   
 - Sentiment analysis
   - task definition, examples
   - supervised learning approach, common features
 
## 13. Question answering
  - task description, existing systems
  - major approaches and their limitations
  - The IR-based approach:
    - question processing
    - query generation
    - retrieval
    - answer ranking
