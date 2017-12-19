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
- Why do we need morphology: tasks
- Phenomena:
    - free and bound morphems, affix placement, affix types
    - compounding, clitics
    - non-concatenative morphology: reduplication, templates, ablaut
    - language types
- Computational morphology:
    - components: lexicon, morphotactics, orthography
    - analysis, generation

## 5. Finite state morphology

- Finite state:
    - FSA, FST, mathematical description
    - Operations on FSTs: inversion, composition, projection
- Morphology:
    - FSTs as morphological analyzers; analysis / generation, backtracking
    - lexc
- Related tasks:
    - spell checking, lemmatization, tokenization
    - how a morphological FST can be used for these tasks

## 6. Syntax: theory

- Basics: definition, grammaticality
- Concepts: POS (open&closed), grammatical relations (predicate, arguments, adjuncts), subcategorization (valency, transitivity) , constituency (tests)
- Phrase-structure grammars:
    - Components: production rules, terminals, nonterminals
    - Derivation, the parse tree, Chomsky Normal Form
- Chomsky Normal Form and its algorithm
- Dependency Grammar:
    - Difference from PSG
    - The dependency tree

## 7. Parsing and PCFG

- Parsing
    - Top-down and bottom-up parsing
    - Challenges: nondeterminism and ambiguity (global/local); examples
    - Dynamic programming; the CKY algorithm
- PCFG
    - Motivation and mathematical formulation
    - Training and testing: treebanks and PARSEVAL
    - Problems and solutions: subcategorization and lexicalization,
                              independence and annotation

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
