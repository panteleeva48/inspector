import os
from app.utils.model import Model
import json

import enchant.checker as spellcheck
chkr = spellcheck.SpellChecker("en_GB")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, 'app')

PATH_UDPIPE_MODEL = os.path.join(BASE_DIR, 'data', 'models', 'english-partut-ud-2.3-181115.udpipe')
UDPIPE_MODEL = Model(PATH_UDPIPE_MODEL)

PATH_LISTS = os.path.join(BASE_DIR, 'data', 'lists', 'lists.json')
with open(PATH_LISTS, encoding='utf-8') as data_file:
    lists = json.load(data_file)
FIVE_T_FREQ_COCA = lists['5000frequentCOCA']
FREQ_VERBS_COCA_FROM_FIVE_T = lists['frequentverbsCOCAfrom5000']
UWL = lists['UWL']

OPEN_CLASS = ['NOUN', 'VERB', 'ADV', 'ADJ', 'PROPN']

PATH_LINKINGS = os.path.join(BASE_DIR, 'data', 'lists', 'linkings.json')
with open(PATH_LINKINGS, encoding='utf-8') as data_file:
    LINKINGS = json.load(data_file)

PATH_FUNC_NGRAMS = os.path.join(BASE_DIR, 'data', 'lists', 'functional_ngrams.json')
with open(PATH_FUNC_NGRAMS, encoding='utf-8') as data_file:
    FUNC_NGRAMS = json.load(data_file)

PATH_SUFFIXES = os.path.join(BASE_DIR, 'data', 'lists', 'suffixes.json')
with open(PATH_SUFFIXES, encoding='utf-8') as data_file:
    SUFFIXES = json.load(data_file)

PATH_NGRAMS = os.path.join(BASE_DIR, 'data', 'lists', 'ngrams.txt')
with open(PATH_NGRAMS, encoding='utf-8') as data_file:
    NGRAMS = [x.split() for x in data_file.read().split('\n')]

PATH_CONNECTORS = os.path.join(BASE_DIR, 'data', 'lists', 'connectors.txt')
with open(PATH_CONNECTORS, 'r', encoding='utf-8') as data_file:
    CONNECTORS = data_file.read().strip()

DONS = [
    'thing', 'fact', 'point', 'argument', 'result', 'dispute',
    'problem', 'factor', 'approach', 'view', 'feeling', 'process',
    'theme', 'attempt', 'controversy', 'statement', 'task', 'issue',
    'dream', 'matter', 'situation', 'need', 'reason', 'solution',
    'possibility', 'change', 'debate', 'sense', 'method', 'theory',
    'finding', 'question', 'idea', 'concept', 'opinion', 'ideas', 'things'
]

NUM_LIST = ['millions', 'hundreds',
            'thousands', 'milliards',
            'billions', 'trillions']

numeric_features = ['av_depth', 'max_depth', 'min_depth', 'num_acl', 'num_rel_cl', 'num_advcl', 'num_sent', 'num_tok',
                    'av_tok_before_root', 'av_len_sent', 'num_cl', 'num_tu', 'num_compl_tu', 'num_coord', 'num_poss',
                    'num_prep', 'num_adj_noun', 'num_part_noun', 'num_noun_inf', 'pos_sim_nei', 'pos_sim_all',
                    'lemma_sim_all', 'lemma_sim_nei', 'density', 'ls', 'corrected_vs', 'lfp_1000', 'lfp_2000',
                    'lfp_uwl', 'lfp_rest', 'ndw', 'corrected_ttr', 'lv', 'corrected_vv', 'vvii', 'nv', 'adjv', 'advv',
                    'modv', 'der_level3', 'der_level4', 'der_level5', 'der_level6', 'mci', 'freq_finite_forms',
                    'freq_aux', 'num_inf', 'num_gerunds', 'num_pres_sing', 'num_pres_plur', 'num_past_part',
                    'num_past_simple', 'num_linkings', 'num_4grams', 'num_func_ngrams', 'num_shell_noun',
                    'num_misspelled_tokens', 'million_mistake', 'sum_punct'
                    ]

important_features = ['corrected_vv', 'num_linkings', 'av_len_sent',
                      'ls', 'num_gerunds', 'der_level3', 'nv', 'der_level4', 'num_inf',
                      'ndw', 'der_level6', 'num_func_ngrams', 'num_cl', 'num_acl', 'av_tok_before_root',
                      'num_sent', 'num_past_simple',
                      'freq_aux', 'corrected_vs', 'der_level5', 'density', 'num_coord', 'num_poss']

feature_mapping = {
    'corrected_vv': 'Verb variation',
    'num_linkings': 'Number of linking phrases',
    'av_len_sent': 'Average length of the sentence',
    'ls': 'Lexical sophistication',
    'num_gerunds': 'Number of gerunds',
    'nv': 'Noun variation',
    'num_inf': 'Number of infinitives',
    'ndw': 'Number of lemmas',
    'num_func_ngrams': 'Number of functional n-grams',
    'num_cl': 'Number of clauses',
    'num_acl': 'Number of adjectival clauses',
    'av_tok_before_root': 'Average number of words before root',
    'num_sent': 'Number of sentences',
    'num_past_simple': 'Number of verbs in past simple',
    'freq_aux': 'Number of auxilaries',
    'corrected_vs': 'Verb sophistication',
    'density': 'Lexical density',
    'num_coord': 'Number of coordinated phrases',
    'num_poss': 'Number of possessives',
    'der_level5': 'Derivational level 5',
    'der_level3': 'Derivational level 3',
    'der_level4': 'Derivational level 4',
    'der_level6': 'Derivational level 6'
}

feature_feedback_to_improve = {
    'corrected_vv': 'different verbs',
    'num_linkings': 'linking phrases',
    'av_len_sent': 'large sentences',
    'ls': 'academic words',
    'num_gerunds': 'gerunds',
    'nv': 'different nouns',
    'num_inf': 'infinitives',
    'ndw': 'different words',
    'num_func_ngrams': 'functional n-grams',
    'num_cl': 'complex sentences',
    'num_acl': 'relative clauses',
    'av_tok_before_root': 'words before the main predicate',
    'num_sent': 'sentences',
    'num_past_simple': 'verbs in past simple',
    'freq_aux': 'auxiliary verbs',
    'corrected_vs': 'sophisticated verbs',
    'density': 'nouns, verbs, adverbs, and adjectives',
    'num_coord': 'coordinate constructions',
    'num_poss': 'possessive constructions',
    'der_level5': 'morphologically complex words',
    'der_level3': 'morphologically complex words',
    'der_level4': 'morphologically complex words',
    'der_level6': 'morphologically complex words'
}

feature_feedback_best = {
    'corrected_vv': 'different verbs',
    'num_linkings': 'linking phrases',
    'av_len_sent': 'long sentences',
    'ls': 'academic words',
    'num_gerunds': 'gerunds',
    'nv': 'different nouns',
    'num_inf': 'infinitives',
    'ndw': 'different words',
    'num_func_ngrams': 'functional n-grams',
    'num_cl': 'complex sentences',
    'num_acl': 'relative clauses',
    'av_tok_before_root': 'words before the main predicate',
    'num_sent': 'sentences',
    'num_past_simple': 'verbs in past simple',
    'freq_aux': 'auxiliary verbs',
    'corrected_vs': 'sophisticated verbs',
    'density': 'nouns, verbs, adverbs, and adjectives',
    'num_coord': 'coordinate constructions',
    'num_poss': 'possessive constructions',
    'der_level5': 'morphologically complex words',
    'der_level3': 'morphologically complex words',
    'der_level4': 'morphologically complex words',
    'der_level6': 'morphologically complex words'
}