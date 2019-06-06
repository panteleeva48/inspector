from app import app, render_template, request, Markup
import os
from app.utils.get_feature_values import GetFeatures
from config import (
    UDPIPE_MODEL, chkr, BASE_DIR, numeric_features, UWL, CONNECTORS,
    important_features, feature_mapping, feature_feedback_to_improve, feature_feedback_best)
import pickle
from scipy import sparse
import numpy as np
import re
from json2html import *
import json
from string import punctuation
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

VECTORIZER_BIN_PATH = os.path.join(BASE_DIR, 'data', 'models', 'vectorizer_binary.pickle')
with open(VECTORIZER_BIN_PATH, 'rb') as vcrzr:
    VECTORIZER_BIN = pickle.load(vcrzr)

VECTORIZER_MULTI_PATH = os.path.join(BASE_DIR, 'data', 'models', 'vectorizer_multi.pickle')
with open(VECTORIZER_MULTI_PATH, 'rb') as vcrzr:
    VECTORIZER_MULTI = pickle.load(vcrzr)

GRADE_BIN_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'binary_model.pickle')
with open(GRADE_BIN_MODEL_PATH, 'rb') as f:
    GRADE_BIN_MODEL = pickle.load(f)

GRADE_MULTI_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'multi_model.pickle')
with open(GRADE_MULTI_MODEL_PATH, 'rb') as f:
    GRADE_MULTI_MODEL = pickle.load(f)

MEAN_PATH = os.path.join(BASE_DIR, 'data', 'data_for_train', 'means.json')
with open(MEAN_PATH, 'r') as f:
    MEAN = json.load(f)

VECTORIZER_TYPE_PATH = os.path.join(BASE_DIR, 'data', 'models', 'vectorizer_type.pickle')
with open(VECTORIZER_TYPE_PATH, 'rb') as vcrzr:
    VECTORIZER_TYPE = pickle.load(vcrzr)

TYPE_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'type_model.pickle')
with open(TYPE_MODEL_PATH, 'rb') as f:
    TYPE_MODEL = pickle.load(f)

loaded_vectorizer_bin = TfidfVectorizer(decode_error='replace',
                                        vocabulary=VECTORIZER_BIN
                                        )

loaded_vectorizer_multi = TfidfVectorizer(decode_error='replace',
                                          vocabulary=VECTORIZER_MULTI
                                          )

loaded_vectorizer_type = TfidfVectorizer(decode_error='replace',
                                         vocabulary=VECTORIZER_TYPE
                                         )

gf = GetFeatures(UDPIPE_MODEL)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'files')
ALLOWED_EXTENSIONS = {'txt'}


def tokenizer(text):
    return text.split()


DON_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'shell.pickle')
with open(DON_MODEL_PATH, 'rb') as mdl:
    DON_MODEL = pickle.load(mdl)


def check_spelling(text):
    chkr.set_text(text)
    mistakes = []
    for err in chkr:
        suggestions = err.suggest()
        if suggestions:
            suggest = suggestions[0]
            err.replace(suggest)
            word = err.word
            mistakes.append((word, suggest))
    text = chkr.get_text()
    return text, mistakes


def gather_values(gf):
    result = dict()
    result['av_depth'] = gf.av_depth()
    result['max_depth'] = gf.max_depth()
    result['min_depth'] = gf.min_depth()
    result['num_acl'], result['num_rel_cl'], result['num_advcl'] = gf.count_dep_sent()
    result['num_sent'] = gf.count_sent()
    result['num_tok'] = gf.count_tokens()
    result['av_tok_before_root'] = gf.tokens_before_root()
    result['av_len_sent'] = gf.mean_len_sent()
    result['num_cl'], result['num_tu'], result['num_compl_tu'] = gf.count_units()
    result['num_coord'] = gf.count_coord()
    result['num_poss'], result['num_prep'] = gf.count_poss_prep()
    result['num_adj_noun'] = gf.count_adj_noun()
    result['num_part_noun'] = gf.count_part_noun()
    result['num_noun_inf'] = gf.count_noun_inf()
    result['pos_sim_nei'], result['lemma_sim_nei'] = gf.simularity_neibour()
    result['pos_sim_all'], result['lemma_sim_all'] = gf.simularity_mean()
    result['density'] = gf.density()
    result['ls'] = gf.LS()
    result['vs'], result['corrected_vs'], result['squared_vs'] =  gf.VS()
    result['lfp_1000'], result['lfp_2000'], result['lfp_uwl'], result['lfp_rest'] = gf.LFP()
    result['ndw'] = gf.NDW()
    result['ttr'], result['corrected_ttr'], result['root_ttr'], result['log_ttr'], result['uber_ttr'] = gf.TTR()
    result['d'] = gf.D()
    result['lv'] = gf.LV()
    result['vvi'], result['squared_vv'], result['corrected_vv'], result['vvii'] = gf.VV()
    result['nv'] = gf.NV()
    result['adjv'] = gf.AdjV()
    result['advv'] = gf.AdvV()
    result['modv'] = gf.ModV()
    (result['der_level3'], result['der_level4'],
     result['der_level5'], result['der_level6']) = gf.derivational_suffixation()
    result['mci'] = gf.MCI()
    result['freq_finite_forms'] = gf.freq_finite_forms()
    result['freq_aux'] = gf.freq_aux()
    (result['num_inf'], result['num_gerunds'], result['num_pres_sing'],
     result['num_pres_plur'], result['num_past_part'], result['num_past_simple']) = gf.num_verb_forms()
    result['num_linkings'] = gf.num_linkings().get('link_all')
    result['num_4grams'] = gf.num_4grams()
    result['num_func_ngrams'] = gf.num_func_ngrams().get('4grams_all')
    result['num_shell_noun'] = gf.shell_nouns(DON_MODEL)
    result['num_misspelled_tokens'] = gf.number_of_misspelled()
    result['sum_punct'] = gf.count_punct_mistakes_participle_phrase() + \
                          gf.count_punct_mistakes_before(before='because') + \
                          gf.count_punct_mistakes_before(before='but') + \
                          gf.count_punct_mistakes_before(before='than') + \
                          gf.count_punct_mistakes_before(before='like')
    result['million_mistake'] = gf.count_million_mistakes()
    return result


def read_file(path):
    with open(path, 'r', encoding='utf-8') as fr:
        text = fr.read()
    return text


def predict_binary(text, feature_dict):
    num_values = [round(feature_dict.get(feature_value, 0), 2) for feature_value in numeric_features]
    text = loaded_vectorizer_bin.fit_transform(np.array([text]))
    X = sparse.hstack([num_values, text])
    group = GRADE_BIN_MODEL.predict(X)[0]
    return group


def predict_multi(text, feature_dict):
    num_values = [round(feature_dict.get(feature_value, 0), 2) for feature_value in important_features]
    text = loaded_vectorizer_multi.fit_transform(np.array([text]))
    X = sparse.hstack([num_values, text])
    grade = GRADE_MULTI_MODEL.predict(X)[0]
    return grade


def predict_type(text):
    text = loaded_vectorizer_type.fit_transform(np.array([text]))
    X = sparse.hstack([text])
    type = TYPE_MODEL.predict(X)[0]
    return type


def predict_grade(text, feature_dict):
    group = predict_binary(text, feature_dict)
    grade = predict_multi(text, feature_dict)
    binary_result = None
    if group == 'non-best':
        binary_result = 'Your essay is not good enough.'
    elif group == 'best':
        binary_result = 'Your essay is good.'
    return binary_result, grade


def get_academic_words(text):
    bolded_text = ''
    words = text.split()
    num_aca = 0
    for word in words:
        stripped_word = word.strip(punctuation)
        lemma = wordnet_lemmatizer.lemmatize(stripped_word)
        if lemma in UWL:
            marked_word = re.sub(stripped_word, '<span class="academic">{stripped_word}</span>', word)
            bolded_text += str(marked_word) + ' '
            num_aca += 1
        else:
            bolded_text += str(word) + ' '
    return bolded_text, num_aca


def get_connectors(text):
    text = re.sub('( |^|\n)(' + str(CONNECTORS) + ')( |$|\n)', '<span class="connector">\g<1>\g<2>\g<3></span>', text)
    num_con = len(re.findall('<span class="connector">', text))
    return text, num_con


def get_result_spelling_mistakes(text, mistakes):
    result_spelling = ''
    for mistake in mistakes:
        text_word = mistake[0]
        correction = mistake[1]
        sentence = re.search('([^.]*?{' + str(text_word) + '[^.]*\.)', text)
        if sentence:
            sentence = sentence.group()
            result = re.sub(str(text_word), '<span class="spelling">' + str(text_word) + ' &#8594; ' + str(correction) + '</span>', sentence)
            result_spelling += str(result) + '<br>'
    return result_spelling


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_closest_to_non_best(best, non_best, value, feature_name):
    if value >= best:
        return None
    else:
        return feature_name


def get_type_num(type, num_tok):
    if type == 1:
        if num_tok < 150:
            return 'It is not enough words in your essay. The required size of the graph discription essay is at least 150 words. '
    if type == 2:
        if num_tok < 250:
            return 'It is not enough words in your essay. The required size of the opinion essay is at least 250 words. '
    return ''


def recommendation(feature_dict):
    MEAN_PATH = os.path.join(BASE_DIR, 'data', 'data_for_train', 'means.json')
    with open(MEAN_PATH, 'r') as f:
        MEAN = json.load(f)
    features_to_improve = []
    MEAN_ordered = []
    for obj in MEAN:
        feature_name = obj.get('Feature')
        value = round(feature_dict.get(feature_name), 2)
        obj['Your essay'] = value
        feature_to_improve = get_closest_to_non_best(obj['Best'], obj['Non-best'], value, feature_name)
        del obj['Non-best']
        if feature_to_improve:
            features_to_improve.append(feature_to_improve)
            MEAN_ordered.append(obj)
        else:
            MEAN_ordered = [obj] + MEAN_ordered
    table = json2html.convert(json=MEAN_ordered)
    table = table.replace('<table border="1">', '<table class ="table_recommendation">')
    table = table.replace('<td>', '<td style="width: 100px;text-align: center;">')
    feedback_to_improve = 'All in all, in your essay you should use '
    feedback_best = 'You used enough '
    for obj in MEAN:
        feature_name = obj.get('Feature')
        if feature_name in features_to_improve:
            table = table.replace('width: 100px;text-align: center;">' + str(feature_name),
                                  'width: 300px;color: darkred;text-align: center;'
                                  'font-weight: bolder;">' + str(feature_mapping.get(feature_name)))
            fb = feature_feedback_to_improve.get(feature_name)
            if fb not in feedback_to_improve:
                feedback_to_improve += 'more ' + str(fb) + '; '
        else:
            table = table.replace('width: 100px;'
                                  'text-align: center;">' + str(feature_name),
                                  'width: 300px;color: darkgreen;text-align: center;'
                                  'font-weight: bolder;">' + str(feature_mapping.get(feature_name)))
            fb = feature_feedback_best.get(feature_name)
            if fb not in feedback_best:
                feedback_best += str(fb) + '; '
    feedback_to_improve = feedback_to_improve.strip('; ') + '.'
    feedback_to_improve = re.sub('; more', ';', feedback_to_improve)
    feedback_best = feedback_best.strip('; ') + '.'
    return table, str(feedback_to_improve) + ' ' + str(feedback_best)


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/', methods=['GET', 'POST'])
def analyze_text():
    text = None
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            text = read_file(path)
            os.remove(path)
        else:
            text = None
    if not text:
        text = request.form['text']
    if not text:
        return render_template('main.html')
    changed_text, mistakes = check_spelling(text)
    result_spelling = get_result_spelling_mistakes(text, mistakes)
    gf.get_info(changed_text)
    num_tok = gf.count_tokens(punct=False)
    feature_dict = gather_values(gf)
    binary_result, grade = predict_grade(changed_text, feature_dict)
    type = predict_type(changed_text)
    rec_num_tok = get_type_num(type, num_tok)
    bolded_text, num_aca = get_academic_words(changed_text)
    bolded_text_2, num_con = get_connectors(bolded_text)
    mean_info, feedback = recommendation(feature_dict)
    feedback = rec_num_tok + feedback
    return render_template('main.html',
                           text=text,
                           show_results=1,
                           binary_result=binary_result,
                           grade=grade,
                           num_tok=num_tok,
                           num_sent=feature_dict.get('num_sent', 0),
                           ndw=feature_dict.get('ndw', 0),
                           bolded_text=Markup(bolded_text_2),
                           mistakes=Markup(result_spelling),
                           num_con=num_con,
                           num_aca=num_aca,
                           mean=Markup(mean_info),
                           feedback=feedback
                           )
