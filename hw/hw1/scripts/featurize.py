'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

# ====== 新增统计/模式特征（未在现有代码中出现） ======
def text_len_chars(text, freq):
    return float(len(text))

def word_count_feature(text, freq):
    words = re.findall(r'\w+', text)
    return float(len(words))

def hapax_ratio(text, freq):
    """只出现一次的词占比"""
    total = sum(freq.values())
    if total == 0: return 0.0
    ones = sum(1 for v in freq.values() if v == 1)
    return ones / float(len(freq)) if len(freq) > 0 else 0.0

def herfindahl_wordfreq(text, freq):
    """词频分布的集中度 H = sum(p_i^2)"""
    total = float(sum(freq.values()))
    if total == 0: return 0.0
    return sum((v/total)**2 for v in freq.values())

def uppercase_word_ratio(text, freq):
    """全大写的英文词比例（长度>=2）"""
    words = re.findall(r'[A-Za-z]+', text)
    if not words: return 0.0
    uc = sum(1 for w in words if w.isupper() and len(w) >= 2)
    return uc / float(len(words))

def long_word_ratio(text, freq, L=10):
    """长词比例"""
    words = re.findall(r'\w+', text)
    if not words: return 0.0
    return sum(1 for w in words if len(w) >= L) / float(len(words))

def digit_char_ratio(text, freq):
    """数字字符占比（长度归一化）"""
    return sum(c.isdigit() for c in text) / float(max(1, len(text)))

def qmark_density(text, freq):
    """问号密度"""
    return text.count('?') / float(max(1, len(text)))

def percent_density(text, freq):
    """百分号密度（折扣/比例信息）"""
    return text.count('%') / float(max(1, len(text)))

def money_amount_count(text, freq):
    """金额样式出现次数：$1,234.56 或 1234.00 等粗略匹配"""
    pattern = r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
    return float(len(re.findall(pattern, text)))

def url_count(text, freq):
    """URL 出现次数"""
    return float(len(re.findall(r'(https?://|www\.)\S+', text)))

def email_addr_count(text, freq):
    """Email 地址出现次数"""
    pattern = r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}'
    return float(len(re.findall(pattern, text)))

def phone_like_count(text, freq):
    """电话号码样式出现次数（粗略）"""
    return float(len(re.findall(r'\+?\d[\d\-\s]{6,}\d', text)))

def repeat_char_run_max(text, freq):
    """最长重复字符 run 的长度（如 '!!!!!' -> 5）"""
    runs = [len(m.group(0)) for m in re.finditer(r'(.)\1+', text)]
    return float(max(runs) if runs else 1.0)

def spam_phrase_bigram_count(text, freq):
    """典型促销二元短语计数（可按需扩展）"""
    low = text.lower()
    phrases = [
        'free trial', 'limited time', 'win prize', 'best offer',
        'act now', 'click here', 'claim now', 'no obligation'
    ]
    return float(sum(low.count(p) for p in phrases))

# ====== 更多可解释特征（全新，未与前面重复） ======
def line_count(text, freq):
    return float(text.count('\n') + 1)

def empty_line_ratio(text, freq):
    lines = text.split('\n')
    if not lines: return 0.0
    empty = sum(1 for ln in lines if ln.strip() == '')
    return empty / float(len(lines))

def sentence_count(text, freq):
    # 粗略按 . ? ! 分句
    sents = re.split(r'[\.!?]+', text)
    return float(len([s for s in sents if s.strip() != '']))

def avg_sentence_len(text, freq):
    sents = [s.strip() for s in re.split(r'[\.!?]+', text) if s.strip() != '']
    if not sents: return 0.0
    return float(np.mean([len(s) for s in sents]))

def end_exclaim_ratio(text, freq):
    lines = [ln.strip() for ln in text.split('\n') if ln.strip() != '']
    if not lines: return 0.0
    return sum(1 for ln in lines if ln.endswith('!')) / float(len(lines))

def start_with_greeting(text, freq):
    # 是否以常见问候开头（ham 倾向）
    head = text.strip().lower()[:80]
    return float(bool(re.search(r'\b(hi|hello|dear|hey|greetings)\b', head)))

def contains_unsubscribe(text, freq):
    low = text.lower()
    return float(('unsubscribe' in low) or ('退订' in low) or ('取消订阅' in low))

def footer_disclaimer_flag(text, freq):
    # 免责声明/隐私等（ham/营销正规邮件常见）
    low = text.lower()
    patterns = ['privacy policy', 'terms of service', 'do not reply',
                'confidential', '免责声明', '保密', '请勿回复']
    return float(any(p in low for p in patterns))

def tld_suspicious_count(text, freq):
    # 可疑 TLD 次数：.ru .cn .top .xyz 等
    return float(len(re.findall(r'https?://[^/\s]+\.(?:ru|cn|top|xyz|club|click|work|info)\b', text.lower())))

def url_param_avglen(text, freq):
    # URL 参数平均长度 ?... 之后到空白/结束
    qs = re.findall(r'https?://\S+\?(\S+)', text)
    if not qs: return 0.0
    return float(np.mean([len(q.split()[0]) for q in qs]))

def anchor_like_count(text, freq):
    # 伪HTML锚文本或markdown风格 [text](url) 的粗匹配
    n_html = len(re.findall(r'<a[^>]*href=', text, flags=re.I))
    n_md   = len(re.findall(r'\[[^\]]+\]\((https?://|www\.)', text))
    return float(n_html + n_md)

def hex_string_ratio(text, freq):
    # 十六进制串密度（追踪码/哈希痕迹）
    total = max(1, len(text))
    hexspans = sum(len(m.group(0)) for m in re.finditer(r'\b[0-9A-Fa-f]{16,}\b', text))
    return hexspans / float(total)

def non_ascii_ratio(text, freq):
    total = max(1, len(text))
    return sum(ord(c) > 127 for c in text) / float(total)

def unicode_symbol_ratio(text, freq):
    # 特殊符号（★☆◆●•™©® 等）密度
    return len(re.findall(r'[★☆◆●•™©®✓✔✕✖▶►♦♣♥♠→←↑↓…·•—–—]', text)) / float(max(1, len(text)))

def qp_encoding_count(text, freq):
    # Quoted-Printable 痕迹（=3D, =20, =09 等）
    return float(len(re.findall(r'=(?:3D|20|09|0A|0D)[A-Fa-f0-9]{0,2}', text)))

def repeated_word_ratio(text, freq):
    # 重复词比例：出现≥3次的词种数 / 词种数
    if not freq: return 0.0
    types = len(freq)
    rep = sum(1 for v in freq.values() if v >= 3)
    return rep / float(types)

def topk_coverage(text, freq, k=10):
    # Top-k 高频词覆盖率：前k词频/总词频
    if not freq: return 0.0
    vals = sorted(freq.values(), reverse=True)
    return sum(vals[:min(k, len(vals))]) / float(sum(vals))

def vowel_consonant_ratio(text, freq):
    words = re.findall(r'[A-Za-z]+', text.lower())
    if not words: return 0.0
    v = sum(ch in 'aeiou' for w in words for ch in w)
    c = sum(ch in 'bcdfghjklmnpqrstvwxyz' for w in words for ch in w)
    return v / float(max(1, c))


# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    # ====== 新增：通用统计/模式特征（均为 float/int）======
    feature.append(text_len_chars(text, freq))        # 文本字符数
    feature.append(word_count_feature(text, freq))    # 词数
    feature.append(hapax_ratio(text, freq))           # Hapax比例
    feature.append(herfindahl_wordfreq(text, freq))   # 词频集中度H
    feature.append(uppercase_word_ratio(text, freq))  # 全大写词比例
    feature.append(long_word_ratio(text, freq))       # 长词比例
    feature.append(digit_char_ratio(text, freq))      # 数字字符占比
    feature.append(qmark_density(text, freq))         # '?'密度
    feature.append(percent_density(text, freq))       # '%'密度
    feature.append(money_amount_count(text, freq))    # 金额样式计数
    feature.append(url_count(text, freq))             # URL计数
    feature.append(email_addr_count(text, freq))      # Email计数
    feature.append(phone_like_count(text, freq))      # 电话样式计数
    feature.append(repeat_char_run_max(text, freq))   # 最长重复字符run
    feature.append(spam_phrase_bigram_count(text, freq))  # 促销短语二元计数
    feature.append(line_count(text, freq))
    feature.append(empty_line_ratio(text, freq))
    feature.append(sentence_count(text, freq))
    feature.append(avg_sentence_len(text, freq))
    feature.append(end_exclaim_ratio(text, freq))
    feature.append(start_with_greeting(text, freq))
    feature.append(contains_unsubscribe(text, freq))
    feature.append(footer_disclaimer_flag(text, freq))
    feature.append(tld_suspicious_count(text, freq))
    feature.append(url_param_avglen(text, freq))
    feature.append(anchor_like_count(text, freq))
    feature.append(hex_string_ratio(text, freq))
    feature.append(non_ascii_ratio(text, freq))
    feature.append(unicode_symbol_ratio(text, freq))
    feature.append(qp_encoding_count(text, freq))
    feature.append(repeated_word_ratio(text, freq))
    feature.append(topk_coverage(text, freq))
    feature.append(vowel_consonant_ratio(text, freq))

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'spam-data.npz', training_data=X, training_labels=Y, test_data=test_design_matrix)
