import re
import unicodedata
import os 
from data.download import  download 






def preprocess_sentence(sent):
    sent = "".join([c for c in unicodedata.normalize("NFD", sent) 
        if unicodedata.category(c) != "Mn"])
    sent = re.sub(r"([!.?])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)
    sent = re.sub(r"\s+", " ", sent)
    sent = sent.lower()
    return sent


def download_and_read():
   
    en_sents, fr_sents_in, fr_sents_out = [], [], []
    local_file = os.path.join("./dataset/fra.txt")
    with open(local_file, "r") as fin:
        for i, line in enumerate(fin):
            en_sent, fr_sent = line.strip().split('\t')[:2]
            en_sent = [w for w in preprocess_sentence(en_sent).split()]
            fr_sent = preprocess_sentence(fr_sent)
            fr_sent_in = [w for w in ("BOS " + fr_sent).split()]
            fr_sent_out = [w for w in (fr_sent + " EOS").split()]
            en_sents.append(en_sent)
            fr_sents_in.append(fr_sent_in)
            fr_sents_out.append(fr_sent_out)
            if i >= 300 - 1:
                break
    return en_sents, fr_sents_in, fr_sents_out