import sacrebleu
from .utils import PAD, BOS, EOS

def bleu_chrf_from_texts(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none").score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return bleu, chrf

def decode_ids_to_text(ids, tgt_vocab):
    toks = tgt_vocab.decode(ids)
    toks = [t for t in toks if t not in {PAD, BOS, EOS}]
    return " ".join(toks).strip()
