from typing import List

RULE_LEXICON = {
    "1P": "I", "2P": "you", "3P": "he",
    "IX": "", "NOT": "not", "NO": "not",
    "WHERE": "where", "WHAT": "what",
    "GO": "go", "EAT": "eat", "WAKE": "wake", "UP": "up",
}

def rule_based_translate(gloss_tokens: List[str]) -> str:
    words = []
    for g in gloss_tokens:
        g = g.strip()
        if g in RULE_LEXICON:
            w = RULE_LEXICON[g]
            if w:
                words.append(w)
            continue
        parts = [p for p in g.split("-") if p]
        mapped = []
        for p in parts:
            if p in RULE_LEXICON:
                if RULE_LEXICON[p]:
                    mapped.append(RULE_LEXICON[p])
            else:
                mapped.append(p.lower())
        if mapped:
            words.append(" ".join(mapped))
    return " ".join(words).strip()
