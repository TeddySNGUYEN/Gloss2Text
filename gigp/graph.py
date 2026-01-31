import re
from dataclasses import dataclass
from typing import List, Tuple

from .vocab import Vocab
from .utils import PAD, UNK

ROLE_REFERENCE = "REFERENCE"
ROLE_AGREEMENT = "AGREEMENT"
ROLE_ASPECT    = "ASPECT"
ROLE_NEGATION  = "NEGATION"
ROLE_LOCATIVE  = "LOCATIVE"
ROLE_CLASSIF   = "CLASSIFIER"
ROLE_LEXROOT   = "LEXICALROOT"
ROLE_OTHER     = "OTHER"

REL_REFERENCE  = "REL_REFERENCE"
REL_MODIFY     = "REL_MODIFICATION"
REL_AGREE      = "REL_AGREEMENT"
REL_OTHER      = "REL_OTHER"

_ix_pat = re.compile(r"^IX(-|$)")
_person_pat = re.compile(r"^(1P|2P|3P)$")
_neg_pat = re.compile(r"^(NOT|NO)$")
_loc_pat = re.compile(r"^(LOC|LOCATION)$")
_cls_pat = re.compile(r"^CL:")
ASPECT_SET = {"REPEATED", "DURATIVE", "HABITUAL", "UP"}

def segment_gloss(gloss: str) -> List[str]:
    parts = [p for p in gloss.split("-") if p]
    return parts if len(parts) > 1 else [gloss]

def assign_role(component: str) -> str:
    if _ix_pat.match(component): return ROLE_REFERENCE
    if _person_pat.match(component): return ROLE_AGREEMENT
    if _neg_pat.match(component): return ROLE_NEGATION
    if _loc_pat.match(component): return ROLE_LOCATIVE
    if _cls_pat.match(component): return ROLE_CLASSIF
    if component in ASPECT_SET: return ROLE_ASPECT
    return ROLE_LEXROOT

def infer_relation(role_child: str) -> str:
    if role_child == ROLE_REFERENCE: return REL_REFERENCE
    if role_child in {ROLE_ASPECT, ROLE_NEGATION, ROLE_LOCATIVE, ROLE_CLASSIF}: return REL_MODIFY
    if role_child == ROLE_AGREEMENT: return REL_AGREE
    return REL_OTHER

def is_agreement_pair(role_a: str, role_b: str) -> bool:
    return (role_a == ROLE_REFERENCE and role_b == ROLE_AGREEMENT) or (role_a == ROLE_AGREEMENT and role_b == ROLE_REFERENCE)

@dataclass(frozen=True)
class Graph:
    node_tokens: Tuple[str, ...]
    node_roles:  Tuple[str, ...]
    edges:       Tuple[Tuple[int, int, str], ...]
    root_idx:    int

def build_gloss_graph(gloss: str) -> Graph:
    parts = segment_gloss(gloss)
    if len(parts) == 1:
        return Graph(node_tokens=(gloss,), node_roles=("LEXICALROOT",), edges=(), root_idx=0)

    roles = [assign_role(p) for p in parts]
    root_idx = next((i for i, r in enumerate(roles) if r == "LEXICALROOT"), len(parts) - 1)

    edges = []
    for i, r in enumerate(roles):
        if i == root_idx:
            continue
        edges.append((i, root_idx, infer_relation(r)))

    for i in range(len(parts)):
        for j in range(len(parts)):
            if i != j and is_agreement_pair(roles[i], roles[j]):
                edges.append((i, j, REL_AGREE))

    return Graph(node_tokens=tuple(parts), node_roles=tuple(roles), edges=tuple(edges), root_idx=int(root_idx))

def build_graph_vocabs(train_ds, min_freq=1):
    node_vocab = Vocab(min_freq=min_freq)
    role_vocab = Vocab(min_freq=1, specials=(PAD, UNK))
    rel_vocab  = Vocab(min_freq=1, specials=(PAD, UNK))

    for r in [ROLE_REFERENCE, ROLE_AGREEMENT, ROLE_ASPECT, ROLE_NEGATION, ROLE_LOCATIVE, ROLE_CLASSIF, ROLE_LEXROOT, ROLE_OTHER]:
        role_vocab.add_sentence([r])
    for r in [REL_REFERENCE, REL_MODIFY, REL_AGREE, REL_OTHER]:
        rel_vocab.add_sentence([r])

    for gloss_tokens, _, _, _ in train_ds:
        for g in gloss_tokens:
            G = build_gloss_graph(g)
            node_vocab.add_sentence(list(G.node_tokens))
            role_vocab.add_sentence(list(G.node_roles))
            if G.edges:
                rel_vocab.add_sentence([e[2] for e in G.edges])

    node_vocab.build()
    role_vocab.build()
    rel_vocab.build()
    return node_vocab, role_vocab, rel_vocab
