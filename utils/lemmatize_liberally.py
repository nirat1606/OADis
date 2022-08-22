# Source: https://gist.github.com/nschneid/d49de87813bd6499ff7c2d861eba196c

import pattern.en
from nltk.corpus import wordnet as wn

exception = {'pant', 'trouser', 'short', 'jean', 'fry', 'french fry'}
norm_object_name = dict()


def normalize_object(obj):
    if obj in norm_object_name:
        return norm_object_name[obj]
    # Special case handling (e.g., grass).
    if obj.endswith('ss'):
        norm_object_name[obj] = obj
        return obj

    if obj in exception:
        res = pattern.en.pluralize(obj)
    else:
        if "'" in obj:
            tmp = pattern.en.singularize(obj)
        else:
            tmp = lemmatize_liberally(obj, 'NNS')
        if tmp not in exception:
            res = tmp
        else:
            res = obj
    norm_object_name[obj] = res
    return res


def lemmatize_liberally(w, p):
    """
    Given an English word and its PTB POS tag (whose fine-grained information helps disambiguate some words), 
    produce its lemma/stem. Uses WordNet, but is more aggressive than the `morphy` behavior, giving less surprising 
    results for words like 'people' and 'eggs'.
    
    >>> lemmatize_liberally('people', 'NNS')
    person
    >>> lemmatize_liberally('men', 'NNS')
    man
    >>> lemmatize_liberally('arms', 'NNS')
    arm
    >>> lemmatize_liberally('eggs', 'NNS')
    egg
    >>> lemmatize_liberally('twenties', 'NNS')
    twenty
    >>> lemmatize_liberally('woods', 'NNS')
    wood
    >>> lemmatize_liberally('glasses', 'NNS')
    glasses
    >>> lemmatize_liberally('moses', 'NNS')
    moses
    >>> lemmatize_liberally('alps', 'NNS')
    alps
    """
    
    w = w.lower()
    if p in {'VBD','VBN'}:
        if w=='found': return 'find'
        elif w=='ground': return 'grind'
        elif w=='rent': return 'rend'
        elif w=='smelt': return 'smell'
        elif w=='wound': return 'wind'
        elif p=='VBD':
            if w=='fell': return 'fall'
            elif w=='lay': return 'lie'
            elif w=='saw': return 'see'
    elif p[0]=='V' and w=='stove': return 'stove'  # WordNet has only the past/ppt form of 'stave', but apparently 'stove' can be a verb
    # 'ridden' is a past participle of 'rid' and the past participle of 'ride'. The POS is not enough to disambiguate, but 'ride' (which WordNet gives) is probably more common.
    elif p=='NNS':  # checked http://www.esldesk.com/vocabulary/irregular-nouns for irregular forms
        if w=='people': return 'person'
        elif w=='teeth': return 'tooth'
        elif w=='men': return 'man'
        elif w=='brethren': return 'brother'
        elif w=='dice': return 'die'
        elif w=='elves': return 'elf'
        elif w=='fungi': return 'fungus'
        elif w=='memoranda': return 'memorandum'
        elif w=='oxen': return 'ox'
        elif w=='vitae': return 'vita'
        elif w in {'clutches', 'losses', 'marches', 'masses', 'starches'}:  # sibilant + -es
            return w[:-2]
            # 'axes' will stem to 'ax', though it could also be 'axis'
            # 'bases' will stem to 'base' though it could also be 'basis'
            # 'glasses' will stem to 'glasses' (i.e. eyewear) though it could be the plural of 'glass'
            # 'breeches', 'riches' will not change when stemmed.
    
    # TODO: these->this, those->that, an->a?
    # TODO: towards->toward, o'er->over, till->until, outta->out of, etc.?
    
    tt = [w]
    lem = wn.morphy(tt[0], p and {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}.get(p[0]))
    if lem:
        if lem==w:
            if p=='NNS' and w.endswith('ies') and wn.lemmas(w[:-3]+'y', wn.NOUN):   # exclude: 'rabies', etc.
                if w not in {'alleghenies', 'mounties', 'humanities', 'species'}:
                    lem = w[:-3]+'y'    # replace -ies with -y
                    # TODO: if 'mounties' is tagged as NNPS, could stem to 'mountie'. don't do that for 'alleghenies'
            elif p=='NNS' and len(w)>=4 and w.endswith('s') and not w.endswith('ss') and not w.endswith('us') and wn.lemmas(w[:-1], wn.NOUN):
                # excludes 'yes', 'halitosis', etc. (which are probably not NNS)
                if w=='abcs':
                    lem = 'abc'
                elif all(l.name()[0].isupper() for l in wn.lemmas(w, wn.NOUN)):
                    pass    # always capitalized in the singular, so probably should be NNP or NNPS. e.g. 'mormons'
                elif w in {'brits', 'romans', 'alps', 'anas', 'hays'}:
                    pass    # 'brits' is usually NNPS; 'romans' is NNP or NNPS; others are NNP
                elif w in {'alas', 'amnios', 'corps', 'mores', 'acoustics', 'aquatics', 'auspices', 'statics', 'pragmatics', 'winnings'}:
                    pass    # typically not the plural form of a singular noun
                else:
                    # 575 single unhyphenated words are specially listed in their plural forms in WordNet, 
                    # though they can be regular plurals
                    # e.g. 'acres', 'arms', 'basics', 'eggs', 'proverbs', 'shorts', 'woods'
                    lem = w[:-1]    # remove -s
        tt[0] = lem
    return tt[0]