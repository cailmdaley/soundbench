"""Lyric syllables to sung phonemes.

A score's lyrics reach the renderer as one text per sung syllable, in order,
with a flag saying whether a hyphen joins it to the next (LilyPond's
``Dai -- sy``; see scripts/lilypond/sung-lyrics.ily). Syllables joined by
hyphens are one word. Each word is looked up in CMUdict (ARPAbet with
stress digits), or in the score's own lexicon first, and its phonemes are
split across the word's sung syllables: one vowel nucleus per syllable,
consonants between nuclei divided by the maximal-onset principle (as many
as form a legal English onset go to the following syllable, the rest close
the previous one).

When a word has more vowels than sung syllables, neighbouring nuclei share
a syllable (``flow-er`` on one note is sung ``flow'r`` with both vowels);
when it has fewer, the pronunciation variant with the right count is
chosen, and failing that the last syllables continue the vowel like a
melisma. A syllable with no vowel at all (``hm``, ``mm``, ``shh``) makes
its consonants the nucleus: a held nasal or fricative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re

VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
CONSONANTS = frozenset("B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split())

# Legal English onsets (Giegerich 1992; the clusters a word may begin with).
_SINGLE = CONSONANTS - {"NG"}
_CLUSTERS = """
P-L P-R P-Y B-L B-R B-Y T-R T-W T-Y D-R D-W D-Y K-L K-R K-W K-Y G-L G-R G-W G-Y
F-L F-R F-Y TH-R TH-W SH-R S-P S-T S-K S-M S-N S-L S-W S-F S-Y V-Y M-Y N-Y H-Y HH-Y
S-P-L S-P-R S-P-Y S-T-R S-T-Y S-K-L S-K-R S-K-W S-K-Y
"""
ONSETS = frozenset({(c,) for c in _SINGLE} | {tuple(c.split("-")) for c in _CLUSTERS.split()})

# Words a sung lyric needs that CMUdict lacks or gives in a speech form.
# A score's own [lexicon] is consulted before these.
BUILTIN = {
    "hm": "M", "hmm": "M", "mm": "M", "mmm": "M", "nn": "N", "ng": "NG",
    "shh": "SH", "sh": "SH", "ss": "S",
    "ah": "AA1", "oh": "OW1", "ooh": "UW1", "oo": "UW1", "ee": "IY1", "eh": "EH1",
    "aw": "AO1", "uh": "AH1", "ai": "AY1", "ay": "AY1", "ae": "AE1", "er": "ER1",
    "la": "L AA1", "lo": "L OW1", "da": "D AA1", "dum": "D AH1 M",
    "wayfaring": "W EY1 F EH2 R IH0 NG",
}


@dataclass
class Syllable:
    """One sung syllable: the consonants before its vowel, the vowel(s), those after."""
    text: str
    onset: list[str] = field(default_factory=list)
    nucleus: list[str] = field(default_factory=list)
    coda: list[str] = field(default_factory=list)
    word: str = ""
    word_start: bool = True
    word_end: bool = True

    @property
    def phones(self):
        return [*self.onset, *self.nucleus, *self.coda]


def base(phone):
    """ARPAbet phone without its stress digit."""
    return phone.rstrip("012")


def is_vowel(phone):
    return base(phone) in VOWELS


@lru_cache(maxsize=1)
def cmudict():
    import cmudict as package
    return package.dict()


def clean_word(text):
    """Lower-case letters and inner apostrophes; LilyPond's elision ties become spaces."""
    text = text.casefold().replace("’", "'").replace("‿", " ").replace("_", " ").replace("~", " ")
    return " ".join(w.strip("'") if not w.endswith("in'") else w.lstrip("'")
                    for w in re.sub(r"[^a-z' ]+", " ", text).split() if w.strip("'"))


def _parse(spec):
    phones = spec.split() if isinstance(spec, str) else list(spec)
    for p in phones:
        if base(p) not in VOWELS | CONSONANTS:
            raise ValueError(f"unknown ARPAbet phone {p!r} in {spec!r}")
    return phones


def pronunciations(word, lexicon=None):
    """Candidate phone lists for one cleaned word, the preferred one first."""
    lexicon = {k.casefold(): v for k, v in (lexicon or {}).items()}
    if word in lexicon:
        spec = lexicon[word]
        return [_parse(s) for s in (spec if isinstance(spec, list) else [spec])]
    if word in BUILTIN:
        return [_parse(BUILTIN[word])]
    entries = cmudict().get(word)
    if entries:
        return [list(e) for e in entries]
    # Hymn and folk elisions: trav'ling, heav'n, ev'ry, pow'r take back their vowel.
    if "'" in word.strip("'"):
        for full in (word.replace("'", "e"), word.replace("'", "")):
            found = cmudict().get(full)
            if found:
                return [list(e) for e in found]
    # goin' -> going with the final N, then compounds and endings.
    if word.endswith("in'"):
        for stem in (word[:-1] + "g",):
            found = cmudict().get(stem)
            if found:
                return [[*e[:-1], "N"] if e[-1] == "NG" else list(e) for e in found]
    # Regular endings on a known stem; -s and -ed take their voicing from the stem's last sound.
    for suffix in ("es", "s", "ed", "ing", "ly", "er", "est"):
        if word.endswith(suffix):
            stem = cmudict().get(word[:-len(suffix)]) or cmudict().get(word[:-len(suffix)] + "e")
            if stem:
                return [[*stem[0], *_ending(suffix, stem[0][-1])]]
    raise ValueError(f"no pronunciation for {word!r}: add it to the piece's [voice3.lexicon] as "
                     "ARPAbet with stress digits, as CMUdict writes them (wayfaring = \"W EY1 F EH2 R IH0 NG\")")


VOICELESS = frozenset("P T K F TH S SH CH HH".split())
SIBILANTS = frozenset("S Z SH ZH CH JH".split())


def _ending(suffix, last):
    """The phones of a regular ending after a stem ending in `last`."""
    last = base(last)
    if suffix in ("s", "es"):
        return ["IH0", "Z"] if last in SIBILANTS else ["S"] if last in VOICELESS else ["Z"]
    if suffix == "ed":
        return ["IH0", "D"] if last in ("T", "D") else ["T"] if last in VOICELESS else ["D"]
    return {"ing": ["IH0", "NG"], "ly": ["L", "IY0"], "er": ["ER0"], "est": ["AH0", "S", "T"]}[suffix]


def _split(phones, count):
    """Divide one word's phones into `count` syllables by maximal onset."""
    nuclei = [i for i, p in enumerate(phones) if is_vowel(p)]
    if not nuclei:
        # No vowel: the consonants are the syllable's nucleus (a hum, a hiss).
        return [([], list(phones), [])] + [([], list(phones), []) for _ in range(count - 1)]
    # Group nuclei into `count` syllables: extra vowels join their neighbour
    # (the less stressed side), missing ones become melismatic repeats.
    groups = [[i] for i in nuclei]
    while len(groups) > count:
        stress = [max(int(phones[i][-1]) if phones[i][-1].isdigit() else 0 for i in g) for g in groups]
        # Merge the weakest group into its previous neighbour (or the next, at the start).
        weakest = min(range(len(groups)), key=lambda k: (stress[k], -k))
        into = weakest - 1 if weakest > 0 else 1
        lo, hi = sorted((weakest, into))
        groups[lo:hi + 1] = [groups[lo] + groups[hi]]
    out = []
    starts = []
    for k, g in enumerate(groups):
        if k == 0:
            starts.append(0)
            continue
        prev_end, here = groups[k - 1][-1], g[0]
        between = phones[prev_end + 1:here]
        take = 0
        for n in range(len(between), 0, -1):
            if tuple(base(p) for p in between[-n:]) in ONSETS:
                take = n
                break
        starts.append(here - take)
    starts.append(len(phones))
    for k, g in enumerate(groups):
        chunk = phones[starts[k]:starts[k + 1]]
        first, last = g[0] - starts[k], g[-1] - starts[k]
        out.append((chunk[:first], chunk[first:last + 1], chunk[last + 1:]))
    while len(out) < count:
        out.append(([], [p for p in out[-1][1] if is_vowel(p)][-1:], []))
    return out


def syllabify(events, lexicon=None):
    """Sung syllables for a lyric stream.

    events: (text, hyphen_after) pairs, one per lyric syllable in sung order.
    Returns one Syllable per event. An event's text may hold several words
    (an elision, ``to‿the``); they are sung on its note as one syllable whose
    nucleus runs through each word's vowels.
    """
    out = []
    i = 0
    while i < len(events):
        j = i
        while j < len(events) - 1 and events[j][1]:
            j += 1
        texts = [events[k][0] for k in range(i, j + 1)]
        joined = clean_word("".join(t.strip() for t in texts)) if len(texts) > 1 else clean_word(texts[0])
        words = joined.split()
        if not words:
            raise ValueError(f"lyric {' -- '.join(texts)!r} has no letters to sing")
        if len(texts) == 1 and len(words) > 1:
            # Several words on one note: concatenate their phones into one syllable.
            phones = [p for w in words for p in pronunciations(w, lexicon)[0]]
            onset, nucleus, coda = _merge_elision(phones)
            out.append(Syllable(texts[0], onset, nucleus, coda, word=" ".join(words)))
            i = j + 1
            continue
        word = words[0] if len(words) == 1 else "".join(words)
        candidates = pronunciations(word, lexicon)
        count = len(texts)
        fits = [c for c in candidates if sum(map(is_vowel, c)) == count]
        phones = fits[0] if fits else candidates[0]
        for k, (onset, nucleus, coda) in enumerate(_split(phones, count)):
            out.append(Syllable(texts[k], onset, nucleus, coda, word=word,
                                word_start=k == 0, word_end=k == count - 1))
        i = j + 1
    return out


def _merge_elision(phones):
    """One syllable from several words' phones: onset before the first vowel,
    coda after the last; everything between is the (compound) nucleus."""
    vowels = [i for i, p in enumerate(phones) if is_vowel(p)]
    if not vowels:
        return [], list(phones), []
    return phones[:vowels[0]], phones[vowels[0]:vowels[-1] + 1], phones[vowels[-1] + 1:]


def lyric_event(text):
    """Split a MIDI lyric text into (syllable, hyphen_after). The hyphen
    marker is written by scripts/lilypond/sung-lyrics.ily as a trailing ' --'."""
    text = text.strip()
    if text.endswith("--"):
        return text[:-2].strip(), True
    return text, False
