import json
import csv
import io
import re


# Common OCR character confusions: (wrong, correct)
OCR_CONFUSIONS = [
    ('0', 'O'), ('O', '0'),
    ('1', 'l'), ('l', '1'), ('1', 'I'), ('I', '1'),
    ('5', 'S'), ('S', '5'),
    ('8', 'B'), ('B', '8'),
    ('6', 'G'), ('G', '6'),
    ('rn', 'm'), ('cl', 'd'),
    ('vv', 'w'), ('VV', 'W'),
]


def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


class SpellCorrector:
    def __init__(self, dictionary=None, max_distance=2):
        self.max_distance = max_distance
        if dictionary is None:
            self.dictionary = self._default_dictionary()
        else:
            self.dictionary = set(w.lower() for w in dictionary)

    def _default_dictionary(self):
        # Small built-in word list for common document words
        words = (
            "the a an and or not is are was were be been being have has had "
            "do does did will would could should may might shall can "
            "of in to for on with at by from as into through about "
            "this that these those it its they them their he she we you "
            "all any both each few more most other some such no only "
            "same so than too very just now also back after before "
            "name date address city state zip code total amount due "
            "invoice receipt number quantity price tax subtotal "
            "january february march april may june july august "
            "september october november december "
            "monday tuesday wednesday thursday friday saturday sunday "
            "phone email website fax mobile tel "
            "yes no true false "
        ).split()
        return set(words)

    def load_dictionary(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.dictionary = set(line.strip().lower() for line in f if line.strip())

    def correct_word(self, word):
        word_lower = word.lower()
        if word_lower in self.dictionary or not word.isalpha():
            return word

        best_word = word
        best_dist = self.max_distance + 1
        for dict_word in self.dictionary:
            if abs(len(dict_word) - len(word)) > self.max_distance:
                continue
            dist = levenshtein_distance(word_lower, dict_word)
            if dist < best_dist:
                best_dist = dist
                best_word = dict_word

        if best_dist <= self.max_distance:
            # Preserve original capitalisation
            if word[0].isupper():
                return best_word.capitalize()
            return best_word
        return word

    def correct_text(self, text):
        tokens = re.findall(r'\w+|\W+', text)
        corrected = []
        for token in tokens:
            if re.match(r'^\w+$', token):
                corrected.append(self.correct_word(token))
            else:
                corrected.append(token)
        return ''.join(corrected)


class ContextVerifier:
    """Apply rule-based context corrections for common OCR errors."""

    def __init__(self):
        self._build_rules()

    def _build_rules(self):
        self.rules = [
            # Digits-only context: replace letter lookalikes
            (r'\b([0-9]+)[Oo]([0-9]+)\b', lambda m: m.group(0).replace('O', '0').replace('o', '0')),
            (r'\b([0-9]+)[lI]([0-9]+)\b', lambda m: m.group(0).replace('l', '1').replace('I', '1')),
            (r'\b([0-9]+)[Ss]([0-9]*)\b', lambda m: m.group(0).replace('S', '5').replace('s', '5')),
            # Word context: replace digit lookalikes
            (r'\b([a-zA-Z]+)0([a-zA-Z]+)\b', lambda m: m.group(0).replace('0', 'o')),
            (r'\b([a-zA-Z]+)1([a-zA-Z]+)\b', lambda m: m.group(0).replace('1', 'l')),
            # Fix double-space
            (r'  +', ' '),
            # Fix broken words at end of line
            (r'-\n([a-z])', r'\1'),
        ]

    def apply(self, text):
        result = text
        for pattern, repl in self.rules:
            if callable(repl):
                result = re.sub(pattern, repl, result)
            else:
                result = re.sub(pattern, repl, result)
        return result


class OutputFormatter:
    def to_plain_text(self, words_per_line):
        """words_per_line: list of lists of words (lines of words)"""
        lines = [' '.join(line) for line in words_per_line]
        return '\n'.join(lines)

    def to_json(self, structured_data):
        return json.dumps(structured_data, indent=2, ensure_ascii=False)

    def to_csv(self, rows, fieldnames=None):
        buf = io.StringIO()
        if not rows:
            return ''
        if fieldnames is None:
            fieldnames = list(rows[0].keys()) if isinstance(rows[0], dict) else None
        if fieldnames:
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(buf)
            writer.writerows(rows)
        return buf.getvalue()

    def save(self, content, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def words_to_structured(self, words, regions=None):
        """Convert a flat list of words into a structured dict."""
        data = {
            'word_count': len(words),
            'text': ' '.join(words),
            'words': words,
        }
        if regions:
            data['regions'] = [
                {'type': r['type'], 'bbox': list(r['bbox'])}
                for r in regions
            ]
        return data


class PostProcessor:
    """Facade combining spell correction, context verification, and formatting."""

    def __init__(self, dictionary=None, spell_max_distance=2):
        self.spell_corrector = SpellCorrector(
            dictionary=dictionary, max_distance=spell_max_distance
        )
        self.context_verifier = ContextVerifier()
        self.formatter = OutputFormatter()

    def process(self, raw_text):
        text = self.context_verifier.apply(raw_text)
        text = self.spell_corrector.correct_text(text)
        return text

    def process_words(self, words):
        return [self.spell_corrector.correct_word(w) for w in words]

    def format_output(self, words, regions=None, fmt='json', spell_correct=False):
        # Spell correction is OFF by default: the built-in dictionary is tiny,
        # so "correcting" real OCR output corrupts valid words
        # (e.g. "Kandy"->"And", "Road"->"Had"). Only enable it with a proper
        # domain dictionary loaded via SpellCorrector.load_dictionary().
        corrected_words = self.process_words(words) if spell_correct else list(words)
        structured = self.formatter.words_to_structured(corrected_words, regions)
        if fmt == 'json':
            return self.formatter.to_json(structured)
        elif fmt == 'txt':
            return structured['text']
        elif fmt == 'csv':
            rows = [{'index': i, 'word': w} for i, w in enumerate(corrected_words)]
            return self.formatter.to_csv(rows)
        else:
            raise ValueError(f"Unknown format: {fmt}")
