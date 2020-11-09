import sqlite3
import json
import pymorphy2


morph = pymorphy2.MorphAnalyzer()

ibrae_conn = sqlite3.connect('data/ibrae.db')
ibrae_cursor = ibrae_conn.cursor()

our_conn = sqlite3.connect('data/invIndex.sqlite')
our_cursor = our_conn.cursor()

kw = []
for (l,) in ibrae_cursor.execute('SELECT label FROM words'):
    if l is not None:
        kw.append(json.loads(l))



words = {}
for id, word in our_cursor.execute('SELECT id_word, word FROM Vocab'):
    words[word] = id


expert_words = []
for w in kw:
    for (definition, lang) in w:
        normal = morph.parse(definition)[0].normal_form
        expert_words.append(normal)


out = open('keywords.txt', 'w')
expert_words = set(expert_words)
for expert_word in expert_words:
    if expert_word in words:
        out.write('[{},"{}"]\n'.format(words[expert_word], expert_word))
out.close()


