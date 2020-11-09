from som import Som
from gensim.models import Word2Vec
import numpy as np
import sqlite3
import sklearn.cluster as clust
from scipy.cluster.hierarchy import dendrogram
from collections import defaultdict
import matplotlib as mpl
import matplotlib.cm as cm
import json
import scipy.io as io
import os
import csv
import itertools
from matplotlib import pyplot as plt
from copy import deepcopy
import pymorphy2
from matplotlib import pyplot as plt



CONNECTION = sqlite3.connect('data/invIndex.sqlite')
CURSOR = CONNECTION.cursor()

MODEL = Word2Vec.load('data/word2vec')

N_CLUSTERS = 12

with open('data/keywords.txt', 'r') as f:
    morph = pymorphy2.MorphAnalyzer()
    KEYWORDS = []
    for line in f.readlines():
        word_id, word = json.loads(line)
        KEYWORDS.append(word_id)

KEYWORDS = set(KEYWORDS)


WORDS = {}
for (id_word, word) in CURSOR.execute('SELECT id_word, word FROM Vocab'):
    WORDS[id_word] = word



def load_inv_index():
    result = CURSOR.execute(
        'SELECT InvIndex.id_word, id_parag, cnt, word FROM InvIndex JOIN Vocab ON InvIndex.id_word = Vocab.id_word;'
    ).fetchall()

    inv_index = defaultdict(dict)
    for (id_word, id_parag, cnt, word) in result:
        inv_index[word][id_parag] = cnt

    result = CURSOR.execute('SELECT id_parag, docLen FROM Texts').fetchall()
    parag_lengths = dict()
    for (id_parag, docLen) in result:
        parag_lengths[id_parag] = docLen

    return parag_lengths, inv_index



def bm25(query, parag_lengths, inv_index, id_parags_in_cluster):
    """
    sort id_parags_in_cluster by query
    :param query: ['word1', ..., 'wordN'] list of normalized words
    :param parag_lengths: {id_parag: int} where int is number of words in id_parag
    :param inv_index: {'word': {id_parag: cnt}}
    :param id_parags_in_cluster: set of id_parag which belongs to a cluster
    :return: sorted id_parags_in_cluster
    """

    # average document length
    avrdl = 0
    for id_parag in id_parags_in_cluster:
        avrdl += parag_lengths[id_parag]
    avrdl = avrdl / len(id_parags_in_cluster)

    # total number of documents
    N = len(id_parags_in_cluster)

    score = defaultdict(float)
    k1 = 2
    b = 0.75

    for word in query:
        word_count_in_parags = dict(filter(
            lambda kv: kv[0] in id_parags_in_cluster, inv_index[word].items()
        ))

        # number of documents containing word
        nq = len(word_count_in_parags)
        if nq == 0: continue

        # inverse document frequency
        IDFq = np.log((N - nq + 0.5) / (nq + 0.5))

        if IDFq <= 0: continue

        for id_parag in word_count_in_parags:
            # term frequency of word in id_parag
            fqD = word_count_in_parags[id_parag]

            denominator = fqD + k1 * (1 - b + b * parag_lengths[id_parag] / avrdl)

            if denominator <= 0: continue

            score[id_parag] += IDFq * fqD * (k1 + 1) / denominator

    result = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    return map(lambda kv: kv[0], result)


def count_words():
    """
    count number of times each word occurs in database
    :return: {'word': int}
    """

    words = defaultdict(int)

    result = CURSOR.execute('SELECT id_word, cnt FROM InvIndex').fetchall()

    for (id_word, cnt) in result:
        words[id_word] += cnt

    return words


def count_words_in_cluster(id_parags_in_cluster):
    """
    count number of times each word from cluster occurs in database
    :param id_parags_in_cluster: [id_parag1, id_parag2, ..., id_paragN]
    :return: {'word': int}
    """
    words = defaultdict(int)

    range = ', '.join('{}'.format(k) for k in id_parags_in_cluster)
    result = CURSOR.execute(
        'SELECT id_word, cnt FROM InvIndex WHERE id_parag IN ({})'.format(range)
    ).fetchall()

    for (id_word, cnt) in result:
        words[id_word] += cnt

    return words


def idf(id_parags_in_cluster):
    """
    count number of times each word from cluster occurs in database
    :param id_parags_in_cluster: [id_parag1, id_parag2, ..., id_paragN]
    :return: {'word': int}
    """
    parags = defaultdict(set)

    range = ', '.join('{}'.format(k) for k in id_parags_in_cluster)
    result = CURSOR.execute(
        'SELECT id_word, cnt, id_parag FROM InvIndex WHERE id_parag IN ({})'.format(range)
    ).fetchall()



    for (id_word, cnt, id_parag) in result:
        parags[id_word].add(id_parag)


    counts = {id_word: len(parags) for id_word in parags}
    total = len(id_parags_in_cluster)

    def res(id_word):
        assert id_word in counts
        return np.log(total / counts[id_word])

    return res


def principal_words(total_words, words_in_cluster, idf, n_words):
    """
    yield n principal words among words_in_cluster
    :param total_words: {'word': int}
    :param words_in_cluster: {'word': int}
    :param n_words: number of words to generate
    :return: yields (word, score)
    """


    n_words = min(n_words, len(words_in_cluster))
    total_in_cluster = sum([words_in_cluster[k] for k in words_in_cluster])

    true_score = []
    for word in words_in_cluster:
        if word in KEYWORDS:
            true_score.append(
                (word, words_in_cluster[word])
            )
    true_score = sorted(true_score, key=lambda kv: kv[1], reverse=True)



    our_score = []
    tfidf_score = []
    tf_score = []
    for word in words_in_cluster.keys():
        tf = float(words_in_cluster[word]) / total_in_cluster
        our_score.append(
            (word, float(words_in_cluster[word]) ** 2 / total_words[word])
        )
        tf_score.append(
            (word, tf)
        )
        tfidf_score.append(
            (word, tf * idf(word))
        )
    our_score = sorted(our_score, key=lambda kv: kv[1], reverse=True)
    tf_score = sorted(tf_score, key=lambda kv: kv[1], reverse=True)
    tfidf_score = sorted(tfidf_score, key=lambda kv: kv[1], reverse=True)


    i = 0
    for (id_word, word_score), \
        (true_id_word, true_score), \
        (tf_id_word, tf_score), \
        (tfidf_word, tfidf_score) \
            in zip(our_score, true_score, tf_score, tfidf_score):
        if i == n_words: break
        yield \
            (WORDS[id_word], word_score), \
            (WORDS[true_id_word], true_score), \
            (WORDS[tf_id_word], tf_score), \
            (WORDS[tfidf_word], tfidf_score)
        i += 1


def parag_vectors():
    """
    return all paragraph ids and vectors from database
    :return: ([id_parag1,..., id_paragN], [vector1, ..., vectorN])
    """
    result = CURSOR.execute('SELECT id_parag, doc2vec_dm0 FROM Texts').fetchall()
    #####
    # use vector instead of doc2vec_dm0, doc2vec_dm1
    ####

    id_parags = []
    vectors = []
    for (id, vector) in result:
        id_parags.append(id)

        # TODO store vectors as vectors not as strings
        vector = list(map(float, vector.split()))
        vectors.append(vector)

    return id_parags, np.array(vectors)


def build_som(data, train_steps):
    """
    Build self organized map
    :param data: np.array
    :param train_steps: int
    :return: som
    """

    n_cells = np.sqrt(5 * len(data))
    #n_cells = 20
    print('n_cells = ', n_cells)
    smap = Som(data, n_cells)
    smap.train(data, train_steps)

    return smap


def load_or_build_som(data):
    if not os.path.isfile('data/smap.mat'):
        print('building som...')
        smap = build_som(data, 50)

        codebook = smap.codebook

        # number of cells in rows/columns of the map
        map_shape = smap.mshape

        # winner[i] = cell_id, where i is input data index
        winner, distance = smap.winner(data)

        print('saving som...')
        io.savemat('data/smap.mat', mdict={
            'codebook': codebook,
            'map_shape': map_shape,
            'winner': winner,
            'distance': distance
        })
    else:
        print('loading som...')
        mdict = io.loadmat('data/smap')
        codebook = mdict['codebook']

        # [0] because vectors are saved as matrix [1][n]
        map_shape = mdict['map_shape'][0]
        winner = mdict['winner'][0]
        distance = mdict['distance'][0]

    # convert int64 to int
    map_shape = tuple(map(int, map_shape))

    return codebook, map_shape, winner, distance


def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def cluster(codebook, n_clusters=8):
    """

    :param codebook: np.array n_cells x len(vector)
    :param n_clusters: number of clusters
    :return: [cluster_id_for_cell1, cluster_id_for_cell2, ..., cluster_id_for_cellN]
    """
    model = clust.AgglomerativeClustering(n_clusters=2)
    cl_labels = model.fit_predict(codebook)
    print(len(codebook), len(model.children_))
    exit(0)
    #cl_labels = clust.KMeans(n_clusters=n_clusters, random_state=0).fit_predict(codebook)
    ii = itertools.count(codebook.shape[0])
    r = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_]
    
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, labels=model.labels_)
    plt.show()

    return cl_labels


def color_by_vector_component(codebook, component):
    """
    generate colors for each cell based on value of the vector component
    :param codebook: np.array n_cells x len(vector)
    :param component: vector component to color by
    :return: yields color (i.e. #FFFFFF) for each cell
    """
    ncells = codebook.shape[0]
    vmin = codebook[0][component]
    vmax = codebook[0][component]
    for i in range(1, ncells):
        val = codebook[i][component]
        if val < vmin:
            vmin = val
        if val > vmax:
            vmax = val
    print('vmin, vmax ', vmin, vmax)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.hot

    for i in range(ncells):
        val = codebook[i][component]
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        r, g, b, a = m.to_rgba(val)
        yield '#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255))


def color_by_cluster_id(clusters, size):
    """
    :param clusters: {cluster_id: [cell1, cellN]}
    :return: yields color (i.e. #FFFFFF) for each cell based on culster_id
    """

    colors = [
        "#cc4767", "#6f312a", "#d59081", "#d14530", "#d27f35",
        "#887139", "#d2b64b", "#c7df48", "#c0d296", "#5c8f37",
        "#364d26", "#70d757", "#60db9e", "#4c8f76", "#75d4d5",
        "#6a93c1", "#616ed0", "#46316c", "#8842c4", "#bc87d0"
    ]

    n_colors = len(colors)
    result = [None] * size

    if len(clusters.keys()) > n_colors:
        print('WARNING: only {} distinct colors supported'.format(n_colors))

    for cluster_id in clusters:
        i = cluster_id % n_colors
        for cell_id in clusters[cluster_id]:
            result[cell_id] = colors[i]

    return result


def main():
    parag_lengths, inv_index = load_inv_index()

    def sort_parags_in_cluster(query, id_parags_in_cluster, number):
        res = bm25(query, parag_lengths, inv_index, set(id_parags_in_cluster))
        return list(res)[:number]

    print('reading parag vectors...')
    id_parags, vectors = parag_vectors()

    codebook, map_shape, winner, distance = load_or_build_som(vectors)

    print('codebook.shape = ', codebook.shape)
    print('map_shape = ', map_shape)

    def parags_in_cell(cell_id):
        """
        return [(id_parag, distance)] which belongs to the cell_id
        :param cell_id: id of a cell (i.e. 0, 1, ..., N)
        :return: [(id_parag1, distance1), ..., (id_paragN, distanceN)]
        """
        id_parags_with_distance = []
        for ind, id_parag in enumerate(id_parags):
            #print(winner[ind], type(winner[ind]), winner.shape, ' ', distance.shape)
            if int(winner[ind]) != cell_id: continue
            id_parags_with_distance.append((id_parag, distance[ind]))
        return id_parags_with_distance


    som_config = {
        'map_columns': map_shape[0],
        'map_rows': map_shape[1],
        'data': [],
    }

    docs_config = {}

    print('clustering...')
    size = codebook.shape[0]
    model = clust.AgglomerativeClustering(n_clusters=None, distance_threshold=0)
    model.fit(codebook)
    children = model.children_
    distances = model.distances_

    #print(len(children), len(distances))
    #assert size == len(children), '{} != {}'.format(size, len(children))

    #print(distances.shape, children.shape)
    #print(distances)
    #plt.plot(distances, marker='.')
    #plt.xlabel('step')
    #plt.ylabel('distance')
    #plt.show()
    #exit(0)

    clusters = {i: [i] for i in range(size)}
    pwords = defaultdict(list)
    true_pwords = defaultdict(list)
    tf_pwords = defaultdict(list)
    tfidf_pwords = defaultdict(list)
    total_words = count_words()
    parags_in_cluster = {}

    for zoom in range(size - 1):
        print('ZOOM = ', zoom)

        print(len(clusters))

        # FIXME
        cell_colors = list(color_by_cluster_id(clusters, size))
        #for i, color in enumerate(cell_colors):
        #    print('cell_id = ', i, 'color = ', color)


        print('calculating parags in cluster...')
        #id_parags in cluster sorted by distance
        for label in clusters:
            if label in parags_in_cluster:
                continue
            id_parags_with_distance_in_cluster = []
            for cell_id in clusters[label]:
                id_parags_with_distance_in_cell = parags_in_cell(cell_id)
                id_parags_with_distance_in_cluster.extend(id_parags_with_distance_in_cell)
            #id_parags_with_distance_in_cluster = sorted(id_parags_with_distance_in_cluster, key=lambda kv: kv[1])
            parags_in_cluster[label] = list(map(lambda kv: kv[0], id_parags_with_distance_in_cluster))

        # {cluster_id : ['word1', ..., 'wordN']}
        for label in clusters:
            if label in pwords:
                continue

            words_in_cluster = count_words_in_cluster(parags_in_cluster[label])
            print('principal words for cluster_id = ', label)
            words = principal_words(total_words, words_in_cluster, idf(parags_in_cluster[label]), n_words=10)
            for \
                    (word, word_score), \
                    (true_word, true_score), \
                    (tf_word, tf_score), \
                    (tfidf_word, tfidf_score) \
                    in words:
                print('word = {}, score = {}; true_word = {}, true_score = {}'.format(
                    word, word_score, true_word, true_score))
                pwords[label].append(word)
                true_pwords[label].append(true_word)
                tf_pwords[label].append(tf_word)
                tfidf_pwords[label].append(tfidf_word)
            print('*' * 20)


        som_config['data'].append({
            'colors': cell_colors,
            'clusters': [],
            'dist': distances[zoom]
        })



        for label in clusters:
            cluster_id = 'cluster_{}_{}'.format(zoom, label)

            som_config['data'][zoom]['clusters'].append({
                'cell_ids': clusters[label],
                'label': ', '.join(pwords[label]),
                'true_label': ', '.join(true_pwords[label]),
                'tf_label': ', '.join(tf_pwords[label]),
                'tfidf_label': ', '.join(tfidf_pwords[label]),
                'id': cluster_id
            })

            docs_config[cluster_id] = {
                'parag_ids': sort_parags_in_cluster(
                    pwords[label], parags_in_cluster[label], 100)
            }

        if zoom == size - 1:
            break

        left = deepcopy(clusters[children[zoom][0]])
        right = deepcopy(clusters[children[zoom][1]])
        del clusters[children[zoom][0]]
        del clusters[children[zoom][1]]

        clusters[zoom + size] = left
        clusters[zoom + size].extend(right)


    with open('data/som_config.json', 'w') as output:
        output.write(json.dumps(som_config))

    with open('data/docs_config.json', 'w') as output:
        output.write(json.dumps(docs_config))


    def cluster_to_csv(cluster_id):
        with open('cluster_{}.csv'.format(cluster_id), 'w', encoding='utf-8') as outcsv:
            outcsv.write('\ufeff') #tell excel this is unicode file
            writer = csv.writer(outcsv, delimiter=';')
            writer.writerow(['id_file', 'name', 'id_parag', 'parag', 'page_number'])
            id_ps = sort_parags_in_cluster(pwords[cluster_id], parags_in_cluster[cluster_id], None)
            range = ', '.join('{}'.format(k) for k in id_ps)
            result = CURSOR.execute(
                'SELECT Texts.id_file, name, id_parag, parag, page_number FROM Texts '
                'JOIN Files ON Texts.id_file = Files.id_file '
                'WHERE id_parag IN ({}) '.format(range)
            ).fetchall()

            for id_file, name, id_parag, parag, page_number in result:
                writer.writerow([id_file, name, id_parag, parag, page_number])

    # usage
    # cluster_co_csv(6)


    CONNECTION.commit()
    CURSOR.close()
    CONNECTION.close()

if __name__ == '__main__':
    main()
