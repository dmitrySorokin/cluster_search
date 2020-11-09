import sqlite3
import json
from jsonschema import validate


def validate_som_config(config):
    schema = {
        'type': 'object',
        'properties': {
            'map_columns': {
                'type': 'integer'
            },
            'map_rows': {
                'type': 'integer'
            },
            'data': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'colors': {
                            'type': 'array',
                            'items': {
                                'type': 'string'
                            }
                        },
                        'clusters': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'cell_ids': {
                                        'type': 'array',
                                        'items': {
                                            'type': 'integer'
                                        }
                                    },
                                    'id': {
                                        'type': 'string'
                                    },
                                    'label': {
                                        'type': 'string'
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    validate(config, schema)

    """
    expected format:
    {
        'map_columns': int,
        'map_rows': int,
        'data': [{                               #each item corresponds to each zoom 
            'colors': [str, str, ..., str],
            'clusters': [{
                'cell_ids': [int, int, ..., int] #where int is cell_id = i + j * columns
                'label': str
                'id': str
            }]
        }]
    }
    """
    columns = config['map_columns']
    rows = config['map_rows']
    print('validate som_config: rows = {}, columns = {}'.format(rows, columns))

    for zoom_config in config['data']:
        colors = zoom_config['colors']
        clusters = zoom_config['clusters']
        print(len(colors))
        print(len(clusters))

        ncells = rows * columns
        if ncells != len(colors):
            raise ValueError('ncells {} != len(colors) {}'.format(ncells, len(colors)))

        def clusters_len():
            result = 0
            for cluster in clusters:
                result += len(cluster['cell_ids'])
                print('len of {} is {}'.format(cluster['id'], len(cluster['cell_ids'])))
            return result

        if ncells != clusters_len():
            raise ValueError('ncells {} != ncells in clusters {}'.format(ncells, clusters_len()))

        def in_clusters(cell_id):
            for cluster in clusters:
                if cell_id in cluster['cell_ids']:
                    return True
            return False

        for i in range(rows):
            for j in range(columns):
                cell_id = j + i * columns
                if not in_clusters(cell_id):
                    raise ValueError('cell_id {} not found in clusters'.format(cell_id))


def build_docs_config(json_config):
    """"
    json_config = {
        'cluster_id': [int, int, ..., int] # where int is id_parag from invIndex.sqlite
    }
    """

    docs_config = {}
    docs_path = {}

    conn = sqlite3.connect('cluster_search/core/data/invIndex.sqlite')
    cursor = conn.cursor()
    for cluster_id in json_config:
        docs_config[cluster_id] = []
        parag_ids = json_config[cluster_id]['parag_ids']

        parags_range = ', '.join(map(str, parag_ids))
        result = cursor.execute(
            'SELECT Texts.id_file, name, page_number, parag FROM Texts '
            'JOIN Files ON Texts.id_file = Files.id_file '
            'WHERE id_parag IN ({}) '.format(parags_range)
        ).fetchall()

        for id_file, name, page_number, parag in result:
            item = {
                'id_file': id_file,
                'name': name,
                'page': page_number,
                'parag': parag
            }

            json_data = json.dumps(item)
            docs_config[cluster_id].append(json_data)

            # TODO store file path in database
            docs_path[id_file] = '/static/js/external/pdfjs/web/compressed.tracemonkey-pldi-09.pdf'

    conn.commit()
    cursor.close()
    conn.close()

    return docs_config, docs_path


def get_docs_config(cluster_id, parag_ids):
    """"
    json_config = {
        'cluster_id': [int, int, ..., int] # where int is id_parag from invIndex.sqlite
    }
    """

    docs_config = {}
    docs_path = {}

    conn = sqlite3.connect('cluster_search/core/data/invIndex.sqlite')
    cursor = conn.cursor()
    docs_config[cluster_id] = []

    parags_range = ', '.join(map(str, parag_ids))
    result = cursor.execute(
        'SELECT Texts.id_file, name, page_number, parag FROM Texts '
        'JOIN Files ON Texts.id_file = Files.id_file '
        'WHERE id_parag IN ({}) '.format(parags_range)
    ).fetchall()

    for id_file, name, page_number, parag in result:
        item = {
            'id_file': id_file,
            'name': name,
            'page': page_number,
            'parag': parag
        }

        json_data = json.dumps(item)
        docs_config[cluster_id].append(json_data)        

    conn.commit()
    cursor.close()
    conn.close()

    return docs_config


class ClusterSearchApi:
    def __init__(self):
        with open('cluster_search/core/data/som_config.json') as config:
            self._som_config = config.read()
            #validate_som_config(json.loads(self._som_config))
            conf = json.loads(self._som_config)
            conf['data'] = conf['data'][-100:]
            self._som_config = json.dumps(conf)

        with open('cluster_search/core/data/docs_config.json') as config:
            self._docs_config = json.loads(config.read())
        #    self._docs_config, self._docs_path = build_docs_config(config)

    def som_config(self):
        return self._som_config

    def pdf_links(self, cluster_id):
        """
        :param cluster_id: string
        :return: [json1, json2, ..., jsonN] or None if cluster_id is wrong
        """
        config = get_docs_config(cluster_id, self._docs_config.get(cluster_id).get('parag_ids'))
        return config.get(cluster_id)

    def path_to_file(self, id_file):
        """
        :param id_file: int
        :return: string or None if id_file is wrong
        """
        return '/static/js/external/pdfjs/web/compressed.tracemonkey-pldi-09.pdf'
