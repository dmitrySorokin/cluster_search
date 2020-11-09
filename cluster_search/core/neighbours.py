import json


def to_xy(cell_id, columns):
    row = int(cell_id / columns)
    column = cell_id - row * columns
    return row, column


def to_id(row, column, columns):
    return row * columns + column


def find_border(cluster1, cluster2, rows, columns):
    def neighbours(cell_id):
        row, column = to_xy(cell_id, columns)
        if row % 2 == 0:
            if row - 1 >= 0:
                if column - 1 >= 0:
                    yield to_id(row-1, column-1, columns)
                yield to_id(row - 1, column, columns)
            if column - 1 >= 0:
                yield to_id(row, column - 1, columns)
            if column + 1 < columns:
                yield to_id(row, column + 1, columns)
            if row + 1 < rows:
                if column - 1 >= 0:
                    yield to_id(row + 1, column - 1, columns)
                yield to_id(row + 1, column, columns)
        else:
            if row - 1 >= 0:
                if column + 1 < columns:
                    yield to_id(row-1, column+1, columns)
                yield to_id(row - 1, column, columns)
            if column - 1 >= 0:
                yield to_id(row, column - 1, columns)
            if column + 1 < columns:
                yield to_id(row, column + 1, columns)
            if row + 1 < rows:
                if column + 1 < columns:
                    yield to_id(row + 1, column + 1, columns)
                yield to_id(row + 1, column, columns)

    result = []
    for cell_id in cluster1:
        for neighbour in neighbours(cell_id):
            if neighbour in cluster2:
                result.append((cell_id, neighbour))
    return result

def find_neighbours(rows, columns, clusters):
    result = []
    for i, cluster in enumerate(clusters):
        for j in range(i + 1, len(clusters)):
            border_ij = find_border(cluster['data'], clusters[j]['data'], rows, columns)
            for neighbours in border_ij:
                result.append(neighbours)
    return result


if __name__ == '__main__':
    with open('data/som_config.json') as config:
        data = config.read()
        som_config = json.loads(data)
        clusters = som_config['clusters']
        ncolumns = som_config['map_columns']
        nrows = som_config['map_rows']


    print(clusters)
    print(ncolumns)
    res = find_neighbours(nrows, ncolumns, clusters)
    print(res)