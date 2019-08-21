import numpy
import click
from utils import parse_options, import_mat
import scipy.sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans

mat_decorators = [
    click.option('-j', '--json-data', type=click.STRING,
                 help='input matrix as a valid json object or as text'),
    click.option('-f', '--csv-file', type=click.Path(exists=True),
                 help='read matrix from a csv file'),
    click.option('-p', '--pickle-file', type=click.Path(exists=True),
                 help='read matrix from a pickle file'),
    click.option('-s', '--sparse-coo', type=click.Path(exists=True),
                 help='read matrix in COO format from a file'),
    click.pass_context
]


def mat_decs(f):
    for dec in mat_decorators:
        f = dec(f)
    return f


dist_decorators = [
    click.option('-m', '--metric', default='euclidean', type=click.STRING,
                 help="distance metric used."),
    click.option('--distance', default=False, is_flag=True,
                 type=click.BOOL, help='print the distance between the pair of rows'),
]


def dist_decs(f):
    for dec in dist_decorators:
        f = dec(f)
    return f


@click.group()
def main():
    """
    pymatrix: A command line tool for working with matrices.
    """
    # print(os.getcwd())
    pass


@main.command()
@mat_decs
def print_mat(ctx, *args, **kwargs):
    """
    print's the supplied matrix
    """
    input_type, value = parse_options(kwargs)
    mat = import_mat(ctx)
    if isinstance(mat, scipy.sparse.spmatrix):
        mat = mat.todense()
    click.echo(mat)


@main.command()
@mat_decs
@click.argument('n', type=click.INT, default=1)
def echo(ctx, n, *args, **kwargs):
    """
    Display the passed options N times
    """
    input_type, value = parse_options(kwargs)
    click.echo(
        "\nThe given input was of type: {}\n"
        "And the value was: {}\n".format(input_type, value) * n)


def print_pair(x, y, dist=None):
    """helper funtion for printing a pair of indices"""
    x, y = numpy.sort((x, y))
    if dist:
        click.echo("{0} {1} {2}".format(x, y, dist))
    else:
        click.echo("{0} {1}".format(x, y))


@main.command()
@mat_decs
@dist_decs
@click.argument('row_i', type=click.INT)
def closest_to(ctx, row_i, distance, *args, **kwargs):
    """
    Find the row that is the minimal distance from row_i and
    optionally display the distance as well

    Output Format:\n
      i j [d_ij]
    """

    mat = import_mat(ctx)
    dist = pairwise_distances(mat, mat[row_i].reshape(1, -1),
                              metric=ctx.params['metric'])
    dist[row_i] = numpy.inf
    min_i = numpy.argmin(dist)
    if distance:
        print_pair(row_i, min_i, dist.item(min_i))
    else:
        print_pair(row_i, min_i)


@main.command()
@mat_decs
@dist_decs
@click.argument('n', default=1, type=click.INT)
def closest(ctx, n, distance, *args, **kwargs):
    """
    Find the N distinct pairs of rows that are the smallest distance
    apart and optionally display the distance as well

    Output Format:\n
      i j [d_ij]
    """

    mat = import_mat(ctx)
    dist = pairwise_distances(mat, metric=ctx.params['metric'])
    t_x, t_y = numpy.tril_indices_from(dist, k=0)
    dist[t_x, t_y] = numpy.inf
    min_x, min_y = numpy.unravel_index(
        dist.argsort(axis=None), dims=dist.shape)
    for i in range(n):
        x, y = min_x[i], min_y[i]
        if distance:
            print_pair(x, y, dist[x, y])
        else:
            print_pair(x, y)


@main.command()
@mat_decs
@dist_decs
@click.argument('n', default=1, type=click.INT)
def furthest(ctx, n, distance, *args, **kwargs):
    """
    Find the N distinct pairs of rows that are the furthest distance
    apart and optionally display the distance as well

    Output Format:\n
      i j [d_ij]
    """

    mat = import_mat(ctx)
    dist = pairwise_distances(mat, metric=ctx.params["metric"])
    t_x, t_y = numpy.tril_indices_from(dist, k=0)
    dist[t_x, t_y] = 0
    min_x, min_y = numpy.unravel_index(
        dist.argsort(axis=None), dims=dist.shape)
    for i in range(1, n + 1):
        x, y = min_x[-i], min_y[-i]
        if distance:
            print_pair(x, y, dist[x, y])
        else:
            print_pair(x, y)


@main.command()
@mat_decs
@click.argument('n', default=1, type=click.INT)
def centroids(ctx, n, *args, **kwargs):
    """
    Cluster the given data set and return the N centroids,
    one for each cluster
    """

    mat = import_mat(ctx)
    model = KMeans(n_clusters=n).fit(mat)
    # sorting better supports testing
    sort_i = numpy.lexsort(model.cluster_centers_.T)
    centroids = model.cluster_centers_[sort_i, :]
    for c in centroids:
        print(" ".join([str(i) for i in c]))


if __name__ == '__main__':
    main(obj={})
