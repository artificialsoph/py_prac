import pandas
import numpy
import scipy
import scipy.sparse

"""
Util functions for parsing data inputs
"""


def parse_options(params):
    keys = ["json_data", "csv_file", "pickle_file", "sparse_coo"]
    opts = [(k, v) for k, v in params.items()
            if (k in keys) and (v is not None)]
    if len(opts) != 1:
        raise ValueError("Exactly one input data type must be provided, got "
                         " {}".format(len(opts)))
    return opts[0]


def import_pickle(mat_path):
    return numpy.array(pandas.read_pickle(mat_path))


def import_csv(mat_path):
    return pandas.read_csv(mat_path, header=None).as_matrix()


def import_json(mat_path):
    return pandas.read_json(mat_path).as_matrix()


def import_coo(mat_path):
    input_arr = pandas.read_table(mat_path, sep=' ', header=None).as_matrix()
    data, i, j = input_arr[:,2], input_arr[:,0], input_arr[:,1]
    return scipy.sparse.csr_matrix((data, (i, j)))


def import_mat(ctx):
    input_type, value = parse_options(ctx.params)
    import_fun = {
        'csv_file': import_csv,
        'pickle_file': import_pickle,
        'json_data': import_json,
        'sparse_coo': import_coo
    }[input_type]
    return import_fun(value)
