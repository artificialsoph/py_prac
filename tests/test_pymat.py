from click.testing import CliRunner
from pymatrix import main


def assert_sub_out(command, output):
    runner = CliRunner()
    result = runner.invoke(main, command)
    assert result.exit_code == 0
    assert result.output == output


def test_import():
    """
    test all the input types using the sample data
    """
    mat_out = '[[1 2 3]\n [2 3 4]\n [1 1 1]]\n'
    input_list = [
        ['-f', 'data/sample.csv'],
        ['--csv-file', 'data/sample.csv'],
        ['-p', 'data/sample.p'],
        ['--pickle-file', 'data/sample.p'],
        ['-s', 'data/sample.coo'],
        ['--sparse-coo', 'data/sample.coo'],
        ['-j', 'data/sample.json'],
        ['--json-data', 'data/sample.json'],
    ]

    for in_pair in input_list:
        assert_sub_out(["print_mat"] + in_pair, mat_out)


def test_echo_json():
    assert_sub_out(["echo", '-j', 'data/sample.json', '2'],
                   ("\nThe given input was of type: json_data\n"
                    "And the value was: data/sample.json\n"
                    "\nThe given input was of type: json_data\n"
                    "And the value was: data/sample.json\n\n"))


def test_closest_to():
    assert_sub_out(["closest_to", '-j', "data/test.json", '8'], "1 8\n")
    assert_sub_out(["closest_to", '-j', "data/test.json", '8',
                    "--distance"],
                   "1 8 15.8113883008\n")


def test_closest():
    assert_sub_out(["closest", '-j', "data/test.json", '2'],
                   "6 7\n5 6\n")
    assert_sub_out(["closest", '-j', "data/test.json", '2', '--distance'],
                   "6 7 2.82842712475\n5 6 10.4403065089\n")


def test_furthest():
    assert_sub_out(["furthest", '-j', "data/test.json", '3'],
                   "7 8\n6 8\n5 8\n")
    assert_sub_out(["furthest", '-j', "data/test.json", '3', '--distance'],
                   "7 8 117.796434581\n6 8 114.978258814\n5 8 106.40018797\n")


def test_centroids():
    assert_sub_out(["centroids", '-j', "data/test.json", '3'],
                   "88.6666666667 18.6666666667\n46.25 37.5\n"
                   "11.3333333333 81.0\n")
