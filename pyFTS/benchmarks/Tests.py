
import numpy as np
import pandas as pd

from pyFTS.benchmarks.Measures import acf


def BoxPierceStatistic(data, h):
    """
    Q Statistic for Box-Pierce test

    :param data:
    :param h:
    :return:
    """
    n = len(data)
    s = 0
    for k in np.arange(1, h + 1):
        r = acf(data, k)
        s += r ** 2
    return n * s


def BoxLjungStatistic(data, h):
    """
    Q Statistic for Ljung–Box test

    :param data:
    :param h:
    :return:
    """
    n = len(data)
    s = 0
    for k in np.arange(1, h + 1):
        r = acf(data, k)
        s += r ** 2 / (n - k)
    return n * (n - 2) * s


def format_experiment_table(df, exclude=[], replace={}, csv=True, std=False):
    rows = []
    columns = []
    datasets = df.Dataset.unique()
    models = df.Model.unique()
    for model in models:
        test = np.any([model.rfind(k) != -1 for k in exclude]) if len(exclude) > 0 else False
        if not test:
            columns.append(model)

    for dataset in datasets:
        row = [dataset]
        if std:
            row_std = [dataset]
        for model in columns:
            avg = np.nanmin(df[(df.Dataset == dataset) & (df.Model == model)]["AVG"].values)
            row.append(round(avg, 3))
            if std:
                _std = np.nanmin(df[(df.Dataset == dataset) & (df.Model == model)]["STD"].values)
                row_std.append("(" + str(round(_std, 3)) + ")")

        rows.append(row)
        if std:
            rows.append(row_std)

    for k in range(len(columns)):
        if columns[k] in replace:
            columns[k] = replace[columns[k]]

    columns.insert(0, "dataset")

    if csv:
        header = ""
        for k in range(len(columns)):
            if k > 0:
                header += ","
            header += columns[k]

        body = ""
        for k in range(len(rows)):
            row = ""
            for w in range(len(rows[k])):
                if w > 0:
                    row += ","
                row += str(rows[k][w])
            body += '\n{}'.format(row)

        return header + body
    else:
        ret = pd.DataFrame(rows, columns=columns)
        return ret


def test_mean_equality(tests, alpha=.05, method='friedman'):
    """
    Test for the equality of the means, with alpha confidence level.

    H_0: There's no significant difference between the means
    H_1: There is at least one significant difference between the means

    :param tests:
    :param alpha:
    :param method:
    :return:
    """
    from stac.stac import nonparametric_tests as npt

    methods = tests.columns[1:]

    values = []
    for k in methods:
        values.append(tests[k].values)

    if method=='quade':
        f_value, p_value, rankings, pivots = npt.quade_test(*values)
    elif method=='friedman':
        f_value, p_value, rankings, pivots = npt.friedman_aligned_ranks_test(*values)
    else:
        raise Exception('Unknown test method!')

    print("F-Value: {} \tp-Value: {}".format(f_value, p_value))

    if p_value < alpha:
        print("\nH0 is rejected!\n")
    else:
        print("\nH0 is accepted!\n")

    post_hoc = {}
    rows = []
    for k in np.arange(0, len(methods)):
        rows.append([methods[k], rankings[k]])
        post_hoc[methods[k]] = pivots[k]

    return [pd.DataFrame(rows, columns=['METHOD', 'RANK']).sort_values(['RANK']), post_hoc]


def post_hoc_tests(post_hoc, control_method, alpha=.05, method='finner'):
    '''
    Finner paired post-hoc test with NSFTS as control method.

    $H_0$: There is no significant difference between the means

    $H_1$: There is a significant difference between the means

    :param post_hoc:
    :param control_method:
    :param alpha:
    :param method:
    :return:
    '''
    from stac.stac import nonparametric_tests as npt

    if method == 'bonferroni_dunn':
        comparisons, z_values, p_values, adj_p_values = npt.bonferroni_dunn_test(post_hoc,control_method)
    elif method == 'holm':
        comparisons, z_values, p_values, adj_p_values = npt.holm_test(post_hoc,control_method)
    elif  method == 'finner':
        comparisons, z_values, p_values, adj_p_values = npt.finner_test(post_hoc, control_method)
    else:
        raise Exception('Unknown test method!')

    rows = []
    for k in np.arange(len(comparisons)):
        test = 'H0 Accepted' if adj_p_values[k] > alpha else 'H0 Rejected'
        rows.append([comparisons[k], z_values[k], p_values[k], adj_p_values[k], test])

    return pd.DataFrame(rows, columns=['COMPARISON', 'Z-VALUE', 'P-VALUE', 'ADJUSTED P-VALUE', 'Result'])

