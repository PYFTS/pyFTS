"""
Common facilities for hyperparameter optimization
"""

import sqlite3


def open_hyperparam_db(name):
    """
    Open a connection with a Sqlite database designed to store benchmark results.

    :param name: database filenem
    :return: a sqlite3 database connection
    """
    conn = sqlite3.connect(name)

    #performance optimizations
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    create_hyperparam_tables(conn)
    return conn


def create_hyperparam_tables(conn):
    """
    Create a sqlite3 table designed to store benchmark results.

    :param conn: a sqlite3 database connection
    """
    c = conn.cursor()

    c.execute('''CREATE TABLE if not exists hyperparam(
                 ID integer primary key, Date int, Dataset text, Tag text, 
                 Model text, Transformation text, mf text, 'Order' int, 
                 Partitioner text, Partitions int, alpha real, lags text, 
                 Measure text, Value real)''')

    conn.commit()


def insert_hyperparam(data, conn):
    """
    Insert benchmark data on database

    :param data: a tuple with the benchmark data with format:

    Dataset: Identify on which dataset the dataset was performed
    Tag: a user defined word that indentify a benchmark set
    Model: FTS model
    Transformation: The name of data transformation, if one was used
    mf: membership function
    Order: the order of the FTS method
    Partitioner: UoD partitioning scheme
    Partitions: Number of partitions
    alpha: alpha cut
    lags: lags
    Measure: accuracy measure
    Value: the measure value

    :param conn: a sqlite3 database connection
    :return:
    """
    c = conn.cursor()

    c.execute("INSERT INTO hyperparam(Date, Dataset, Tag, Model, "
              + "Transformation, mf, 'Order', Partitioner, Partitions, "
              + "alpha, lags, Measure, Value) "
              + "VALUES(datetime('now'),?,?,?,?,?,?,?,?,?,?,?,?)", data)
    conn.commit()
