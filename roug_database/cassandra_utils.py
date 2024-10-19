"""
Cassandra Database Utility Functions

This script provides utility functions for interacting with a Cassandra database.
It includes functions for establishing a connection to a Cassandra cluster,
creating keyspaces, and creating tables.

These utilities are designed to simplify common Cassandra operations in Python applications,
making it easier to set up and manage Cassandra databases.

Key functionalities:
- Establish a connection to a Cassandra cluster
- Create a new keyspace in the Cassandra database
- Create a new table within a keyspace

Usage:
This script can be imported and used in other Python scripts that require
Cassandra database operations.
"""
from datetime import datetime

from cassandra.cluster import Cluster
from cassandra.util import Date


def get_cassandra_session(keyspace: str = None) -> Cluster.connect:
    """
    Establish a connection to a Cassandra cluster and optionally connect to a keyspace.

    :param keyspace: Name of the keyspace to connect to (optional)
    :return: Cassandra session object
    """
    cluster = Cluster(['localhost'])
    if keyspace:
        return cluster.connect(keyspace)
    return cluster.connect()

def create_keyspace(session: Cluster.connect, keyspace_name: str, replication_strategy: str = 'SimpleStrategy',
                    replication_factor: int = 1) -> None:
    """
    Create a new keyspace in the Cassandra database if it doesn't already exist.

    :param session: Cassandra session object
    :param keyspace_name: Name of the keyspace to create
    :param replication_strategy: Replication strategy for the keyspace (default: 'SimpleStrategy')
    :param replication_factor: Replication factor for the keyspace (default: 1)
    """
    session.execute(f"""
    CREATE KEYSPACE IF NOT EXISTS {keyspace_name}
    WITH replication = {{
        'class': '{replication_strategy}',
        'replication_factor': {replication_factor}
    }}
    """)

def create_table(session: Cluster.connect, table_name: str, schema: str) -> None:
    """
    Create a new table in the Cassandra database if it doesn't already exist.

    :param session: Cassandra session object
    :param table_name: Name of the table to create
    :param schema: Schema definition for the table (e.g., "id UUID PRIMARY KEY, name TEXT")
    """
    session.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})")


def convert_cassandra_date(cassandra_date: Date) -> datetime.date:
    """
    Convert Cassandra Date object to Python date object.

    :param cassandra_date: Cassandra Date object to convert
    :return: Converted Python date object
    """
    if isinstance(cassandra_date, Date):
        return cassandra_date.date()
    return cassandra_date
