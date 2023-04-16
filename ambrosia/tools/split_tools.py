#  Copyright 2022 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Methods for group splitting tasks.
"""

import hashlib
import os
from base64 import b64encode
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import ambrosia.tools.stratification as strat_pkg
from ambrosia import types
from ambrosia.tools.knn import NMTree

GROUPS_COLUMNS: str = "group"
AVAILLABLE_SPLIT_METHODS = ["simple", "hash", "metric", "dim_decrease"]


def check_ids_duplicates(
    dataframe: pd.DataFrame,
    id_column: Optional[types.ColumnNameType] = None,
) -> None:
    """
    Check if column with objects ids contains duplicates.
    """
    indices: np.ndarray = dataframe[id_column].values if id_column is not None else dataframe.index
    if len(indices) > len(set(indices)):
        if id_column is None:
            msg_part: str = "Index"
        else:
            msg_part: str = f"Id column {id_column}"
        raise ValueError(f"{msg_part} contains duplicates, ids must be unique for split")


def get_integer_salt(salt: Optional[str]) -> int:
    """
    Returns integer for random state numpy.

    Parameters
    ----------
    salt: Optional str

    Returns
    -------
    random_state: int
    """
    return None if salt is None else int(hashlib.shake_256(salt.encode()).hexdigest(3), 16)


def encode_id(enc_id: Any, salt: str, hash_function: Union[str, Callable] = "sha256") -> int:
    """
    Generate hash ids using prefered salt value and encoding algorithm.

    Parameters
    ----------
    enc_id : Any
        Encoding id.
    salt : str
        Salt for endoing (``enc_id`` + ``salt``).
    hash_function : str or Callable, default: ``"sha256"``
        Function that is used for id encoding.

    Returns
    -------
    enc_id_hashed : int
        Reduced hashed id.
    """
    # some constants
    LEN_HASH: int = 10  # pylint: disable=C0103
    HASH_BASE: int = 16  # pylint: disable=C0103

    enc_id = str(enc_id)
    if salt:
        enc_id += salt
    enc_id = enc_id.encode()
    hash_dict: Dict[str] = {
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512,
        "blake2": hashlib.blake2b,
    }
    if hash_function in hash_dict:
        hash_function = hash_dict[hash_function]
    elif isinstance(hash_function, str):
        raise ValueError(f"Unknown string value for hash function {', '.join(hash_dict)}")
    enc_id_hashed = hash_function(enc_id)
    if hasattr(enc_id_hashed, "hexdigest") and callable(getattr(enc_id_hashed, "hexdigest")):
        enc_id_hashed = enc_id_hashed.hexdigest()
    else:
        raise AttributeError("hexdigest() method must be implemented in hash_function")
    enc_id_hashed = enc_id_hashed[:LEN_HASH]
    enc_id_hashed = int(enc_id_hashed, HASH_BASE)
    return enc_id_hashed


def get_simple_split(
    df: pd.DataFrame, group_size: int = None, groups_number: int = 2, group_b_indices: Optional[Iterable] = None
) -> List[np.ndarray]:
    """
    Simple random uniform split approach.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe used for groups sampling.
    group_size : int
        Sampling size of groups.
    grops_number : int, default: ``2``
        Number of groups to be sampled.
    group_b_indices: np.ndarray, optional
        If group B was fixed, indices could be passed.

    Returns
    -------
    groups : List[np.ndarray]
       List of arrays with indices for each groups.
    """
    if group_b_indices is not None:
        a_indices: np.ndarray = np.random.choice(df.index.drop(group_b_indices).values, group_size, replace=False)
        return a_indices, group_b_indices
    groups: List[np.ndarray] = np.random.choice(df.index.values, size=(groups_number, group_size), replace=False)
    return groups


def get_hash_split(
    df: pd.DataFrame,
    id_column: str,
    group_size: int,
    groups_number: int = 2,
    group_b_indices: Optional[np.ndarray] = None,
    salt: Optional[str] = None,
    hash_function: Union[str, Callable] = "sha256",
) -> List[np.ndarray]:
    """
    Generate groups (basically control / test) of ids using SHA-256 hashing with salt.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe used for groups sampling.
    id_column : str
        Name of ids column in dataframe.
    group_size : int
        Sampling size of groups.
    grops_number : int, default: ``2``
        Number of groups to be sampled.
    group_b_indices: np.ndarray, optional
        If group B was fixed, indices could be passed.
    salt : str, optional
        Salt for determenistic hashing.
        If salt is not passed, a random one would be generated.
    hash_function : str or Callable, default: ``"sha256"``
        Hash function for encoding, for example sha256.
        If function is custom Callable object,
        a hexdigest() method must be implemented.

    Returns
    -------
    groups : List[np.ndarray]
       List of arrays with indices for each groups.
    """
    if group_b_indices is not None and groups_number != 2:
        raise RuntimeError(
            f"Fixed B group is available only for groups_number = 2, but groups_number = {groups_number}"
        )
    if groups_number * group_size > df.shape[0]:
        raise RuntimeError(f"Required value more, than table size {groups_number} * {group_size} > {df.shape[0]}")
    if not salt:
        salt = b64encode(os.urandom(8)).decode("ascii")
    if id_column is None:
        df_id = pd.DataFrame(df.index.values, columns=["id"])
        id_column = "id"
    else:
        df_id = df[[id_column]].copy()
    df_id.set_index(df.index.values, inplace=True)
    if group_b_indices is not None:
        df_id.drop(group_b_indices, inplace=True)
    df_id["hashed_id"] = df_id[id_column].apply(lambda x: encode_id(x, salt, hash_function)).values
    hashes: np.ndarray = np.sort(df_id["hashed_id"].values)
    groups: List[List[Any]] = []
    for group_index in range(groups_number):
        if (group_index + 1) * group_size > hashes.shape[0]:
            err_message: str = "Some problems with hash, it can be caused by collisions\n            "
            err_message += f"Required {(group_index + 1) * group_size} indexes, but hash size - {hashes.shape[0]}"
            raise RuntimeError(err_message)
        current_hashes: np.ndarray = hashes[group_index * group_size : (group_index + 1) * group_size]
        current_group: np.ndarray = df_id[df_id["hashed_id"].isin(current_hashes)].index.values
        groups.append(current_group)
        if group_b_indices is not None:
            groups.append(group_b_indices)
            break
    return groups


def get_metric_split(
    df: pd.DataFrame,
    group_size: int,
    fit_columns: List[Any],
    groups_number: int = 2,
    group_b_indices: Optional[np.ndarray] = None,
    threads: int = 1,
) -> List[np.ndarray]:
    """
    Generate groups of ids using metric approach(nearest neighbour search).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe used for groups sampling.
    group_size : int
        Sampling size of groups.
        For strong equality choose method 'equal_size'.
    fit_columns : List
        List of columns names which values will be interpreted as
        coordinates of points in multidimensional space during metric split.
    groups_number : int, default: ``2``
        Number of groups to be sampled.
    group_b_indices : np.ndarray, optional
        If group B was fixed, indices could be passed.

    Returns
    -------
    groups : List[np.ndarray]
       List of arrays with indices for each groups.
    """
    if group_b_indices is None:
        group_b_indices = np.random.choice(df.index, size=group_size, replace=False)
    rest_index: np.ndarray = df.index.drop(group_b_indices)
    ef_search: int = group_size * (groups_number - 1)
    nm_tree = NMTree(df.loc[rest_index, fit_columns].values, payload=rest_index, ef_search=ef_search)
    result: List[List[Any]] = []
    nm_tree.query_batch(
        points=df.loc[group_b_indices, fit_columns].values,
        payload=group_b_indices,
        out=result,
        group_size=group_size,
        amount=groups_number - 1,
        threads=threads,
    )
    return np.array(result).T


def get_dim_decrease_split(
    df: pd.DataFrame,
    group_size: int,
    fit_columns: List[Any],
    groups_number: int = 2,
) -> List[np.ndarray]:
    """
    Decreased dimension split method using TSNE.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe used for groups sampling.
    group_size : int
        Sampling size of groups.
        For strong equality choose method 'equal_size'.
    fit_columns : List
        List of columns names which values will be interpreted as
        coordinates of points in multidimensional space during metric split.
    groups_number : int, default: ``2``
        Number of groups to be sampled.

    Returns
    -------
    groups : List[np.ndarray]
       List of arrays with indices for each groups.
    """
    data: np.ndarray = df[fit_columns].values
    model: TSNE = TSNE(n_components=1, learning_rate="auto", init="random", n_jobs=-2)
    labels: np.ndarray = model.fit_transform(data)
    groups: pd.DataFrame = pd.DataFrame(index=df.index)
    groups["label"] = labels
    groups.sort_values(by="label", inplace=True)
    groups = groups[: groups_number * group_size]
    result: List[np.ndarray, np.ndarray] = []
    for j in range(groups_number):
        result.append(groups.index[j::groups_number])
    return result


def make_labels_for_groups(groups_number: int) -> List[str]:
    """
    Build list with labels for groups.

    Parameters
    ----------
    groups_number : int
        Groups number for splitting.

    Returns
    -------
    List with labels len(result) = groups_number.
    """
    alphabet_size: int = ord("Z") - ord("A") + 1
    if 2 * alphabet_size < groups_number:
        raise NotImplementedError("Groups number should be <= 52")
    if groups_number <= alphabet_size:
        return [chr(ord("A") + j) for j in range(groups_number)]
    else:
        first_part: List[str] = [chr(ord("A") + j) for j in range(alphabet_size)]
        second_part: List[str] = [chr(ord("a") + j) for j in range(groups_number - alphabet_size)]
        return first_part + second_part


def add_to_required_size(
    dataframe: pd.DataFrame, required_number: int, used_ids: np.ndarray, salt: Optional[str]
) -> np.ndarray:
    """
    Finds the required number of elements from the remaining, could be stable, using salt
    Parameters
    ----------
    dataframe: pandas Dataframe
        Dataframe with all ids
    required_number: int
        Number of elements required from not used ids
    ised_ids: np.ndarray
        Ids which was used
    salt: str, Optional
        Salt for stability
    Returns
    -------
    additional_elements: np.ndarray
        Required amount new elements from dataframe
    """
    stable_random: np.random.RandomState = np.random.RandomState(get_integer_salt(salt))
    not_used_id: np.ndarray = dataframe.loc[~dataframe.index.isin(used_ids)].index.values
    additional_elements: np.ndarray = stable_random.choice(not_used_id, required_number, replace=False)
    return additional_elements


def get_split(
    dataframe: pd.DataFrame,
    split_method: str,
    groups_size: Optional[int] = None,
    groups_number: int = 2,
    group_b_indices: Iterable[int] = None,
    id_column: types.ColumnNameType = None,
    salt: Optional[str] = None,
    fit_columns: Optional[List] = None,
    strat_columns: Optional[List] = None,
    stratifier: Optional[strat_pkg.Stratification] = None,
    labels: Optional[Sequence[Any]] = None,
    threads: int = 1,
    hash_function: Union[str, Callable] = "sha256",
) -> pd.DataFrame:
    """
    Create groups split from global pool with selected set of parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe used for groups sampling.
    split_method : str
        Method used for split task.
    id_column : str
        Name of ids column in dataframe.
    groups_size : int
        Sampling size of groups.
    groups_number : int, default: ``2``
        Number of groups to be sampled
    group_b_indices: np.ndarray, optional
        If group B was fixed, indices could be passed.
    fit_columns : Optional[List],
        Columns of the dataframe for metric calculation
    strat_columns : Optional[List]
        Columns for stratification https://en.wikipedia.org/wiki/Stratified_sampling
    salt : str, optional
        Salt for determenistic hashing.
        If salt is not passed, a random one would be generated.
    stratifier : Stratification, optional
        Instance of stratifier class.
    labels : Sequence[int], optional
        Labels for groups, default A, B, C ...
    threads : int, default: ``1``
        Number of threads for thread pool for metric split.
    hash_function : str or Callable, default: ``"sha256"``
        Hash function used for hash-approach, for example sha256.
        If function is custom Callable object,
        a hexdigest() method must be implemented.

    Returns
    -------
    groups : pd.DataFrame
        DataFrame with "group" column with group labels.
    """
    check_ids_duplicates(dataframe, id_column)
    if stratifier is None:
        stratifier = strat_pkg.Stratification()
        stratifier.fit(dataframe, strat_columns)
    elif not stratifier.is_trained():
        stratifier.fit(dataframe, strat_columns)

    error_size_msg: str = "Total size for all groups is bigger than total shape of table"
    cond_empty_b: bool = group_b_indices is None and groups_number * groups_size > stratifier.size()
    cond_b_id: bool = group_b_indices is not None and groups_number * len(group_b_indices) > stratifier.size()

    if cond_empty_b or cond_b_id:
        raise ValueError(error_size_msg)

    # Can't split for more than two groups with fixed test group
    if groups_number > 2 and group_b_indices is not None:
        raise NotImplementedError("Fixed group only for two groups")

    if split_method == "dim_decrease" and group_b_indices is not None:
        raise NotImplementedError("For fixed B group choose other method")

    group_ids: List[List[np.ndarray]] = [[] for _ in range(groups_number)]

    if group_b_indices is not None:
        test_inds: Dict[Tuple, Tuple[List, int]] = stratifier.get_test_inds(group_b_indices, id_column)
        groups_size = len(group_b_indices)
    elif groups_size is None:
        raise ValueError("Set groups size, if you do not set ids for B group")
    else:
        strat_sizes: Dict[Tuple, int] = stratifier.get_group_sizes(groups_size)

    for strat_value, strat_table in stratifier.groups():
        if group_b_indices is None:
            strat_group_size: int = int(strat_sizes[strat_value])
            id_b = None
        else:
            strat_group_size: int = len(test_inds[strat_value][0])
            if id_column is None:
                id_b = test_inds[strat_value][0]
            else:
                id_b = dataframe.loc[dataframe[id_column].isin(test_inds[strat_value][0])].index.values
        if split_method == "hash":
            splitted_groups = get_hash_split(
                strat_table,
                id_column=id_column,
                group_size=strat_group_size,
                groups_number=groups_number,
                group_b_indices=id_b,
                salt=salt,
                hash_function=hash_function,
            )
        elif split_method == "metric":
            splitted_groups = get_metric_split(
                strat_table,
                group_size=strat_group_size,
                fit_columns=fit_columns,
                groups_number=groups_number,
                group_b_indices=id_b,
                threads=threads,
            )
        elif split_method == "simple":
            splitted_groups = get_simple_split(
                df=strat_table, group_size=strat_group_size, groups_number=groups_number, group_b_indices=id_b
            )
        elif split_method == "dim_decrease":
            splitted_groups = get_dim_decrease_split(
                df=strat_table, group_size=strat_group_size, groups_number=groups_number, fit_columns=fit_columns
            )
        else:
            raise ValueError(f"Get incorrect split method {split_method}, choose one from {AVAILLABLE_SPLIT_METHODS}")
        for j in range(groups_number):
            group_ids[j].append(splitted_groups[j])

    # We should add new rest values for groups, which are not enough to the required size
    group_indices: List[np.ndarray] = [np.hstack(group_ids[j]) for j in range(groups_number)]
    used_ids: np.ndarray = np.hstack(group_indices)

    for j in range(groups_number):
        group: np.ndarray = group_indices[j]
        if group.shape[0] < groups_size:
            required: int = groups_size - group.shape[0]
            new_elements: np.ndarray = add_to_required_size(dataframe, required, used_ids, salt)
            group_indices[j] = np.append(group_indices[j], new_elements)
            used_ids = np.append(used_ids, new_elements)

    # Set labels
    labels = labels if labels is not None else make_labels_for_groups(groups_number)
    groups = dataframe.loc[used_ids].copy()
    groups[GROUPS_COLUMNS] = None
    for j in range(groups_number):
        groups.loc[group_indices[j], GROUPS_COLUMNS] = labels[j]
    return groups
