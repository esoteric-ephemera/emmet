""" This module scrapes datasets from matminer for later use in build validation. """

from __future__ import annotations
from matminer.datasets import load_dataset
import pandas as pd
from monty.serialization import dumpfn

def get_kingsbury_datasets(output_file_name : str | None = "expt_kingsbury_data.json.gz"):
    """
    Concatenate the experimental gap and formation enthalpy datasets from matminer.

    We keep only data that has a matched MP ID.
    
    Parameters
    -----------
    output_file_name : str or None
        If a str, the name of the output file in JSON format.

    Returns
    --------
        pandas.DataFrame representation of the concatenated datasets.
    """

    datasets = ("expt_formation_enthalpy_kingsbury", "expt_gap_kingsbury")
    column_map = {
        "uncertainty": "expt_form_e_uncertainty",
        "phaseinfo": "phase_info"
    }
    columns = [
        "expt_form_e", "uncertainty", "expt_gap", "phaseinfo", "formula"
    ]
    indices = set()

    data = {}
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)

        use_col = [col for col in columns if col in dataset]
        indices.update(set([mpid for mpid in dataset["likely_mpid"] if mpid is not None]))

        for idx in dataset.index:

            if (mpid := dataset["likely_mpid"][idx]) is None:
                continue

            data[mpid] = {
                **data.get(mpid,{}),
                **{
                    column_map.get(k,k) : dataset[k][idx] for k in use_col
                }
            }
    indices = sorted(list(indices), key = lambda val : int(val.split("-")[-1]))

    mapped_columns = [column_map.get(k,k) for k in columns]


    df = pd.DataFrame(
        [[data[mpid].get(k) for k in mapped_columns] for mpid in indices],
        columns=mapped_columns,
        index=indices
    )

    if output_file_name is not None:

        efficient_json = {
            **{
                col: [v if v == v else None for v in df[col]] # convert NaN to null
                for col in mapped_columns
            },
            **{
                "mpid": indices
            }
        }

        dumpfn(efficient_json,output_file_name)

    return df