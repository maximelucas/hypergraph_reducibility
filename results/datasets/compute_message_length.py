import argparse
import os
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from tqdm import tqdm

sys.path.append("../src/")

import xgi
from reducibility_hg import *

sb.set_theme(style="ticks", context="paper")

results_dir = "../results/"
out_dir = f"{results_dir}datasets/"

existing_datasets = [
    "email-enron",
    "email-eu",
    "hospital-lyon",
    "contact-high-school",
    "contact-primary-school",
    "tags-ask-ubuntu",
    "congress-bills",
    "disgenenet",
    "diseasome",
    "ndc-substances",
    "coauth-mag-geology",
    "coauth-mag-history",
    "PACS0",
    "PACS1",
    "PACS2",
    "PACS3",
    "PACS4",
    "PACS5",
    "PACS6",
    "PACS7",
    "PACS8",
    "PACS9",
]


def plot_message_length_3panels(orders, Ds_H, lZs_H, Q, dataset_name, save=True):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 2.1), constrained_layout=True)

    ax1.plot(orders, Ds_H, "o-", mfc="w")

    ax1.set_title("Information Loss (KL)")
    ax1.set_xlabel("Max Order")
    ax1.set_ylabel("Bits")

    ax2.plot(orders, lZs_H, "o-", mfc="w")
    ax2.set_title("Model Complexity ($\delta S$)")
    ax2.set_xlabel("Max Order")

    ax3.plot(orders, Q, "o-", mfc="w")
    y = np.where(Q == min(Q)) * np.ones(len(Q)) + 1
    x = np.linspace(np.min(Q), np.max(Q), len(Q))
    ax3.plot(y[0], x, c="r", alpha=0.7, ls="--", zorder=-2)

    ax3.set_title("Message Length")
    ax3.set_xlabel("Max Order")

    fig.suptitle(f"{dataset_name}")

    fig_name = f"message_length_{dataset_name}"

    if save:
        plt.savefig(f"{out_dir}{fig_name}.png", dpi=250, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process dataset name.")
    parser.add_argument(
        "-n", "--name", nargs="+", help="Name of the dataset", required=True
    )
    args = parser.parse_args()

    if args.name is None:
        raise ValueError(
            "No dataset name provided. Existing datasets are: " + str(existing_datasets)
        )

    print(args.name)
    if args.name == ["all"]:
        datasets = existing_datasets
    else:
        datasets = args.name

    rescale_per_node = False
    sparse = True

    print("Analysing the following datasets:")
    print(datasets)

    for i, dataset in enumerate(datasets):

        print("========")
        print(f"== {i+1}/{len(datasets)} {dataset}...")
        print("========")

        # compute only if not computed yet
        tag = f"message_length_{dataset}"
        file_name = f"{out_dir}{tag}.npz"
        if os.path.isfile(file_name):
            print(f"Dataset {dataset} was already computed and saved at {file_name}.")
            continue

        try:
            if "PACS" in dataset:
                H0 = xgi.read_json(f"../data/{dataset}.json")  # not yet online
            else:
                H0 = xgi.load_xgi_data(dataset, max_order=None, cache=True)
        except Exception as e:
            print(e)
            continue

        print(H0)
        print("max order:", xgi.max_edge_order(H0))
        H0.cleanup(isolates=False, singletons=False, multiedges=False)
        print(H0)

        orders = np.array(xgi.unique_edge_sizes(H0)) - 1
        weights = np.ones(len(orders))

        # compute characteristic timescale
        print("Computing characteristic tau...")
        tau_c = find_charact_tau(H0, orders, weights, sparse=sparse)

        # compute message length
        print("Computing message length...")
        Ds_H, lZs_H, orders = compute_information(
            H0, tau_c, rescale_per_node=rescale_per_node, sparse=sparse
        )
        Q = Ds_H + lZs_H

        # plot results
        plot_message_length_3panels(orders, Ds_H, lZs_H, Q, dataset, save=True)

        # save results
        np.savez(
            f"{out_dir}{tag}.npz",
            Q=Q,
            Ds_H=Ds_H,
            lZs_H=lZs_H,
            orders=orders,
            tau_c=tau_c,
            rescale_per_node=rescale_per_node,
        )

    shutil.copy2(__file__, out_dir)
    print("script copied, results saved, done")
