import argparse
import os
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


def plot_message_length_shuffled(Qs_h, orders, d_shuffles, dataset, save=True):

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
        
    ax.plot(orders, Qs_H[0], "o-", label=f"not shuffled", ms=10, mfc="white")    
    ax.axvline(d_shuffles[0], ls="--", c=f"C0", zorder=-2, alpha=0.8)

    for j in range(len(Qs_H)):
        ax.plot(orders, Qs_H[j+1], "o-", label=f"d={d_shuffles[j]} shuffled")

        ax.axvline(d_shuffles[j], ls="--", c=f"C{j+1}", zorder=-2, alpha=0.8)

    ax.set_xlabel("Max order")
    #ax.set_xticks(orders)

    ax.set_ylabel("Quality function")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    #ax.set_xlim([0, 20])

    sb.despine()

    fig.suptitle(f"{dataset} $p_s={p_shuffle}$")

    fig_name = f"shuffled_{dataset}_p_s_{p_shuffle}" #lambda2_HG_SC_N_{N}_ps_{ps}_nrep_{n_repetitions}"

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
        tau_c = find_charact_tau(H0, orders, weights)

        # compute shuffled hypergraphs
        d_shuffles = [3, 5, 7] # orders to shuffle
        p_shuffle = 1 # probability of shuffling

        Hs = []

        # create copies of the hypergraph with edges shuffled
        for d_shuffle in d_shuffles:
            if d_shuffle <= xgi.max_edge_order(H0):
                print(d_shuffle)
                Hs.append(shuffle_hyperedges(S=H0, order=d_shuffle, p=p_shuffle))

        # compute message length
        print("Computing message length...")
        # compute message length
        Ds_H = []
        lZs_H = []
        Qs_H = []

        for H in tqdm([H0] + Hs):
            
            Ds_H_i, lZs_H_i, orders = compute_information(H, tau_c, rescale_per_node=rescale_per_node)
            Q_i = Ds_H_i + lZs_H_i
            
            Ds_H.append(Ds_H_i)
            lZs_H.append(lZs_H_i)
            Qs_H.append(Q_i)

        # plot results
        plot_message_length_shuffled(Qs_H, orders, d_shuffles, dataset, save=True)

        # save results
        np.savez(
            f"{out_dir}shuffled_{dataset}_ds_{d_shuffles}.npz",
            Ds_H=Ds_H,
            lZs_H=lZs_H,
            Qs_H=Qs_H,
            d_shuffles=d_shuffles,
            orders=orders,
            p_shuffle=p_shuffle,
            tau_c=tau_c
        )
