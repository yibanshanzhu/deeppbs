#!/usr/bin/env python

import argparse
import csv
import json
import os
import pickle
from os.path import join as ospj

import numpy as np
import torch
from torch_geometric.data import DataLoader

from deeppbs.nn import processBatch
from deeppbs.nn.metrics import mae
from deeppbs.nn.utils import loadDataset
from models.model_v2 import Model


def get_model_name(condition, readout):
    if readout == "all" and condition == "prot_shape":
        return "DeepPBS"
    if readout == "all" and condition == "prot_shape_ag":
        return "DeepPBSwithDNAseqInfo"
    if readout == "base" and condition == "prot":
        return "BaseReadout"
    if readout == "shape" and condition == "prot_shape":
        return "ShapeReadout"
    raise ValueError("No default model list for condition={!r}, readout={!r}".format(condition, readout))


def load_model_list(script_dir, config, model_list):
    if model_list is None:
        model_name = get_model_name(config["condition"], config.get("readout", "all"))
        model_list = ospj(script_dir, "plot_scripts", "txts", "{}.txt".format(model_name))

    with open(model_list) as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def load_model_inputs(config, data_names, run_names, script_dir):
    dataloaders = []
    infos = []
    for run_name in run_names:
        scaler_path = ospj(script_dir, "output", run_name, "scaler.pkl")
        with open(scaler_path, "rb") as handle:
            scaler = pickle.load(handle)

        dataset, _, info, _ = loadDataset(
            data_names,
            config["nc"],
            config["labels_key"],
            config["data_dir"],
            cache_dataset=config.get("cache_dataset", False),
            balance=config.get("balance", "unmasked"),
            remove_mask=False,
            scale=True,
            scaler=scaler,
            pre_transform=None,
            feature_mask=config.get("feature_mask", None),
        )
        dataloaders.append([batch for batch in DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)])
        infos.append(info)
    return dataloaders, infos


def load_models(config, run_names, script_dir, info, device):
    readout = config.get("readout", "all")
    models = []
    for run_name in run_names:
        checkpoint_path = ospj(script_dir, "output", run_name, "Model.best.tar")
        model = Model(info["prot_features"], info["dna_features"], condition=config["condition"], readout=readout)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
    return models


def ensemble_prediction(models, dataloaders, data_idx, device):
    outputs = []
    for model, dataloader in zip(models, dataloaders):
        batch_data = processBatch(device, dataloader[data_idx])
        with torch.no_grad():
            outputs.append(torch.softmax(model(batch_data["batch"]), dim=1).detach().cpu().numpy())

    output = sum(outputs) / len(outputs)
    midpoint = output.shape[0] // 2
    return (output[:midpoint, :] + np.flip(output[midpoint:, :])) / 2


def sample_mae(batch, prediction):
    target = batch.y_pwm0.detach().cpu().numpy()
    target_mask = batch.pwm_mask0.detach().cpu().numpy().astype(bool)
    prediction_mask = batch.dna_mask0.detach().cpu().numpy().astype(bool)

    target_aligned = target[target_mask]
    prediction_aligned = prediction[prediction_mask]
    if target_aligned.shape[0] != prediction_aligned.shape[0]:
        raise ValueError(
            "Aligned target length {} != prediction length {}".format(
                target_aligned.shape[0], prediction_aligned.shape[0]
            )
        )
    if target_aligned.shape[0] == 0:
        return float("nan"), 0, target.shape[0], prediction.shape[0]
    return (
        float(mae(target_aligned, prediction_aligned)),
        int(target_aligned.shape[0]),
        int(target.shape[0]),
        int(prediction.shape[0]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="Benchmark data file list, for example ./folds/id.txt.")
    parser.add_argument("-c", "--config", dest="config_file", required=True, help="DeepPBS config JSON.")
    parser.add_argument("--model_list", help="Text file containing the 5 ensemble run names.")
    parser.add_argument("--output", default="./output/benchmark_per_sample_mae.csv", help="Output CSV path.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config_file) as handle:
        config = json.load(handle)

    with open(args.data_file) as handle:
        data_names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]

    run_names = load_model_list(script_dir, config, args.model_list)
    if len(run_names) == 0:
        raise ValueError("No ensemble runs found.")

    dataloaders, infos = load_model_inputs(config, data_names, run_names, script_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = load_models(config, run_names, script_dir, infos[0], device)

    rows = []
    for data_idx, data_name in enumerate(data_names):
        prediction = ensemble_prediction(models, dataloaders, data_idx, device)
        value, aligned_len, pwm_len, dna_len = sample_mae(dataloaders[0][data_idx], prediction)
        rows.append({
            "id": data_name,
            "mae": value,
            "aligned_len": aligned_len,
            "pwm_len": pwm_len,
            "dna_len": dna_len,
            "pwm_coverage": aligned_len / pwm_len if pwm_len else float("nan"),
            "dna_coverage": aligned_len / dna_len if dna_len else float("nan"),
        })

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fieldnames = ["id", "mae", "aligned_len", "pwm_len", "dna_len", "pwm_coverage", "dna_coverage"]
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    values = np.array([row["mae"] for row in rows], dtype=float)
    mean_mae = float(np.nanmean(values))
    print("num_samples:", len(rows))
    print("mean_mae:", mean_mae)
    print("output:", args.output)


if __name__ == "__main__":
    main()
