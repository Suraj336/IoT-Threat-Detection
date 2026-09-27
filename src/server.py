"""Runs the federated simulation: spins up N SmartHomeClient instances in-process
and coordinates FedAvg rounds via Flower. This is the "in one process" simulation
mode — for a real deployment, run client.py as a separate process per physical
device/gateway and point it at a real fl.server.start_server host, one flag change.
"""
import argparse
import csv
import json
import os
import time

import flwr as fl
import torch
import yaml
from flwr.common import Metrics

from client import SmartHomeClient
from model import ThreatClassifier, get_parameters, set_parameters
from utils import list_client_files, make_client_dataloaders

HERE = os.path.dirname(__file__)


def weighted_average(metrics_list):
    """Aggregate per-client eval metrics into federated-wide metrics, weighted
    by each client's number of examples — the standard FedAvg eval aggregation.
    """
    total_examples = sum(n for n, _ in metrics_list)
    agg = {}
    for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
        agg[key] = sum(n * m[key] for n, m in metrics_list) / total_examples
    return agg


def make_client_fn(client_files, cfg, num_features, num_classes):
    def client_fn(cid: str):
        train_loader, test_loader, _ = make_client_dataloaders(
            client_files[int(cid)],
            test_split=cfg["data"]["test_split"],
            batch_size=cfg["federated"]["batch_size"],
            seed=cfg["data"]["seed"],
        )
        return SmartHomeClient(
            client_id=int(cid), train_loader=train_loader, test_loader=test_loader,
            num_features=num_features, num_classes=num_classes, cfg=cfg,
        ).to_client()
    return client_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "..", "config.yaml"))
    parser.add_argument("--clients_dir", default=os.path.join(HERE, "..", "data", "clients"))
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--dp_noise_multiplier", type=float, default=None)
    parser.add_argument("--out_dir", default=os.path.join(HERE, "..", "models"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.rounds:
        cfg["federated"]["rounds"] = args.rounds
    if args.dp_noise_multiplier is not None:
        cfg["privacy"]["noise_multiplier"] = args.dp_noise_multiplier

    client_files = list_client_files(args.clients_dir)
    if not client_files:
        raise SystemExit(
            f"No client shards found in {args.clients_dir}. "
            f"Run `python data/prepare_data.py` first."
        )
    num_clients = args.num_clients or len(client_files)
    client_files = client_files[:num_clients]

    num_features = len(cfg["data"]["classes"]) and cfg["data"]["num_features"]
    num_classes = len(cfg["data"]["classes"])

    init_model = ThreatClassifier(
        num_features, num_classes,
        hidden_dims=cfg["model"]["hidden_dims"], dropout=cfg["model"]["dropout"],
    )
    init_parameters = fl.common.ndarrays_to_parameters(get_parameters(init_model))

    round_metrics = []

    os.makedirs(args.out_dir, exist_ok=True)
    live_path = os.path.join(args.out_dir, "live_metrics.json")
    total_rounds = cfg["federated"]["rounds"]
    run_started_at = time.time()

    def write_live_metrics(status="running"):
        payload = {
            "status": status,
            "total_rounds": total_rounds,
            "num_clients": num_clients,
            "started_at": run_started_at,
            "updated_at": time.time(),
            "rounds": [
                {"round": i, **m} for i, m in enumerate(round_metrics, start=1)
            ],
        }
        # Write atomically so the dashboard never reads a half-written file
        tmp_path = live_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, live_path)

    write_live_metrics(status="running")  # so the dashboard has something to poll immediately

    def fit_config(server_round):
        return {"round": server_round}

    def evaluate_metrics_aggregation_fn(metrics: list) -> Metrics:
        agg = weighted_average(metrics)
        round_metrics.append(agg)
        print(f"[round] agg eval metrics: {agg}")
        write_live_metrics(status="running")
        return agg

    # Track the latest global weights so we can save a usable checkpoint at the
    # end of the run. Flower's History object doesn't carry final parameters by
    # default, so we capture them ourselves on every aggregate_fit call.
    latest_parameters = {"value": init_parameters}

    class SavingFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated is not None and aggregated[0] is not None:
                latest_parameters["value"] = aggregated[0]
            return aggregated

    # TODO (secure aggregation): a production deployment would replace FedAvg's
    # plain-sum aggregation with a secure-aggregation protocol (e.g. pairwise
    # additive masking a la Bonawitz et al. 2017) so the server only ever sees
    # the SUM of client updates, never an individual client's (already DP-noised)
    # update. Flower's `SecAgg`/`SecAgg+` mods can be attached here.
    strategy = SavingFedAvg(
        fraction_fit=cfg["federated"]["fraction_fit"],
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=init_parameters,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    client_fn = make_client_fn(client_files, cfg, num_features, num_classes)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg["federated"]["rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    write_live_metrics(status="finished")

    # Save the final global model so the dashboard (or a real device) can load
    # it and run inference, instead of only ever seeing aggregate metrics.
    final_ndarrays = fl.common.parameters_to_ndarrays(latest_parameters["value"])
    set_parameters(init_model, final_ndarrays)
    torch.save(init_model.state_dict(), os.path.join(args.out_dir, "global_model.pt"))
    print(f"Saved final global model to {os.path.join(args.out_dir, 'global_model.pt')}")

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "f1_macro",
                                                "precision_macro", "recall_macro"])
        writer.writeheader()
        for i, m in enumerate(round_metrics, start=1):
            writer.writerow({"round": i, **m})
    print(f"\nSaved per-round metrics to {metrics_path}")
    print(f"DP setting: enabled={cfg['privacy']['enable_dp']}, "
          f"noise_multiplier={cfg['privacy']['noise_multiplier']}, "
          f"max_grad_norm={cfg['privacy']['max_grad_norm']}")


if __name__ == "__main__":
    main()