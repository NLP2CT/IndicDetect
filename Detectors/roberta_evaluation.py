import logging
import random
import numpy as np
import torch
import tqdm
import argparse
import json
import transformers
from metrics import get_roc_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def experiment(args):
    # load model
    logging.info(f"Loading base model of type {args.model_name}...")
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(args.model_name).to(args.DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Test in {filename}")
        test_data = json.load(open(filename, "r"))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        predictions = {'human': [], 'llm': []}
        with torch.no_grad():
            for item in tqdm.tqdm(test_data):
                text = item["text"]
                label = item["label"]

                if label == "human":
                    tokenized = tokenizer([text], padding=True, truncation=True, max_length=512,
                                           return_tensors="pt").to(args.DEVICE)
                    predictions["human"].append(detector(**tokenized).logits.softmax(-1)[:, 0].tolist())
                elif label == "llm":
                    tokenized = tokenizer([text], padding=True, truncation=True, max_length=512,
                                          return_tensors="pt").to(args.DEVICE)
                    predictions["llm"].append(detector(**tokenized).logits.softmax(-1)[:, 0].tolist())
                else:
                    raise ValueError(f"Unknown label {label}")

        predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy = get_roc_metrics(predictions['human'],
                                                                                                   predictions['llm'])

        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

        if "xlm-roberta-base" in args.model_name:
            result["model_type"] = "xlm-roberta-base"
        if "xlm-roberta-large" in args.model_name:
            result["model_type"] = "xlm-roberta-large"

        logging.info(f"{result}")
        with open(filename.split(".json")[0] + f"_{args.model_name}_data.json", "w") as f:
            json.dump(test_data, f, indent=4)

        with open(filename.split(".json")[0] + f"_{args.model_name}_result.json", "w") as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--model_name', default="roberta-base-openai-detector", type=str, required=False)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)
