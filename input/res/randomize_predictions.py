import json
import random

input_file = "predictions.jsonl"
output_file = "predictions_random.jsonl"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        obj = json.loads(line)
        obj["prediction"] = random.randint(1, 5)
        fout.write(json.dumps(obj) + "\n")

print("Stop", output_file)