#%%
from datasets import load_dataset
import json
from datetime import datetime

ds = load_dataset("/path/to/code_generation_lite", version_tag="release_v5", split='test')
filter_date = datetime.strptime("2024-08-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
ds = ds.filter(
    lambda x: datetime.strptime(x["contest_date"], "%Y-%m-%dT%H:%M:%S") >= filter_date
).sort("question_id")

#%%
data = []

for example in ds:
    data.append({
        "prompt": [
            {
                "from": "user",
                "value": example["question_content"]
            }
        ],
        "final_answer": {
            "question_title": example["question_title"],
            "question_id": example["question_id"],
            "contest_id": example["contest_id"],
            "contest_date": example["contest_date"],
            "starter_code": example["starter_code"],
            "difficulty": example["difficulty"]
        }
    })

with open("livecodebench.json", "w") as f:
    json.dump(data, f,indent=2)

# %%
