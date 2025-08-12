import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Parse LVEF values from discharge CSV.")
parser.add_argument("--csv_path", type=str, help="Path to the discharge CSV file (can be .gz)", default="/media/Volume/data/MIMIC_IV/discharge.csv.gz")
parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it", help="Model ID for the LVEF parsing model")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the  model on (e.g., 'cuda' or 'cpu')")
parser.add_argument("--output_path", type=str, default="/media/Volume/data/MIMIC_IV/discharge_lvef.csv", help="Path to the output CSV file")
parser.add_argument('--test', action='store_true', help='Run in test mode with a small subset of data')

args = parser.parse_args()

csv = pd.read_csv(args.csv_path)
print(csv.head())
# print unique subject IDs
unique_subject_ids = csv['subject_id'].unique()
print("Unique subject IDs in discharge.csv:", len(unique_subject_ids))

# total len
print("Total number of ecgs:", len(csv))

csv['lvef'] = None



# Choose a light model
pipe = pipeline(
    "image-text-to-text",
    model=args.model_id,
    torch_dtype=torch.bfloat16,
    device="cuda",
)
# Define system prompt
system_prompt = {
    "role": "system",
    "content": [{
        "type": "text",
        "text": """
        You are a medical expert specializing in parsing LVE (left ventricular ejection fraction) values from medical reports.

        Analyze this medical report and extract ONLY the LVEF value (number or range) from the text, if no LVEF value is found, return "-". Output only one value, no explanation, no other markers.
        Be very careful to return a value only if it is clearly stated in the report, otherwise return "-": this is very crucial for the success of this medical application.

        """
    }]
}

batch_size = 8  # Choose appropriate batch size for your GPU
texts = csv["text"].tolist()

for start in range(0, len(texts), batch_size):
    end = start + batch_size
    batch_texts = texts[start:end]

    batch_inputs = []
    for text in batch_texts:
        batch_inputs.append([
            system_prompt,
            {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
        ])

    with torch.no_grad():
        outputs = pipe(batch_inputs, max_new_tokens=5)

    for i, output in enumerate(outputs):
        row_idx = start + i
        out = output[0]["generated_text"][-1]["content"].strip()
        if out != "-":
            print(f"Row {row_idx}: {out}")
        csv.iloc[row_idx, csv.columns.get_loc("lvef")] = out


    if args.test and end > 1000:
        break


# remove column 'text'
csv.drop(columns=['text'], inplace=True)
csv.to_csv(args.output_path)