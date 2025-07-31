import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Parse LVEF values from discharge CSV.")
parser.add_argument("csv_path", type=str, help="Path to the discharge CSV file (can be .gz)", default="/media/Volume/data/MIMIC_IV/discharge.csv.gz")
parser.add_argument("--model_id", type=str, default="google/gemma-2-9b-it", help="Model ID for the LVEF parsing model")
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

# Load tokenizer and model (4-bit quantized)
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    device_map="cuda",
    torch_dtype=torch.float16,
    load_in_4bit=True  # needs bitsandbytes
)
for i, text in enumerate(csv["text"]):
    prompt = f"""Extract only the LVEF value (number or range) from the text, for example:
    ... LVEF>55%   --> >55
    Doppler elt his LVEF was ~30% and reduced ...  --> 30
    HFpEF (LVEF 50% ___, PAD, CKD (stage IV), prior DVT c/b severe --> 50
    Her TTE showed mild-moderate mitral and moderate tricuspid regurgitation, LVEF 50-55%, and pulmonary hypertension. --> 50-55

    if no LVEF value is found, return "unknown".
    text: {text}
    
    Answer: The LVEF value is:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=15)
        print(outputs)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.replace(prompt, "").strip()
    print(f"Row {i}: {answer}")
    csv.at[i, 'lvef'] = answer
    if args.test:
        break

csv.to_csv(args.output_path)