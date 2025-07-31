import pandas as pd

csv= pd.read_csv('/media/Volume/data/MIMIC_IV/discharge.csv.gz')
print(csv.head())
# print unique subject IDs
unique_subject_ids = csv['subject_id'].unique()
print("Unique subject IDs in discharge.csv:", len(unique_subject_ids))

# total len
print("Total number of ecgs:", len(csv))

csv['lvef'] = None

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

# Choose a light model
model_id = "google/gemma-2-9b-it"

# Load tokenizer and model (4-bit quantized)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
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
    break

csv.to_csv('/media/Volume/data/MIMIC_IV/discharge_lvef.csv')