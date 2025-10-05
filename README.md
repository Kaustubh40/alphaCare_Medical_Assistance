# AlpaCare MedInstruct Assistant

A fine-tuned medical educational assistant built on **Microsoft Phi-2** using **LoRA adapters** for efficient supervised fine-tuning.

---

## 🚀 Features

* Adapted for **medical instruction & health awareness**.
* Lightweight **LoRA fine-tuning** (runs on Google Colab GPU).
* Ensures safe communication with **automatic disclaimers**.
* Easy deployment with Hugging Face `transformers` + `peft`.

---

## 📂 Project Structure

```
AlpaCare/
│
├── data_loader.py             # Loads dataset
├── requirements.txt           # Dependencies
├── README.md                  # Project guide
├── REPORT.pdf                 # Final report
└── notebooks/
    ├── colab_finetune.ipynb   # Training notebook
    └── inference_demo.ipynb   # Inference & outputs
```

---

## ⚙️ Setup

```bash
pip install torch transformers datasets accelerate peft trl sentencepiece
```

---

## 🏋️ Training (LoRA Fine-Tuning)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTTrainingArguments

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

# Apply LoRA
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)

# Train with SFTTrainer...
```

---

## 💡 Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "microsoft/phi-2"
adapter_path = "./alpacare_lora"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model)

def ask(prompt):
    text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=250)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

ask("Explain why hydration is important during fever recovery.")
```

---

## 📊 Sample Outputs

**Q:** Why is hydration important during fever recovery?
**A:** Hydration replaces fluids lost from sweating, regulates body temperature, and maintains electrolytes for faster recovery.

**Q:** List healthy habits for heart health.
**A:** Balanced diet, regular exercise, no smoking, adequate sleep, stress management.

---

## 🚀 How to Run the Project (Simple Demo Flow)

1. **Open Google Colab**  
   👉 [Google Colab](https://colab.research.google.com)

2. **Upload the notebooks:**  
   - `colab_finetune.ipynb` → (training process, optional if you want to show fine-tuning)  
   - `inference_demo.ipynb` → (main demo notebook for testing outputs)  

3. **Change runtime settings:**  
   - Go to **Runtime → Change runtime type**  
   - Select:  
     - **Python:** 3.10  
     - **Hardware Accelerator:** GPU (T4 / A100 or available)  
   - Save.  

4. **Run notebook cells one by one:**  
   - The first cell installs dependencies (`transformers`, `peft`, `trl`, etc.).  
   - The base model (`microsoft/phi-2`) and tokenizer load from Hugging Face.  
   - Your fine-tuned **LoRA adapter folder** (`alpacare_lora_adapter`) is loaded.  

5. **Test outputs (in `inference_demo.ipynb`):**  
   - Run the `ask()` function with your prompt.  
   - Example:  
     ```python
     ask("Explain why hydration is important during fever recovery.")
     ```  
   - ✅ The model generates a safe medical-style response with an **educational disclaimer**.  


## ⚠️ Disclaimer

This project is **for educational and research purposes only**.
It is **not a medical device** and should not replace professional healthcare advice.

