# sales llm project

a small-scale language model trained on sales data (chat logs, pdfs, images).  
the goal: build a lightweight **sales assistant** that can answer leads, summarize documents, and pull relevant info.  

---

## note from jabez
this whole thing is kinda dumb because i don’t even have enough ram to train it properly. but it’s a start.  

---

## why
most sales tools are either too generic or too heavy. this project is about building something **domain-specific**, but still trainable on consumer hardware. think of it as a custom llm fine-tuned just for sales workflows.

---

## features
- train a transformer (~50m–125m params) from scratch or fine-tune on your sales data  
- process pdfs, chat logs, and scanned images (ocr) into one dataset  
- inference api + simple chat interface  
- document-aware responses (quotes, invoices, etc.)  

---

## project structure

```
sales-llm-project/
├── src/
│   ├── data/           # data processing scripts
│   ├── model/          # model architecture
│   ├── training/       # training loop + utils
│   └── inference/      # chat + api code
├── data/
│   ├── upload/         # drop your files here
│   └── processed/      # ready for training
├── models/             # saved checkpoints + tokenizer
├── configs/            # yaml config files
└── process_and_train.py  # one-shot processing + training
```

---

## getting started

### 1. setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. add your data
drop everything into `data/upload/`:
- pdfs, txt files, images (for ocr), json chat logs  
- json format: `[{"role": "user", "content": "..."}]`  

or download some starter data:
```bash
python download_training_data.py
```

### 3. process & train (all-in-one)
```bash
python process_and_train.py
```

or do it manually:
```bash
python src/data/process_data.py
python src/training/train.py
```

### 4. chat with it
```bash
python src/inference/cli_chat.py
```

or run the api:
```bash
python src/inference/api.py
```

---

## hardware notes
- minimum: 16gb ram + 8gb vram gpu  
- recommended: 32gb ram + 12gb+ vram  
- training: a few hours → a couple of days depending on dataset size  

---

## roadmap
- [ ] add web ui for sales team demo  
- [ ] improve document retrieval  
- [ ] try lora fine-tuning for efficiency  
- [ ] deploy lightweight inference server  
- [ ] add more documents

---

## license
mit — use it, hack it, improve it.  
