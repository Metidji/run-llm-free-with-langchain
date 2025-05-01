# run-llm-free-with-langchain

This project shows how to run a lightweight language model (LLM) for **free** on **Google Colab** using [Alpaca-native](https://huggingface.co/chavinlo/alpaca-native), and how to integrate it with **LangChain** for building smart prompt pipelines — including an easy-to-understand example using **Few-Shot Prompt Templates**.

---

## 🌟 Highlights

- 🚀 Run LLMs on Colab with minimal memory (4-bit quantized)
- 🧠 Use LangChain to build structured prompts and chains
- 💬 Learn Few-Shot Prompting for more accurate answers
- 🔌 Hugging Face Transformers + bitsandbytes = efficient setup

---

## 📂 Files

- `_langChain_.ipynb` — Notebook with all setup and examples
- `README.md` — This guide

---


### ✅ Setup Steps in Colab

1. **Enable GPU**  
   Go to **Runtime > Change runtime type > GPU**

2. **Check GPU availability**
   ```python
   import torch
   print("GPU available:", torch.cuda.is_available())
   ```

3. **Install dependencies** (done automatically in the notebook)

4. **Run all cells**

---

## 🔤 Few-Shot Prompting with LangChain

LangChain makes it easy to create prompts that include a few example pairs — this helps guide the model with better structure and output quality.

### 💡 Example: Translate Sentences to French

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# 1. Define example input-output pairs
examples = [
    {"input": "Translate 'Thank you' to French.", "output": "merci"},
    {"input": "Translate 'Goodbye' to French.", "output": "Au revoir"},
]

# 2. Format each example
example_prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")

# 3. Build the full prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Translate the following sentences to French:",
    suffix="Input: Translate '{sentence}' to French.\nOutput:",
    input_variables=["sentence"]
)

# 4. Use the template with a new sentence
formatted_prompt = few_shot_prompt.format(sentence="Good morning")
response = llm(formatted_prompt)
print(response)
```

### ✅ Result
This setup shows 2 examples first, then asks the model to continue the pattern — great for translation, classification, formatting tasks, etc.

---

## 🛠️ Tips to Customize

- Modify prompts for summarization, dialogue, explanations, etc.
- Use `ConversationChain`, `LLMMathChain`, or memory modules
- Chain multiple prompts together for complex workflows

---

