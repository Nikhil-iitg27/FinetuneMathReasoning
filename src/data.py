from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

INSTRUCTION = """Answer the above question. Please provide your reasoning, output, and answer in the specified 
                XML format: <reasoning>...</reasoning> <output>...</output> <answer>...</answer>"""

class GSM8kXMLDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512):
        self.data = load_dataset("gsm8k", "main", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        question = question + f"\n{INSTRUCTION}\n"
        # Split answer into reasoning and answer
        if '####' in item['answer']:
            reasoning, answer = item['answer'].split('####', 1)
            reasoning = reasoning.strip()
            answer = answer.strip()
        else:
            reasoning = item['answer'].strip()
            answer = ""

        if '.' in reasoning:
            # Split into sentences, remove empty, take the last non-empty before ####
            sentences = [s.strip() for s in reasoning.split('.') if s.strip()]
            output = sentences[-1] + '.' if sentences else reasoning
        else:
            output = reasoning

        # Prepare the target string in your XML-like format
        target = f"<reasoning>{reasoning}</reasoning> <output>{output}</output> <answer>{answer}</answer>"

        enc = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        target_enc = self.tokenizer(
            target,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': target_enc['input_ids'].squeeze(0),
            'ground_truth_answer': answer
        }

def get_gsm8k_dataloader(split, tokenizer, batch_size=8, shuffle=True):
    dataset = GSM8kXMLDataset(split, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)