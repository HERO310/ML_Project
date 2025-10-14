"""
Dataset handler for knowledge editing benchmarks (ZsRE and CounterFact)
"""
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import random

class KnowledgeEditDataset(Dataset):
    """Dataset for knowledge editing with context generation"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        context_template: str = None,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.data = self._process_data(raw_data)
        
        # Default context template
        if context_template is None:
            self.context_template = "Context: {context}\nQuestion: {question}\nAnswer:"
        else:
            self.context_template = context_template
    
    def _process_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Process raw data into standardized format"""
        processed = []
        
        for item in raw_data:
            # Handle different dataset formats
            if 'src' in item:  # ZsRE format
                processed_item = {
                    'question': item['src'],
                    'target': item['answers'][0] if isinstance(item['answers'], list) else item['answers'],
                    'subject': item.get('subject', ''),
                    'relation': item.get('relation_id', ''),
                    'alt_answers': item.get('answers', [])
                }
            elif 'requested_rewrite' in item:  # CounterFact format
                rewrite = item['requested_rewrite']
                processed_item = {
                    'question': rewrite['prompt'].format(rewrite['subject']),
                    'target': rewrite['target_new']['str'],
                    'subject': rewrite['subject'],
                    'relation': rewrite.get('relation_id', ''),
                    'alt_answers': [rewrite['target_new']['str']]
                }
            else:
                continue
            
            # Generate context for ICE
            processed_item['context'] = self._generate_context(processed_item)
            
            # Generate related queries for portability testing
            processed_item['related_queries'] = self._generate_related_queries(processed_item)
            
            # Generate unrelated queries for locality testing
            processed_item['unrelated_queries'] = self._generate_unrelated_queries()
            
            processed.append(processed_item)
        
        return processed
    
    def _generate_context(self, item: Dict) -> str:
        """Generate context for in-context learning"""
        # Simple template-based context generation
        subject = item.get('subject', '')
        target = item.get('target', '')
        
        contexts = [
            f"{subject} is known for {target}.",
            f"The answer related to {subject} is {target}.",
            f"{subject}: {target}",
            f"Information: {subject} - {target}"
        ]
        
        return random.choice(contexts)
    
    def _generate_related_queries(self, item: Dict) -> List[str]:
        """Generate related queries for portability evaluation"""
        subject = item.get('subject', '')
        target = item.get('target', '')
        
        related = [
            f"What is {subject}?",
            f"Tell me about {subject}.",
            f"Describe {subject}.",
            f"Who is {subject}?"
        ]
        
        return related[:2]
    
    def _generate_unrelated_queries(self) -> List[str]:
        """Generate unrelated queries for locality evaluation"""
        unrelated = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is 2 + 2?",
            "What color is the sky?"
        ]
        
        return [random.choice(unrelated)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create context-conditioned input
        context_query = self.context_template.format(
            context=item['context'],
            question=item['question']
        )
        
        # Tokenize context-conditioned input
        context_encoding = self.tokenizer(
            context_query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize query alone (for ICE loss)
        query_encoding = self.tokenizer(
            item['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            item['target'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'context_input_ids': context_encoding['input_ids'].squeeze(0),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(0),
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'target_input_ids': target_encoding['input_ids'].squeeze(0),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(0),
            'target_text': item['target'],
            'question': item['question'],
            'context': item['context'],
            'related_queries': item['related_queries'],
            'unrelated_queries': item['unrelated_queries']
        }

def create_sample_dataset(output_path: str, dataset_type: str = "zsre", num_samples: int = 100):
    """Create a sample dataset for testing"""
    if dataset_type == "zsre":
        data = []
        subjects = ["Albert Einstein", "Marie Curie", "Isaac Newton", "Stephen Hawking", "Galileo Galilei"]
        facts = ["physics", "chemistry", "gravity", "black holes", "astronomy"]
        
        for i in range(num_samples):
            subject = random.choice(subjects)
            fact = random.choice(facts)
            data.append({
                "src": f"What is {subject} known for?",
                "answers": [fact],
                "subject": subject,
                "relation_id": "P106"
            })
    
    elif dataset_type == "counterfact":
        data = []
        for i in range(num_samples):
            data.append({
                "requested_rewrite": {
                    "prompt": "What is {} known for?",
                    "subject": f"Person {i}",
                    "target_new": {"str": f"Field {i}"},
                    "relation_id": "P106"
                }
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created sample {dataset_type} dataset with {num_samples} samples at {output_path}")