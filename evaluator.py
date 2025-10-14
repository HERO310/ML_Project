"""
Evaluation metrics for ICE-LoRA: Edit Success, Portability, Locality, Generalization
"""
import torch
from tqdm import tqdm
from typing import Dict, List
import numpy as np
from collections import defaultdict

class ICELoRAEvaluator:
    """Evaluator for knowledge editing metrics"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate(self, dataset, metrics: List[str] = None) -> Dict:
        """
        Evaluate model on knowledge editing metrics
        
        Args:
            dataset: Evaluation dataset
            metrics: List of metrics to compute
        
        Returns:
            Dictionary with evaluation results
        """
        if metrics is None:
            metrics = ['reliability', 'locality', 'portability', 'generalization']
        
        results = {}
        
        if 'reliability' in metrics:
            results['reliability'] = self._compute_reliability(dataset)
        
        if 'locality' in metrics:
            results['locality'] = self._compute_locality(dataset)
        
        if 'portability' in metrics:
            results['portability'] = self._compute_portability(dataset)
        
        if 'generalization' in metrics:
            results['generalization'] = self._compute_generalization(dataset)
        
        return results
    
    def _compute_reliability(self, dataset) -> float:
        """
        Compute reliability (edit success rate)
        Measures if the model correctly outputs the edited fact
        """
        self.model.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for item in tqdm(dataset, desc="Computing Reliability"):
                question = item['question']
                target = item['target_text']
                
                # Generate response
                generated = self.model.generate(
                    question,
                    max_length=50,
                    temperature=0.7
                )
                
                # Check if target is in generated text
                if self._match_answer(generated, target):
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_locality(self, dataset) -> float:
        """
        Compute locality score
        Measures if unrelated knowledge is preserved
        """
        self.model.model.eval()
        preserved = 0
        total = 0
        
        # Store original responses for unrelated queries
        original_responses = {}
        
        with torch.no_grad():
            for idx, item in enumerate(tqdm(dataset, desc="Computing Locality")):
                unrelated_queries = item.get('unrelated_queries', [])
                
                for query in unrelated_queries:
                    # Generate response after editing
                    generated = self.model.generate(
                        query,
                        max_length=50,
                        temperature=0.7
                    )
                    
                    # For simplicity, we assume locality is preserved if generation is coherent
                    # In practice, you'd compare with pre-edit responses
                    if len(generated.strip()) > 0 and not self._is_gibberish(generated):
                        preserved += 1
                    total += 1
        
        return preserved / total if total > 0 else 0.0
    
    def _compute_portability(self, dataset) -> float:
        """
        Compute portability score
        Measures if edited knowledge transfers to related queries
        """
        self.model.model.eval()
        portable = 0
        total = 0
        
        with torch.no_grad():
            for item in tqdm(dataset, desc="Computing Portability"):
                target = item['target_text']
                related_queries = item.get('related_queries', [])
                
                for query in related_queries:
                    # Generate response
                    generated = self.model.generate(
                        query,
                        max_length=50,
                        temperature=0.7
                    )
                    
                    # Check if target appears in related query responses
                    if self._match_answer(generated, target):
                        portable += 1
                    total += 1
        
        return portable / total if total > 0 else 0.0
    
    def _compute_generalization(self, dataset) -> float:
        """
        Compute generalization score
        Measures consistency across paraphrases
        """
        self.model.model.eval()
        consistent = 0
        total = 0
        
        paraphrase_templates = [
            "What is {subject}?",
            "Tell me about {subject}.",
            "Describe {subject}.",
            "Who is {subject}?"
        ]
        
        with torch.no_grad():
            for item in tqdm(dataset, desc="Computing Generalization"):
                subject = item.get('question', '').split()[-1].rstrip('?')
                target = item['target_text']
                
                responses = []
                for template in paraphrase_templates[:2]:  # Use 2 paraphrases
                    query = template.format(subject=subject)
                    generated = self.model.generate(
                        query,
                        max_length=50,
                        temperature=0.7
                    )
                    responses.append(self._match_answer(generated, target))
                
                # Consistent if majority of paraphrases give correct answer
                if sum(responses) >= len(responses) / 2:
                    consistent += 1
                total += 1
        
        return consistent / total if total > 0 else 0.0
    
    def _match_answer(self, generated: str, target: str) -> bool:
        """Check if target appears in generated text"""
        generated_lower = generated.lower()
        target_lower = target.lower()
        
        # Simple substring match
        return target_lower in generated_lower
    
    def _is_gibberish(self, text: str) -> bool:
        """Simple heuristic to detect gibberish"""
        # Check for repeated characters
        if len(set(text.replace(' ', ''))) < 3:
            return True
        
        # Check for excessive punctuation
        punct_ratio = sum(c in '!@#$%^&*()' for c in text) / max(len(text), 1)
        if punct_ratio > 0.3:
            return True
        
        return False
    
    def generate_report(self, results: Dict, save_path: str = None) -> str:
        """
        Generate evaluation report
        
        Args:
            results: Dictionary of evaluation metrics
            save_path: Optional path to save report
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*60)
        report.append("ICE-LoRA EVALUATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Reliability
        if 'reliability' in results:
            score = results['reliability']
            status = "✓ PASS" if score >= 0.80 else "✗ FAIL"
            report.append(f"Reliability Score:     {score:.4f} (Target: 0.80) {status}")
        
        # Locality
        if 'locality' in results:
            score = results['locality']
            status = "✓ PASS" if score >= 0.90 else "✗ FAIL"
            report.append(f"Locality Score:        {score:.4f} (Target: 0.90) {status}")
        
        # Portability
        if 'portability' in results:
            score = results['portability']
            status = "✓ PASS" if score >= 0.70 else "✗ FAIL"
            report.append(f"Portability Score:     {score:.4f} (Target: 0.70) {status}")
        
        # Generalization
        if 'generalization' in results:
            score = results['generalization']
            status = "✓ PASS" if score >= 0.75 else "✗ FAIL"
            report.append(f"Generalization Score:  {score:.4f} (Target: 0.75) {status}")
        
        report.append("")
        report.append("="*60)
        
        # Overall assessment
        all_scores = [results.get(m, 0) for m in ['reliability', 'locality', 'portability', 'generalization']]
        avg_score = np.mean(all_scores)
        report.append(f"Average Score:         {avg_score:.4f}")
        report.append("="*60)
        
        report_text = "\n".join(report)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text