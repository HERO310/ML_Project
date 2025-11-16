
# **ICE-LoRA: Efficient Knowledge Editing with Low-Rank Adaptation**
---
## **📌 Project Overview**

Knowledge editing aims to update specific facts in large language models (LLMs) without retraining or harming unrelated knowledge.
Traditional fine-tuning uses cross-entropy on one-hot targets and often leads to:

* Catastrophic forgetting
* Low generalization
* Overfitting
* High computational cost

The **ICE (In-Context Editing)** method solves this using a distribution-based KL divergence loss.
However, the original ICE implementation uses **MEMIT** (full-layer updates), which is computationally expensive.

This project integrates ICE with **parameter-efficient fine-tuning (PEFT)**, implementing:

* **Baseline LoRA**
* **AdaLoRA**
* **GoRA**
* **ICE-LoRA (KL-divergence + LoRA)**

This reduces memory usage by **95%** while maintaining high-quality knowledge edits.

---

## **🎯 Objectives**

* Implement ICE-LoRA using LoRA adapters + ICE KL loss
* Compare with Baseline LoRA, AdaLoRA, and GoRA
* Evaluate performance on:

  * Edit Success
  * Portability
  * Locality
  * Fluency
* Run all experiments on consumer hardware (RTX 3060, T4, etc.)

---

## **📦 Requirements**
```
torch
transformers>=4.45.0
peft>=0.13.0
huggingface_hub
tqdm
matplotlib
seaborn
accelerate>=0.34.0
bitsandbytes>=0.44.0
```
---

## **📘 Methods Implemented**

### **1. Baseline LoRA**

A standard PEFT method using low-rank matrices injected into attention layers.

### **2. AdaLoRA**

Adaptive LoRA dynamically reallocates rank during training based on importance scores → improves efficiency.

### **3. GoRA**

Gradient-orthogonalized LoRA introduces constraints to prevent interference during edits.

### **4. ICE-LoRA (Main Contribution)**

The core technique of the project.

**ICE Loss:**
[
L_{\text{ICE}} = D_{KL}(p_\theta(x|[c, q]) | p_\theta(x|q))
]

Total loss:
[
L_{\text{total}} = L_{FT} + \lambda \cdot L_{\text{ICE}}
]

This allows the model to learn from **self-induced distributions** instead of fixed targets.

### **5. Training Pipeline**
* ICELoRAEditor class
* Loss calculation (CE + KL)
* Context generation
* Evaluation utilities
* LoRA, AdaLoRA, GoRA wrappers
* Model saving + inference

---

## **📊 Datasets Used**

### Wikidata_counterfact

---

## **📈 Evaluation Metrics**

You implemented all four ICE metrics:

| Metric           | Meaning                           | 
| ---------------- | --------------------------------- | 
| **Edit Success** | Whether the new fact is learned   | 
| **Locality**     | No damage to unrelated knowledge  | 
| **Portability**  | Knowledge applies to new contexts | 


---

## **🏆 Results Summary**

Based on your experiments (as described in the report):

* ICE-LoRA outperforms all baselines on:

  * Edit accuracy
  * Portability
  * Fluency
* LoRA/AdaLoRA/GoRA are fast but weaker for true knowledge integration
* ICE-LoRA maintains locality (low interference) thanks to KL regularization
* Memory usage reduced by **95%** vs. MEMIT full-layer updates

---

## **📚 References**

* Mitchell et al., **In-Context Editing (ICE)**
* Hu et al., **LoRA**
* Meng et al., **MEMIT**
* CounterFact

---
