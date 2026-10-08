# Product Requirements Document (PRD) — FinTune

## 1. Executive Summary & Problem Statement

**Problem Statement:**  
Training state-of-the-art financial language models like BloombergGPT is prohibitively expensive (costing upwards of $3M and 1M GPU hours) and inaccessible to most researchers, startups, and academic institutions. The financial sector requires specialized AI models for sentiment analysis, XBRL statement analysis, and compliance checking, but the barrier to entry prevents widespread innovation.

**Executive Summary:**  
FinTune is a production-grade benchmarking framework that democratizes financial AI through parameter-efficient fine-tuning (PEFT). By leveraging advanced LoRA techniques (LoRA, QLoRA, DoRA, rsLoRA, Federated LoRA), FinTune allows the fine-tuning of Large Language Models (LLMs) on 19 distinct financial datasets across 4 critical categories. FinTune achieves this at a fraction of the cost—under $100 using 4x A5000 GPUs (~$1.05/hr)—making it a game-changer for quantitative analysis, financial research, and fintech product development.

---

## 2. Target User Personas & Value Proposition Matrix

| Persona | Pain Points | Value Delivered | Success Metrics |
|---------|-------------|-----------------|-----------------|
| **Quantitative Analyst (Hedge Fund)** | High cost of custom models; slow iteration cycles for sentiment extraction on alternative data. | Low-cost, fast fine-tuning on local hardware; ability to experiment with 5 LLM families. | Time-to-deploy < 48 hours; accuracy > 85% on sentiment tasks. |
| **FinTech Startup CTO** | Limited compute budget; need for rapid proof-of-concept models without relying on costly closed APIs. | 16GB/24GB VRAM requirement allows training on consumer/affordable GPUs. | Compute cost < $100 per full fine-tune iteration. |
| **Academic NLP/Finance Researcher** | Lack of a unified framework to benchmark different PEFT methods on financial datasets. | 19 pre-processed financial datasets; built-in Axolotl orchestration and TensorBoard support. | Publication-ready evaluation metrics; 36%+ baseline accuracy gain. |
| **SMB Financial Compliance Officer** | Highly sensitive data cannot leave on-premise infrastructure; compliance checking requires domain expertise. | Support for Federated LoRA (Flower); complete local execution. | 100% data locality; 0 data leaks. |

---

## 3. Product Goals & Quantitative Success Metrics (KPIs)

* **Accuracy Improvement:** ≥36% average improvement over base models across the 19 datasets.
* **Training Cost:** <$100 for a full fine-tuning run on 4x A5000 GPUs.
* **GPU Memory Requirements:** <24GB VRAM (8-bit training) and <16GB VRAM (4-bit training).
* **Dataset Support:** Full integration of 19 datasets spanning 4 categories: General Financial Tasks, Financial Certificate Tasks, Financial Reporting Tasks, and Financial Statement Analysis Tasks.
* **Inference Latency:** <500ms per token on local hardware.
* **Domain Expertise Benchmark:** >80% pass rate equivalent on CFA exam benchmarks (vs. ~20% for base models).

---

## 4. Core MVP Features & In-Scope Capabilities

1. **Multi-Model Support:** Plug-and-play support for 5 base LLMs: Llama 3.1 8B Instruct, Llama 3.1 70B Instruct, DeepSeek V3, GPT-4o, and Gemini 2.0 FL.
2. **Comprehensive PEFT Suite:** Implementation of LoRA, QLoRA, DoRA, and rsLoRA.
3. **Federated Learning Support:** Built-in Federated LoRA using the Flower framework for privacy-preserving distributed training.
4. **Pre-packaged Financial Datasets:** 19 financial datasets pre-processed into JSONL format, categorized into 4 domain-specific tasks.
5. **Axolotl Orchestration:** JSON-based configuration (`finetune_configs.json`) mapped via Axolotl for seamless training orchestration.
6. **Quantization Support:** 4-bit and 8-bit quantization using BitsAndBytes for reduced memory footprints.
7. **Pre-trained Adapter Weights library:** Access to pre-trained LoRA adapter weights (4-bit, 8-bit, fp16) in `lora_adapters/`.
8. **Multi-Provider Evaluation Scripts:** Automated evaluation across local (HF) and API-based models (OpenAI, Gemini, Anthropic) in the `test/` directory.
9. **TensorBoard Integration:** Real-time monitoring of training loss, learning rate, and evaluation metrics.
10. **Interactive Gradio Demo:** HuggingFace Spaces compatible Gradio interface for quick inference and demonstration.
11. **Comprehensive Documentation:** Sphinx-generated docs and Markdown architecture specs for easy onboarding.

---

## 5. User Stories

1. **As a** Quantitative Analyst, **I want** to fine-tune Llama 3.1 8B using QLoRA, **so that** I can run sentiment analysis on SEC filings using a single 24GB GPU.
2. **As an** Academic Researcher, **I want** to easily toggle between DoRA and rsLoRA in the JSON config, **so that** I can benchmark their relative performance on XBRL statement analysis.
3. **As a** FinTech Startup CTO, **I want** the training cost to remain under $100 on standard cloud instances (e.g., A5000s), **so that** we can sustainably iterate on our model weekly.
4. **As a** Compliance Officer, **I want** to use Federated LoRA via Flower, **so that** multiple branches can collaboratively train a model without sharing raw sensitive customer data.
5. **As a** Machine Learning Engineer, **I want** to monitor training progress via TensorBoard, **so that** I can detect overfitting or vanishing gradients early.
6. **As a** Developer, **I want** evaluation scripts that support both HF and OpenAI APIs, **so that** I can benchmark my fine-tuned local model against GPT-4o.
7. **As a** Product Manager, **I want** a Gradio demo interface, **so that** I can showcase the fine-tuned model's capabilities to stakeholders without writing inference code.
8. **As a** Data Scientist, **I want** all 19 datasets pre-formatted into JSONL, **so that** I don't waste time on data wrangling and tokenization alignment.
9. **As a** Researcher, **I want** the system to achieve a >80% simulated pass rate on the CFA exam, **so that** I have a verifiable metric of domain-specific reasoning capability.
10. **As a** Systems Architect, **I want** DeepSpeed integration via Axolotl, **so that** I can scale up to 70B models if we acquire multi-GPU nodes in the future.
11. **As a** Data Engineer, **I want** clear dataset processing scripts in the `data/` directory, **so that** I can easily add a 20th custom dataset following the same schema.
12. **As a** Student, **I want** access to pre-trained adapter weights in `lora_adapters/`, **so that** I can run inference immediately without paying for training compute.

---

## 6. Out-of-Scope Features for MVP

1. **Web-based GUI for Training Configuration:** The MVP will rely on CLI and JSON configuration edits; no web dashboard for initiating training runs.
2. **Native Mobile Application:** The Gradio demo is web-based; no iOS/Android apps.
3. **Retrieval-Augmented Generation (RAG) Pipelines:** FinTune focuses purely on parameter-efficient fine-tuning, not vector databases or RAG architectures.
4. **Pre-training from Scratch:** The framework only supports fine-tuning of existing base LLMs.
5. **Automated Hyperparameter Optimization (AutoML):** Users must manually specify hyperparameters like learning rate, rank, and alpha in the config.
6. **Audio/Video Multimodal Fine-tuning:** Focus is strictly on text-based financial datasets (JSONL format).

---

## 7. Technical Assumptions & Constraints

* **Software Stack:** Python 3.11+, PyTorch 2.4+, CUDA 11.8+.
* **Hardware Constraints:** Minimum 16GB VRAM for 4-bit QLoRA on 8B models; 24GB VRAM for 8-bit. Multi-GPU setups (e.g., 4x A5000) are assumed for optimal training times.
* **Framework Dependencies:** Heavily relies on Hugging Face Transformers, PEFT, Axolotl, DeepSpeed, BitsAndBytes, and Flower.
* **Network Constraints:** Requires internet access during initial setup to download base models from Hugging Face Hub (unless models are pre-cached locally).
* **Data Format:** All custom datasets must strictly adhere to the JSONL conversational format expected by Axolotl.

---

## 8. Risk Management & Mitigation Matrix

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Out-of-Memory (OOM) Errors on 16GB GPUs | High | High | Enforce strict batch size limits in default configs and mandate 4-bit BitsAndBytes quantization for 16GB setups. |
| Breaking changes in Axolotl or PEFT upstream | Medium | High | Pin exact versions of all dependencies in `requirements.txt` and run CI/CD tests weekly. |
| Catastrophic forgetting during fine-tuning | Medium | Medium | Use low learning rates, optimal LoRA alpha/rank ratios, and evaluate on a diverse holdout set from the 19 datasets. |
| Data Privacy leaks in Federated Learning | Low | High | Utilize robust aggregation strategies in Flower and ensure client-side data never leaves local nodes. |
| Inferior performance on 70B models due to quantization | Low | Medium | Provide fp16 fallback documentation and benchmark QLoRA vs LoRA on 70B variants. |
| User configuration errors (JSON syntax) | High | Low | Provide comprehensive validation scripts before triggering the Axolotl pipeline. |
| HuggingFace Hub rate limiting or downtime | Low | Medium | Document offline mode procedures and encourage downloading models/datasets prior to training. |
| API rate limits during test evaluations (OpenAI/Gemini) | Medium | Low | Implement exponential backoff and retry logic in `test/` evaluation scripts. |

---

## 9. Testable Acceptance Criteria Matrix

| MVP Feature | Acceptance Criteria |
|-------------|---------------------|
| Multi-Model Support | System successfully initializes and begins training for Llama 3.1 8B, DeepSeek V3, and Llama 3.1 70B using the same generalized config structure. |
| PEFT Suite | User can successfully train a model using LoRA, QLoRA, DoRA, and rsLoRA by changing a single parameter in `finetune_configs.json`. |
| Memory Constraints | 4-bit QLoRA training of Llama 3.1 8B peaks at ≤15.5GB VRAM. 8-bit training peaks at ≤23.5GB VRAM. |
| Cost Constraint | Training logs show total wall-clock time on 4x A5000s resulting in <$100 cloud provider equivalent cost. |
| Accuracy Target | Evaluation script output shows an average >36% absolute accuracy improvement across test splits for the 19 datasets compared to base Llama 3.1 8B. |
| Dataset Integration | All 19 JSONL datasets load without parsing errors and tokenization aligns correctly with the model's chat template. |
| Federated Learning | Flower server successfully coordinates training rounds between at least 2 local client processes and aggregates the adapter weights. |
| Gradio Demo | `python app.py` launches a local web server where a user can input text and receive an inference response < 2 seconds. |
| Test Coverage | The CFA benchmark script outputs a strict pass/fail percentage, achieving >80% on the fine-tuned model. |
