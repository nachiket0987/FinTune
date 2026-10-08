# Software Requirements Specification (SRS)

## 1. System Overview & Scope
FinTune is a production-grade benchmarking and fine-tuning framework designed for parameter-efficient fine-tuning (PEFT) of Large Language Models (LLMs) on financial datasets. Specifically, FinTune leverages various Low-Rank Adaptation methods (LoRA, QLoRA, DoRA, rsLoRA) to enhance the performance of base models (such as Llama 3.1 8B/70B Instruct, DeepSeek V3, GPT-4o, and Gemini 2.0 FL) on 19 specific financial datasets across sentiment analysis, XBRL analysis, reporting, and examination preparation tasks. 

Operating as a CLI-based Python framework, FinTune relies on the Axolotl framework for robust fine-tuning orchestration, supports federated learning via the Flower framework, and utilizes TensorBoard for progress monitoring. The system is designed to provide cost-effective fine-tuning (under $100 on 4x A5000 GPUs) while ensuring high accuracy improvements across financial domains.

## 2. User Roles & Access Control Matrix

| Role | Access Level | Responsibilities |
|---|---|---|
| **Administrator** | Full | Full system access, configuration management (modifying `finetune_configs.json`), model deployment, API key management, and overall system maintenance. |
| **Researcher** | High | Initiating fine-tuning jobs, running evaluations, managing LoRA adapters (`lora_adapters/`), and monitoring TensorBoard metrics. |
| **Data Engineer** | Medium | Processing datasets (via scripts in `data/`), format conversion (JSONL), and data validation. |
| **Analyst** | Read-Only (Execution) | Running inference using pre-trained adapters in test environments (`test/`), viewing reports, and interacting with the Gradio demo. |

## 3. Functional Requirements

### Data Management
- **FR-001: Dataset Loading** - The system must load financial datasets strictly in JSONL format, supporting both local filesystem paths and HuggingFace Hub repository IDs.
- **FR-002: Dataset Validation** - The system must validate the structure of the loaded JSONL datasets against a predefined schema before initiating any fine-tuning process.
- **FR-003: Dataset Preprocessing** - The system shall tokenize datasets according to the specified base model's tokenizer requirements, enforcing max sequence lengths.

### Configuration & Setup
- **FR-004: LoRA Configuration** - The system must accept and validate a JSON configuration file (`finetune_configs.json`) specifying LoRA parameters including rank (r), alpha, bits, target modules, and learning rate.
- **FR-005: Quantization Support** - The system must support loading base models in 4-bit (NF4), 8-bit, and 16-bit (fp16/bf16) precisions via BitsAndBytes.
- **FR-006: Environment Setup** - The system must verify the availability of CUDA 11.8+ and sufficient GPU memory before execution.

### Fine-Tuning Execution
- **FR-007: Fine-tuning Orchestration** - The system must orchestrate the training process using the Axolotl framework.
- **FR-008: Multi-GPU Support** - The system must support distributed training across multiple GPUs using DeepSpeed.
- **FR-009: LoRA Method Selection** - The system must allow users to select between standard LoRA, QLoRA, DoRA, and rsLoRA methods.
- **FR-010: Federated Learning Support** - The system must provide a mechanism to perform federated LoRA fine-tuning utilizing the Flower framework.
- **FR-011: Checkpointing** - The system must save training checkpoints at configurable intervals to allow for recovery and evaluation.
- **FR-012: Metric Logging** - The system must log training loss, learning rate, and other relevant metrics to TensorBoard in real-time.

### Inference & Evaluation
- **FR-013: Inference Engine Integration** - The system must support inference using local HuggingFace models, as well as external APIs for OpenAI, Gemini, and Anthropic models.
- **FR-014: Adapter Loading** - The system must successfully attach pre-trained LoRA weights from the `lora_adapters/` directory to the base models for inference.
- **FR-015: Evaluation Metrics** - The system must compute and output evaluation metrics including Accuracy, F1-Score, and BERTScore for task predictions.
- **FR-016: Batch Inference** - The system must support batch processing of test sets to generate predictions efficiently.

### Adapter Management
- **FR-017: Adapter Saving** - The system must extract and save only the adapter weights (not the full model) upon completion of training.
- **FR-018: HuggingFace Hub Integration** - The system must allow authenticated users to push trained adapters to the HuggingFace Hub.
- **FR-019: Adapter Versioning** - The system should automatically timestamp and version saved adapters to prevent overwrites.

### Reporting & Demo
- **FR-020: Report Generation** - The system must generate summary reports of fine-tuning runs and evaluation results.
- **FR-021: Gradio Demo Support** - The system must provide a Gradio-based interface deployable on HuggingFace Spaces for interactive testing of the fine-tuned models.

*(Additional reserved functional requirements FR-022 to FR-025+ for future domain-specific tasks and multi-modal expansions).*

## 4. Business Rules & Field-Level Data Validation Rules

### Dataset JSONL Schema
Every row in the training and testing JSONL files must adhere strictly to the conversational or instruction format:
- `instruction` (string, required): The task prompt or question.
- `input` (string, optional): Additional context, such as a financial statement excerpt.
- `output` (string, required): The expected ground-truth response.
*Validation:* If a row is missing the `instruction` or `output` fields, the data processing script must flag an error and halt execution.

### Configuration JSON Schema Validation
The `finetune_configs.json` must be validated against the following rules:
- `lora_r`: Integer > 0 (typically 8, 16, 32, 64)
- `lora_alpha`: Integer > 0
- `learning_rate`: Float between $1e^{-6}$ and $1e^{-3}$
- `batch_size`: Integer > 0
- `quantization`: String in `["4bit", "8bit", "none"]`
*Validation:* Invalid types or out-of-range values will cause the CLI to exit immediately with a descriptive error message.

## 5. Authentication & Authorization Architecture

- **HuggingFace Integration:** Authentication for downloading gated base models (e.g., Llama 3.1) and uploading adapters is handled via the `HF_TOKEN` environment variable. The token must have write access for push operations.
- **API Key Management:** Inference against proprietary models requires valid API keys stored securely in environment variables: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`.
- **System Access:** As a CLI-based tool, access control is inherently tied to the host OS user permissions. Directory-level permissions should enforce the roles specified in the Access Control Matrix.

## 6. Error Handling, Resilience & Edge Cases

- **GPU Out-Of-Memory (OOM):** If an OOM error occurs, the system will catch the exception, output current memory statistics, and suggest reducing the batch size or increasing gradient accumulation steps.
- **Checkpoint Recovery:** In the event of a crash, the system allows resuming training from the latest saved checkpoint stored in the output directory, preventing the loss of previous progress.
- **Invalid Configurations:** The system performs pre-flight checks on configurations. If an invalid parameter (e.g., mismatched LoRA target modules for a specific architecture) is detected, the run aborts before allocating GPU memory.
- **Missing Datasets/Adapters:** Explicit `FileNotFoundError` handling ensures users are notified precisely which path or Hub repository is missing or inaccessible.
- **API Rate Limits:** For external inference engines (OpenAI, Gemini, Anthropic), the system implements exponential backoff and retry logic to handle rate limit (HTTP 429) errors gracefully.

## 7. Security, Privacy & Encryption Specifications

- **API Key Storage:** API keys and access tokens must never be hardcoded or logged. They must be loaded dynamically from secure `.env` files or native OS secret managers.
- **Model Weight Integrity:** Downloaded base models and uploaded adapters rely on SHA256 checksum validations provided by the HuggingFace Hub to ensure weights have not been tampered with.
- **Data Privacy:** Financial datasets processed locally remain on the host machine. When using external APIs for inference (Analyst role), users must be warned if sending sensitive or proprietary financial data, as per the respective provider's terms of service.
- **Federated Learning Security:** In federated setups (via Flower), model updates (gradients/adapter weights) are transmitted instead of raw data, preserving data locality and privacy. Secure RPC channels (gRPC with TLS) should be used for client-server communication.

## 8. Non-Functional Performance Thresholds

- **Fine-Tuning Throughput:** The system must achieve a training throughput of approximately ~1,000 samples per hour on a distributed setup of 4x NVIDIA RTX A5000 GPUs.
- **Inference Latency:** Local inference latency for an 8-bit quantized 8B parameter model must average < 500ms per generated token.
- **Memory Footprint:** 
  - 8-bit quantization fine-tuning must fit within 24GB VRAM per GPU.
  - 4-bit quantization fine-tuning must fit within 16GB VRAM per GPU.
- **Uptime SLA for Inference Server:** If deployed as an API or Gradio space, the expected uptime SLA is 99.5%.

## 9. System Acceptance Criteria

To be considered production-ready, FinTune must pass the following end-to-end test scenarios:
1. **E2E Training (QLoRA):** Successfully load a 1,000-sample JSONL dataset, fine-tune Llama 3.1 8B using 4-bit QLoRA, and save the adapter without OOM errors on a single 24GB GPU.
2. **E2E Evaluation:** Successfully load the trained adapter from Scenario 1, run inference on a 200-sample test set, and output Accuracy and BERTScore metrics to a file.
3. **Multi-GPU DeepSpeed:** Successfully orchestrate a fine-tuning run across 4 GPUs, verifying that workload is distributed evenly and checkpoints are synchronized.
4. **API Integration:** Successfully route an evaluation batch to the OpenAI API, handle any rate limits via backoff, and return the aggregated results.
5. **Federated Execution:** Successfully spin up a Flower server and 2 local clients, complete 3 rounds of federated training, and aggregate the resulting LoRA adapter.
