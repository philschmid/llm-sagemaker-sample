# End-to-End LLMOps for open LLMs on Amazon SageMaker

This repository provides an end-to-end example of using LLMOps practices on Amazon SageMaker for large language models (LLMs). 
The repository demonstrates a sample LLMOps pipeline for training, optimizing, deploying, monitoring, and managing LLMs on SageMaker using infrastructure as code principles.

Currently implemented:

End-to-End:
- [Train and deploy LLMs on SageMaker](./notebooks/train-deploy-llm.ipynb)
- [Train and deploy open Embedding Models on Amazon SageMaker](notebooks/train-deploy-embedding-models.ipynb)

Inference:
- [Deploy Llama 3 on Amazon SageMaker](./notebooks/deploy-llama3.ipynb)
- [Deploy Llama 3.2 Vision on Amazon SageMaker](./notebooks/deploy-llama32-vision.ipynb)
- [Deploy Mixtral 8x7B on Amazon SageMaker](./notebooks/deploy-mixtral.ipynb)
- [Deploy QwQ-32B-Preview on Amazon SageMaker](./notebooks/deploy-qwq.ipynb)
- [Scale LLM Inference on Amazon SageMaker with Multi-Replica Endpoints](notebooks/multi-replica-inference-example.ipynb)

Training: 
- [Fine-tune Llama 3 with PyTorch FSDP and Q-Lora](notebooks/train-deploy-llama3.ipynb)
- [Fine-tune LLMs in 2024 with TRL](notebooks/train-evalaute-llms-2024-trl.ipynb)

## Contents

The repository currently contains:

- `scripts/`: Scripts for training and deploying LLMs on SageMaker
- `notebooks/`: Examples and tutorials for using the pipeline
- `demo/`: Demo applications and utilities for testing deployed models
- `assets/`: Images and other static assets used in documentation

## Pre-requisites

Before we can start make sure you have met the following requirements:

* AWS Account with appropriate service quotas
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed
* AWS IAM user [configured in CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) with permission to create and manage SageMaker resources
* [Hugging Face account](https://huggingface.co/join) for accessing gated models (e.g. Llama)

## Contributions

Contributions are welcome! Please open issues and pull requests.

## License

This repository is licensed under the MIT License.