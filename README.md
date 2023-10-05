# End-to-End LLMOps for open LLMs on Amazon SageMaker

This repository provides an end-to-end example of using LLMOps practices on Amazon SageMaker for large language models (LLMs). 
The repository demonstrates a sample LLMOps pipeline for training, optimizing, deploying, monitoring, and managing LLMs on SageMaker using infrastructure as code principles.

Currently implemented:

- [Training and deploying LLMs on SageMaker](./notebooks/train-deploy-llm.ipynb)
- Optimizing LLMs with Quantization _(coming soon)_
- LLMOps pipeline for training, optimizing, and deploying LLMs on SageMaker _(coming soon)_
- Monitoring and managing LLMs with CloudWatch _(coming soon)_

## Contents

The repository currently contains:

- `scripts/`: Scripts for training and deploying LLMs on SageMaker
- `notebooks/`: Examples and tutorials for using the pipeline

## Pre-requisites

Before we can start make sure you have met the following requirements

* AWS Account with quota
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed
* AWS IAM user [configured in CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) with permission to create and manage ec2 instances


## Contributions

Contributions are welcome! Please open issues and pull requests.

## License

This repository is licensed under the MIT License.