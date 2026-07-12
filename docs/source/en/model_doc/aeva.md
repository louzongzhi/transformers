<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on .*

# Aeva

## Overview

Aeva is a MoE Transformer built on [DeepSeek-V4](./deepseek_v4), targeting **routing stability** and **expert
specialisation** for million-context efficient intelligence. Two key innovations:

* **Gated Delta Attention Residuals** — layers are grouped into blocks; PathMoE ties routers within each block for
  consistent expert assignment, while learned erase/write/decay gates maintain cross-block memory
  ([PathMoE](https://arxiv.org/abs/2603.18297), [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)).
* **Entropy-adaptive routing** — dynamic logit scaling based on routing entropy, combined with load-balancing and ERC
  losses for expert stability and specialisation
  ([ERC](https://arxiv.org/abs/2512.23447)).

Additional: latent bottleneck (`moe_latent_size = hidden_size / 4`), XSA orthogonal projection, mHC-lite.

## AevaConfig

[[autodoc]] AevaConfig

## AevaPreTrainedModel

[[autodoc]] AevaPreTrainedModel
    - forward

## AevaModel

[[autodoc]] AevaModel
    - forward

## AevaForCausalLM

[[autodoc]] AevaForCausalLM
    - forward
