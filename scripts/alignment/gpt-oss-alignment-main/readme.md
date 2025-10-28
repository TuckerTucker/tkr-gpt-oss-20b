From: https://github.com/RiddleHe/gpt-oss-alignment


# GPT-OSS-Alignment

This repo is under active development to do a bunch of interp / alignment stuff on gpt-oss-20b

## Set up

### Conda Env
```
conda create -n gptoss python=3.12 -y
conda activate gptoss

conda install -c conda-forge libstdcxx-ng -y

pip install uv
    
uv pip install -r requirements.txt
```

### Sample from vLLM

```
cd notebooks
bash vllm.sh
# then play with sample_gpt.ipynb
```

## Steering Vector

This pipeline computes the mean difference between activation vectors of unaligned responses and those of aligned responses. It then uses these activation vector to steer the outputs of the model. (Cool observation: I compute the activation of the prompt, not response!)

### Code

I will make these scripts more production grade in the coming days.

```
export CUDA_VISIBLE_DEVICES=[your device] # most stable on single device
cd steering_vector
python find_vector.py
python compute_vector_diff.py
python steer_model.py \
    --layer [19] \
    --visualize_step [10]
python analyze_vectors.py 
```

## Sparse autoencoder
We want to train and eval a full SAE on the mysterious layer 19 (which jailbreaks gpt-oss) to extract the most evil features.

The model code is taken from openai's sparse_autoencoder repo and the training / eval is built from scratch.

### Code

To begin training on a `data_dir` of a single .pt with stacked activation vectors, do:

```
cd sae
python train_sae.py [config.yaml]
python eval.py [model_path.pt] [vector.pt]
```

Enjoy!