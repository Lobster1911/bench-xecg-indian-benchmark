# bench-xecg

Official repository of "BenchECG and xECG: a benchmark and baseline for ECG foundation models"

📄 Paper: [arXiv](https://arxiv.org/abs/2509.10151)

🤗 Model: [HuggingFace](https://huggingface.co/riccardolunelli/xECG_base_model_v1)

## How to use xECG

We share the weights for our xECG on [HuggingFace](https://huggingface.co/riccardolunelli/xECG_base_model_v1) along with some convinient classes for downstream tasks. 

We strongly suggest to use the xECG class from there, as we simplified the code: in this repository it might be less intuitive how to use xECG.

## How to evaluate on BenchECG

Our code comes with a convenient `models.BaseModel` class. This class has some methods that are used by the trainer to setup the optmizer: defining different learning rates for different part of the network (e.g. layerwise decay).

`models.BaseModel` assumes models extending it are composed by a pre-trained core and a head appended to it for the downstream task.

To easily add a new model to our pipeline follow these steps:
- Your model should inherit from `models.BaseModel`
- Add a variable on the configuration (e.g. `use_your_model`) and set it to true
- In `utils.utils` modify `get_base_model` to properly load your model. That function comes with the parameters `feature_classification`, `minute_aggregation` that will tell you wich kind of output is expected by your model. Look at `models.classification.py` for an example.
- Again in `utils.utils` add to `parse_config` the variable you chose for your model and define there `sampling_freq`, `patch_size`, and some eventual default configuration specific to your model and that will not change with different tasks (e.g. preprocessing hyperparams)

We suggest you to be authenticated with `wandb` to see the logs.

If you need to use layerwise decay: 
- Implement the functions `get_layers`, this allow to use layerwise decay
- Implement `additional_params`, again for layerwise decay define all the parameters that are not in the layer list, they will have the smallest learning rate.

### Task specific documentation

[PTB-XL](docs/ptbxl.md)

## Citation

If you use our BenchECG code, xECG model or just find our code helpful, please cite:
```
@misc{lunelli2025benchecgxecgbenchmarkbaseline,
      title={BenchECG and xECG: a benchmark and baseline for ECG foundation models}, 
      author={Riccardo Lunelli and Angus Nicolson and Samuel Martin Pröll and Sebastian Johannes Reinstadler and Axel Bauer and Clemens Dlaska},
      year={2025},
      eprint={2509.10151},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.10151}, 
}
```
