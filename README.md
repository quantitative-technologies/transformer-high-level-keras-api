# Transformer Implementation with the High-Level Keras API

We searched through numerous tutorials including the official TensorFlow tutorial ... 
but none of them followed the Keras implementation guidelines. Since this posed difficulties
when trying out our own customizations, we decided to implement from scratch following
the guidelines (to our best understanding).

The document transformer.pdf gives a detailed explanation of the both the implementation
and the our own explanation of the transformer model.

This README just describes how to use the code.

```bash
$ python -m program --help
```

gives the full list of program options. 

To train the transformer on the default Portuguese--English language pair 
for 20 epochs:

```bash
$ python -m program --epochs 20
Downloading and preparing dataset 124.94 MiB (download: 124.94 MiB, generated: Unknown size, total: 124.94 MiB) to /home/rstudio/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0.0...
Extraction completed...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.46s/ file]
Dl Size...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:04<00:00, 27.79 MiB/s]
Dl Completed...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.46s/ url]
Dataset ted_hrlr_translate downloaded and prepared to /home/rstudio/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0.0. Subsequent calls will reuse this data.                             
Creating tokenizers. Please be patient...                                                                                                                                                     
Creating tokenizers took 139.751894 seconds.
Number of input tokens: 8318
Number of target tokens: 7010
No model has been trained yet.
Epoch 1/20
700/700 [==============================] - 59s 64ms/step - loss: 6.8768 - accuracy: 0.1205 - val_loss: 5.3334 - val_accuracy: 0.2132
Epoch 2/20
700/700 [==============================] - 46s 60ms/step - loss: 4.9205 - accuracy: 0.2521 - val_loss: 4.6257 - val_accuracy: 0.2788
...
Epoch 15/20
700/700 [==============================] - 46s 61ms/step - loss: 1.5657 - accuracy: 0.6654 - val_loss: 2.2668 - val_accuracy: 0.6082
Epoch 16/20
700/700 [==============================] - 46s 60ms/step - loss: 1.5002 - accuracy: 0.6753 - val_loss: 2.2611 - val_accuracy: 0.6082
Epoch 17/20
700/700 [==============================] - 46s 60ms/step - loss: 1.4445 - accuracy: 0.6837 - val_loss: 2.2658 - val_accuracy: 0.6071
Epoch 18/20
700/700 [==============================] - 46s 60ms/step - loss: 1.3911 - accuracy: 0.6912 - val_loss: 2.2745 - val_accuracy: 0.6082
Epoch 19/20
700/700 [==============================] - 46s 61ms/step - loss: 1.3450 - accuracy: 0.6991 - val_loss: 2.2885 - val_accuracy: 0.6105
Epoch 20/20
700/700 [==============================] - 46s 61ms/step - loss: 1.2998 - accuracy: 0.7059 - val_loss: 2.3034 - val_accuracy: 0.6133
```

Notice that the data will only be downloaded and prepared the first time it is accessed.

To perform inference with the trained model, we use the `input` mode, which 
translates the sentences from the input file which is `inputs.pt_en.json` by
default, with the following format:

```json
{
  "examples":
  [
    {
      "input": "este é um problema que temos que resolver.",
      "target": "this is a problem we have to solve ."
    },
    {
      "input": "os meus vizinhos ouviram sobre esta ideia.",
      "target": "and my neighboring homes heard about this idea ."
    },
    {
      "input": "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.",
      "target": "so i \\'ll just share with you some stories very quickly of some magical things that have happened ."
    }
  ]
}
```

We also use the `--show-bleu` option to show the BLEU sentence score for each example:

```bash
$ python -m program --mode input --show-bleu
Loading checkpoint train/checkpoint.20.ckpt

Input: este é um problema que temos que resolver.
Prediction: this is a problem that we have to solve .
Ground truth: this is a problem we have to solve .
BLEU = 66.90 100.0/87.5/71.4/50.0 (BP = 0.895 ratio = 0.900 hyp_len = 9 ref_len = 10)

Input: os meus vizinhos ouviram sobre esta ideia.
Prediction: my neighbors have heard of this idea .
Ground truth: and my neighboring homes heard about this idea .
BLEU = 20.16 55.6/25.0/14.3/8.3 (BP = 1.000 ratio = 1.125 hyp_len = 9 ref_len = 8)

Input: vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.
Prediction: so i ' m going to share with you a few stories of some very magical things that happened .
Ground truth: so i \'ll just share with you some stories very quickly of some magical things that have happened .
BLEU = 17.04 70.0/36.8/11.1/2.9 (BP = 1.000 ratio = 1.000 hyp_len = 20 ref_len = 20)
```

The training output shows that the best validation loss occured at epoch 16.
The `--checkpoint` (or `-c`) flag can be used to select this checkpoint:

```bash
$ python -m program --mode input -c 16 --show-bleu
Loading checkpoint train/checkpoint.16.ckpt

Input: este é um problema que temos que resolver.
Prediction: this is a problem that we have to fix .
Ground truth: this is a problem we have to solve .
BLEU = 39.94 88.9/62.5/42.9/16.7 (BP = 0.895 ratio = 0.900 hyp_len = 9 ref_len = 10)

Input: os meus vizinhos ouviram sobre esta ideia.
Prediction: my neighbors heard about this idea .
Ground truth: and my neighboring homes heard about this idea .
BLEU = 46.71 66.7/50.0/42.9/33.3 (BP = 1.000 ratio = 1.286 hyp_len = 9 ref_len = 7)

Input: vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.
Prediction: so i ' ll very quickly share with you some stories with some very magical things that happened .
Ground truth: so i \'ll just share with you some stories very quickly of some magical things that have happened .
BLEU = 31.04 75.0/47.4/22.2/11.8 (BP = 1.000 ratio = 1.053 hyp_len = 20 ref_len = 19)
```

Indeed it does much better on the last difficult sentence. 

There is also the `evaluation` mode to calculate the corpus level BLEU 
score. This take some time as auto-regressive inference is relatively slow:

```bash
$ python -m program --mode evaluate -c 16
Loading checkpoint train/checkpoint.16.ckpt
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1193/1193 [29:09<00:00,  1.47s/it]
BLEU = 27.76 59.8/34.1/21.4/13.6 (BP = 1.000 ratio = 1.015 hyp_len = 15804 ref_len = 15563)
```

The `--resource` and `--dataset` flags are used to select a datatset from the
`tensorflow-datasets` library. Next we train the transformer on the German to 
English dataset in the `wmt_t2t_translate` resource.  This time we allow both
input and target sequences up to 60 tokens in length using the `--max-len` 
argument. It is important to `--clear` out the checkpoints from the previous 
model.

```bash
$ python -m program --dataset tr_to_en --epochs 20 --max-len 60 --clear
Downloading and preparing dataset 1.61 GiB (download: 1.61 GiB, generated: Unknown size, total: 1.61 GiB) to /home/rstudio/tensorflow_datasets/wmt_t2t_translate/de-en/1.0.0...               
Extraction completed...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [1:01:21<00:00, 920.44s/ file]
Dl Size...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1645/1645 [1:01:21<00:00,  2.24s/ MiB]
Dl Completed...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [1:01:21<00:00, 920.44s/ url]
Extraction completed...: 0 file [00:00, ? file/s]

```
