# Transformer Implementation with the High-Level Keras API

We searched through numerous tutorials including the [official 
TensorFlow transformer tutorial](https://www.tensorflow.org/tutorials/text/transformer "Transformer model for language understanding"),
but none of them used the high-level Keras API which includes built-in methods for 
training and evaluation. Since this posed difficulties
when trying out our own customizations, we decided to implement the transformer 
from scratch following the guidelines on standardizing on Keras for on the 
high-level APIs in TensorFlow 2.0.

The document transformer.pdf gives a detailed explanation of the implementation
as well as our own in-depth description of the transformer model.

This README just demonstrates how to use the code.

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

The training output shows that the best validation loss occurred at epoch 16.
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
score. This takes some time as auto-regressive inference is relatively slow:

```bash
$ python -m program --mode evaluate -c 16
Loading checkpoint train/checkpoint.16.ckpt
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1193/1193 [29:09<00:00,  1.47s/it]
BLEU = 27.76 59.8/34.1/21.4/13.6 (BP = 1.000 ratio = 1.015 hyp_len = 15804 ref_len = 15563)
```

The `--resource` and `--dataset` flags are used to select a dataset from the
`tensorflow-datasets` library. Next we train the transformer on the German to 
English dataset in the `wmt_t2t_translate` resource.  This time we allow both
input and target sequences up to 60 tokens in length using the `--max-len` 
argument. It is important to `--clear` out the training checkpoints from the previous 
model.

```bash
$ python -m program --resource wmt_t2t_translate --dataset de-en --epochs 10 --max-len 60 --clear
Downloading and preparing dataset 1.61 GiB (download: 1.61 GiB, generated: Unknown size, total: 1.61 GiB) to /home/rstudio/tensorflow_datasets/wmt_t2t_translate/de-en/1.0.0...               
Extraction completed...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [1:01:21<00:00, 920.44s/ file]
Dl Size...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1645/1645 [1:01:21<00:00,  2.24s/ MiB]
Dl Completed...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [1:01:21<00:00, 920.44s/ url]
Extraction completed...: 0 file [00:00, ? file/s]
Dataset wmt_t2t_translate downloaded and prepared to /home/rstudio/tensorflow_datasets/wmt_t2t_translate/de-en/1.0.0. Subsequent calls will reuse this data.                                  
Creating tokenizers. Please be patient...
Creating tokenizers took 1578.562916 seconds.
Number of input tokens: 7861
Number of target tokens: 7985
No model has been trained yet.                                                                                                                                                                
Epoch 1/10                                                                                                                                                                                    
2686/2686 [==============================] - 236s 82ms/step - loss: 5.8810 - accuracy: 0.1676 - val_loss: 5.0164 - val_accuracy: 0.2178                                                       
Epoch 2/10                                                                                                                                                                                    
2686/2686 [==============================] - 224s 82ms/step - loss: 4.4499 - accuracy: 0.2751 - val_loss: 4.1136 - val_accuracy: 0.3093                                                       
...
Epoch 8/10                                                                                                                                                                                    
2686/2686 [==============================] - 224s 82ms/step - loss: 2.8028 - accuracy: 0.4847 - val_loss: 2.7983 - val_accuracy: 0.4902                                                       
Epoch 9/10                                                                                                                                                                                    
2686/2686 [==============================] - 224s 82ms/step - loss: 2.7486 - accuracy: 0.4921 - val_loss: 2.7605 - val_accuracy: 0.4933                                                       
Epoch 10/10                                                                                                                                                                                   
2686/2686 [==============================] - 223s 82ms/step - loss: 2.7038 - accuracy: 0.4983 - val_loss: 2.7299 - val_accuracy: 0.5022
```
As there is no sign over overfitting, we continue the training where we left 
off for 10 more epochs (being careful not to use `--clear`). 

```bash
$ python -m program --resource wmt_t2t_translate --dataset de-en --epochs 10 --max-len 60
Loading checkpoint train/checkpoint.10.ckpt
Epoch 1/10
2686/2686 [==============================] - 236s 83ms/step - loss: 2.6670 - accuracy: 0.5032 - val_loss: 2.7077 - val_accuracy: 0.5055
Epoch 2/10
2686/2686 [==============================] - 223s 81ms/step - loss: 2.6351 - accuracy: 0.5077 - val_loss: 2.6863 - val_accuracy: 0.5080
...
Epoch 9/10
2686/2686 [==============================] - 224s 82ms/step - loss: 2.4970 - accuracy: 0.5272 - val_loss: 2.5956 - val_accuracy: 0.5223
Epoch 10/10
2686/2686 [==============================] - 224s 82ms/step - loss: 2.4852 - accuracy: 0.5291 - val_loss: 2.5941 - val_accuracy: 0.5231
```

If `--show-bleu` is specified in `evaluation` mode, then the program iterates
through the validation set, making a prediction for each example and then
computing its BLEU score.

```bash
$ python -m program --resource wmt_t2t_translate --dataset de-en --max-len 60 --mode evaluate --show-bleu
oading checkpoint train/checkpoint.10.ckpt                                                                                                                                                   
                                                                                                                                                                                              
Input: Dies führt dazu, dass ein Spieler wie ich, die Stirn bieten muss und sein Bestes geben will.                                                                                           
Prediction: this leads to a player like i want to offer and give his best .                                                                                                                   
Ground truth: b'Which is what makes a player like me want to face up and give my best.'                                                                                                       
BLEU = 14.09 52.9/31.2/6.7/3.6 (BP = 1.000 ratio = 1.133 hyp_len = 17 ref_len = 15)                                                                                                           

Input: Wie sind Sie zu der Zusammenarbeit mit beiden gekommen?
Prediction: how do you come to cooperation with both ?
Ground truth: b'How did you end up working with them?'
BLEU = 6.27 33.3/6.2/3.6/2.1 (BP = 1.000 ratio = 1.000 hyp_len = 9 ref_len = 9)

Input: Nun sei die Zeit, das Volk an der Wahlurne entscheiden zu lassen, in welche Richtung das Land gehen solle.
Prediction: now , the time is to make the people of choice of the electoralne , which the direction of the country is to go .
Ground truth: b'Now is the time to let the population decide at the ballot box, in which direction the country should move forward.'
BLEU = 5.62 52.2/9.1/2.4/1.2 (BP = 0.917 ratio = 0.920 hyp_len = 23 ref_len = 25)

Input: Aber auch den vielen Wanderarbeitern, die das Virus durchs Land tragen.
Prediction: but also the many hiking workers who contribute to the virus by the country .
Ground truth: b'Another factor is the large number of migrant workers who carry the virus across the country.'
BLEU = 12.94 47.1/25.0/6.7/3.6 (BP = 1.000 ratio = 1.133 hyp_len = 17 ref_len = 15)

Input: Dieses Geschäftsfeld hat der Konzern längst aufgegeben, trotzdem konnte er die Registrierung abwenden.
Prediction: this business field has been launched , yet it could depart the registration .
Ground truth: b'This is a business sector that the group had long vacated, yet nonetheless managed to prevent the registration.'
BLEU = 9.38 30.0/15.8/5.6/2.9 (BP = 1.000 ratio = 1.429 hyp_len = 20 ref_len = 14)

Input: "Wenn er das Referendum ausruft, werden wir zu seinem Palast gehen und ihn stürzen", sagte der Oppositionelle Jasser Said.
Prediction: " if he examines the referendum , we will go to his palace and remember him , the opposition jacse said .
Ground truth: b'"If he calls for the referendum, we will go to his palace and overthrow him," said member of the opposition Jasser Said.'
BLEU = 40.28 66.7/42.3/32.0/29.2 (BP = 1.000 ratio = 1.227 hyp_len = 27 ref_len = 22)

Input: Bei zukünftigen diplomatischen Verhandlungen könnte sich die Hamas als Akteur erweisen, der selbst von Israel und den Vereinigten Staaten nicht herausgedrängt werden kann.
Prediction: in future diplomatic negotiations , hamas may prove to be a player who can not be repressed by israel and the united states .
Ground truth: b'In future diplomacy Hamas may emerge as an actor that cannot be shut out even by Israel and America.'
BLEU = 2.45 30.0/2.6/1.4/0.7 (BP = 0.819 ratio = 0.833 hyp_len = 20 ref_len = 24)

Input: Wir haben mit 79 Punkten abgeschlossen.
Prediction: we have completed 79 points .
Ground truth: b'We finished with 79 points.'
BLEU = 30.21 50.0/40.0/25.0/16.7 (BP = 1.000 ratio = 1.000 hyp_len = 6 ref_len = 6)

Input: Befahrbar sind gleichfalls alle Verkehrswege der zweiten und dritten Straßenklasse, einschließlich der Bergstraßen.
Prediction: the road traffic routes of the second and third roads , including the mountain roads .
Ground truth: b'All secondary and tertiary roads are also passable, including mountain roads.'
BLEU = 12.49 53.8/25.0/9.1/5.0 (BP = 0.794 ratio = 0.812 hyp_len = 13 ref_len = 16)
...
```