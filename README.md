# Transformer Implementation From Scratch Following Keras Guidelines

We searched through numerous tutorial including the official TensorFlow tutorial ... 
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
Downloading and preparing dataset 124.94 MiB (download: 124.94 MiB, generated: Unknown siz
e, total: 124.94 MiB) to /home/rstudio/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0
.0...
Extraction completed...: 100%|████████████████████████████████████████████████████████████
███████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.
46s/ file]
Dl Size...: 100%|█████████████████████████████████████████████████████████████████████████
████████████████████████████████████████████████████████████████| 124/124 [00:04<00:00, 27
.79 MiB/s]
Dl Completed...: 100%|████████████████████████████████████████████████████████████████████
████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4
.46s/ url]
Dataset ted_hrlr_translate downloaded and prepared to /home/rstudio/tensorflow_datasets/te
d_hrlr_translate/pt_to_en/1.0.0. Subsequent calls will reuse this data.                             
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
Epoch 14/20                                                                                                                                                                                   
700/700 [==============================] - 46s 61ms/step - loss: 1.6252 - accuracy: 0.6561 - SequenceAccuracy: 0.6504 - val_loss: 2.2770 - val_accuracy: 0.6016 - val_SequenceAccuracy: 0.5974
Epoch 15/20
700/700 [==============================] - 47s 61ms/step - loss: 1.5517 - accuracy: 0.6667 - SequenceAccuracy: 0.6612 - val_loss: 2.2690 - val_accuracy: 0.6036 - val_SequenceAccuracy: 0.5961
Epoch 16/20
700/700 [==============================] - 47s 62ms/step - loss: 1.4866 - accuracy: 0.6769 - SequenceAccuracy: 0.6717 - val_loss: 2.2885 - val_accuracy: 0.6053 - val_SequenceAccuracy: 0.6034
Epoch 17/20
700/700 [==============================] - 47s 62ms/step - loss: 1.4315 - accuracy: 0.6853 - SequenceAccuracy: 0.6803 - val_loss: 2.2929 - val_accuracy: 0.6030 - val_SequenceAccuracy: 0.6000
Epoch 18/20
700/700 [==============================] - 46s 61ms/step - loss: 1.3773 - accuracy: 0.6939 - SequenceAccuracy: 0.6885 - val_loss: 2.2870 - val_accuracy: 0.6073 - val_SequenceAccuracy: 0.6037
Epoch 19/20
700/700 [==============================] - 46s 61ms/step - loss: 1.3300 - accuracy: 0.7016 - SequenceAccuracy: 0.6967 - val_loss: 2.2925 - val_accuracy: 0.6068 - val_SequenceAccuracy: 0.6027
Epoch 20/20
700/700 [==============================] - 46s 61ms/step - loss: 1.2861 - accuracy: 0.7082 - SequenceAccuracy: 0.7033 - val_loss: 2.3137 - val_accuracy: 0.6102 - val_SequenceAccuracy: 0.6076

```

Notice that the data will only be downloaded and prepared the first time it is accessed.

