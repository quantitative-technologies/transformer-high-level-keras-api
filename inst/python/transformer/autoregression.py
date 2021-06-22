import tensorflow as tf


def autoregress(model, input, delimiters, max_length):
    delimiters = delimiters[0]
    decoder_input = [delimiters[0]]
        
    output = tf.expand_dims(decoder_input, 0)

    done = False
    while not done:
        preds = model({'encoder_input': tf.expand_dims(input, 0), 'decoder_input': output})

        prediction = preds[:, -1, :]
        pred_id = tf.argmax(prediction, axis=-1) if \
            tf.shape(output)[1] < max_length - 1 else tf.expand_dims(delimiters[1], 0)

        done = pred_id == delimiters[1]

        output = tf.concat([output, tf.expand_dims(pred_id, 0)], axis=-1)

    return tf.squeeze(output, axis=0)


def translate(model, input, tokenizers, max_length):
    """
    Translate an input sentence to a target sentence using a model

    """
    input_encoded = tokenizers.inputs.tokenize([input])[0]

    if len(input_encoded) > max_length:
        return None

    prediction = autoregress(model, 
                             input_encoded, 
                             delimiters=tokenizers.targets.tokenize(['']),
                             max_length=max_length)
    prediction_decoded = tokenizers.targets.detokenize([prediction]).numpy()[0][0].decode('utf-8')

    return prediction_decoded

