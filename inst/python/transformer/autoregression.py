import tensorflow as tf
from .mask import create_look_ahead_mask


def autoregress(model, input, delimiters, max_length):
    delimiters = delimiters[0]
    decoder_input = [delimiters[0]]
        
    output = tf.expand_dims(decoder_input, 0)

    done = False
    while not done:
        preds = model({'encoder_input': tf.expand_dims(input, 0), 'decoder_input': output})

        prediction = preds[:, -1, :]
        #pred_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32) \
        pred_id = tf.argmax(prediction, axis=-1) \
            if output.shape[1] < max_length - 1 else tf.expand_dims(delimiters[1], 0)

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
    #prediction_decoded = tokenizers.targets.lookup([prediction])
    prediction_decoded = tokenizers.targets.detokenize([prediction]).numpy()[0][0].decode('utf-8')

    return prediction_decoded


# def evaluate_single(model, input, tokenizers, max_length):
#     # adding the start and end token for the input
#     inp_sentence = tokenizers['inputs'].encode(input, delimit=True)
#     encoder_input = tf.expand_dims(inp_sentence, 0)
#
#     # target start token is the first word
#     delimiters = tokenizers['targets'].encode('', delimit=True)
#     decoder_input = [delimiters[0]]
#     output = tf.expand_dims(decoder_input, 0)
#
#     for i in range(max_length):
#         #enc_padding_mask, combined_mask, dec_padding_mask = _create_masks(encoder_input, output)
#
#         # predictions.shape == (batch_size, seq_len, vocab_size)
#         inputs = dict(encoder_input=encoder_input, decoder_input=output)
#         predictions = model(inputs, False)
#
#         # select the last word from the seq_len dimension
#         predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
#
#         predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
#         # concatenate the predicted output
#         output = tf.concat([output, predicted_id], axis=-1)
#
#         # return the result if the predicted_id is equal to the end token
#         if predicted_id == delimiters[1]:
#             break
#
#     output = tf.squeeze(output, axis=0)
#     prediction = tokenizers['targets'].decode(output, delimit=True)
#     return prediction, output #, attention_weights


# def _create_masks(inputs, targets):
#     # Encoder padding mask
#     enc_padding_mask = create_padding_mask(inputs)
#
#     # Used in the 2nd attention block in the decoder.
#     # This padding mask is used to mask the encoder outputs.
#     dec_padding_mask = create_padding_mask(inputs)
#
#     # Used in the 1st attention block in the decoder.
#     # It is used to pad and mask future tokens in the input received by
#     # the decoder.
#     look_ahead_mask = create_look_ahead_mask(tf.shape(targets)[1])
#     dec_target_padding_mask = create_padding_mask(targets)
#     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
#
#     return enc_padding_mask, combined_mask, dec_padding_mask
