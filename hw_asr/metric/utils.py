from collections import defaultdict

import editdistance

#Based on seminar materials

# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    """
    calculate CER metric
    :param target_text: original text
    :param predicted_text: text from the model 
    """
    if len(target_text) == 0:
        return 1
    return editdistance.distance(target_text, predicted_text) / len(target_text)

def calc_wer(target_text, predicted_text) -> float:
    """
    calculate WER metric
    :param target_text: original text
    :param predicted_text: text from the model 
    """
    splitted_target = target_text.split(' ')
    splitted_predicted = predicted_text.split(' ')
    if len(splitted_target) == 0:
        return 1
    return editdistance.distance(splitted_target, splitted_predicted) / len(splitted_target)

def ctc_beam_search(probs, beam_size, text_encoder):
    beam = defaultdict(float)

    for prob in probs:
        beam = extend_beam(beam, prob, text_encoder)
        beam = cut_beam(beam, beam_size)
        
    sorted_beam = sorted(beam.items(), key=lambda x: -x[1])
    (result, last_char), _ = sorted_beam[0]
    result = (result + last_char).strip().replace(text_encoder.EMPTY_TOK, '')
    return result

def extend_beam(beam, prob, text_encoder):
    if len(beam) == 0:
        for i in range(len(prob)):
            last_char = text_encoder.ind2char[i]
            beam[('', last_char)] += prob[i]
        return beam

    new_beam = defaultdict(float)
       
    for (sentence, last_char), v in beam.items():
        for i in range(len(prob)):
            if text_encoder.ind2char[i] == last_char:
                new_beam[(sentence, last_char)] += v * prob[i]
            else:
                new_last_char = text_encoder.ind2char[i]
                new_sentence = (sentence + last_char).replace(text_encoder.EMPTY_TOK, '')
                new_beam[(new_sentence, new_last_char)] += v * prob[i]

    return new_beam

def cut_beam(beam, beam_size):
    return dict(sorted(beam.items(), key=lambda x: -x[1])[:beam_size])
    