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
    