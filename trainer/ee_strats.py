


def sort_decreasingly(list_of_score_dicts):
    '''Decreasingly sort the list by score'''
    return sorted(list_of_score_dicts, key = lambda i: (i["score"]), reverse=True)
