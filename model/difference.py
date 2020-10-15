import difflib


def get_difference(a: str, b: str, join=False):
    """
    Returns the difference of two sentences.
    join: set this to True if the whole combined string is needed. Default: False.
    """
    a, b = a.split(' '), b.split(' ')
    diff = difflib.ndiff(a, b)
    if join:
        result = ''.join(diff)
    else:
        result = list(diff)
    
    return result


if __name__ == '__main__':

    text_1 = "안녕 나는 사람이야"
    text_2 = "안녕하 나 사람이야 잘가"

    diff = get_difference(text_1, text_2, join=True)
    print(diff)
    diff = get_difference(text_1, text_2, join=False)
    print(diff)
