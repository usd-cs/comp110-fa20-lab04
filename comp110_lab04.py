"""
Module: comp110_lab04

Practice code for working with sounds in Python.
"""
import sound


def get_max_in_range(snd, start, end):
    """
    Returns the maximum left channel value between the start and end indices
    (inclusive).

    Note that this maximum is the absolute value maximum, so -10 is consider
    larger than 6.
    """

    first = snd.get_sample(start)
    val = abs(first.get_left())

    for i in range(start+1, end+1):
        samp = snd.get_sample(i)
        left = abs(samp.get_left())
        if (left > val):
             val = left

    return val

# Put your definition of set_extremes here

jolly = sound.load_sound("jolly.wav")
jolly.play()
sound.wait_until_played()  # waits until jolly is done playing

# To test, you will need to call your function at this point and play the
# result.
