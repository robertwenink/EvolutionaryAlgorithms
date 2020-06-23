from FastGA.fastga.utils import flip
"""
A few utilitary functions.
"""


def one_flip_move(bit_string, index):
    '''
        Executes the one flip move
    '''
    flip(bit_string, index)


def exchange_move(bit_string, index1, index2):
    '''
        Executes the exchange move
        Basically flips the indexes if they are different.
    '''
    flip(bit_string, index1)
    flip(bit_string, index2)
