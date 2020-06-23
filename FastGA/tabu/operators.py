from FastGA.tabu.utils import one_flip_move, exchange_move
import numpy as np
import random


class TabuSearch:
    def __init__(self, instance, upclass, alpha=3):
        # Instance
        self.edges_dict = instance.edges_dict

        # Get instance
        self.instance = instance

        # Set evaluations
        self.upclass = upclass

        # Set alpha
        self.alpha = alpha
    '''
        Returns delta_i flip gain
    '''
    def get_gain(self, index, bit_string):
        current_w = 0
        flip_w = 0
        for i in range(len(bit_string)):
            if i == index:
                None
            elif bit_string[i] == bit_string[index]:
                flip_w += self.edges_dict.get(i, {}).get(index, 0)
            else:
                current_w += self.edges_dict.get(i, {}).get(index, 0)

        return flip_w - current_w

    '''
        Creates delta array of a bit_string
    '''
    def create_delta(self, bit_string):

        delta = np.zeros(self.instance.length_genotypes)
        for i in range(len(delta)):
            delta[i] = self.get_gain(i, bit_string)

        return delta

    '''
        Returns true if the first bit string has geq fitness
    '''
    def compare(self, bit_string_1, bit_string_2):
        self.upclass.inc_eval()
        return self.instance.np_fitness(bit_string_1) > self.instance.np_fitness(bit_string_2)

    '''
        Gets best index while considering tabu list.
        Function is returns two indexes for exchange
    '''
    def get_best_index(self, delta, TL, exchange=False, bit_string=None):
        if exchange:
            dmatrix = np.zeros((len(delta), len(delta)))
            for i in range(len(delta)):
                for j in range(len(delta)):
                    if i != j:
                        dmatrix[i][j] += delta[i] + delta[j]
                        dmatrix[i][j] += self.edges_dict.get(i, {}).get(j, 0)

            indexes = np.dstack(np.unravel_index(np.argsort(-dmatrix.ravel()), dmatrix.shape))[0]
            for index in indexes:
                if TL[index[0]] == 0 and TL[index[1]] == 0:
                    if bit_string[index[0]] != bit_string[index[1]]:
                        return index[0], index[1]

        else:
            sort_delta = delta.copy()
            indexes = np.argsort(-sort_delta)
            # Check tabu and aspiration
            for index in indexes:
                if TL[index] == 0:
                    return index

    '''
        Updates delta array of a bit_string after flip
    '''
    def update_delta(self, bit_string, delta, index1, index2=-1):
        if index2 == -1:
            delta[index1] = -delta[index1]
            neighbors = self.edges_dict.get(index1,{})
            for key, value in neighbors.items():
                if bit_string[key] == bit_string[index1]:
                    delta[key] -= 2*value
                else:
                    delta[key] += 2*value
        else:
            w_ij = self.edges_dict.get(index1,{}).get(index2,0)
            delta[index1] = -delta[index1] - 2*w_ij
            delta[index2] = -delta[index2] - 2*w_ij
            neighbors1 = self.edges_dict.get(index1,{})
            neighbors2 = self.edges_dict.get(index2,{})
            for key, value in neighbors1.items():
                if bit_string[key] == bit_string[index1]:
                    delta[key] -= 2 * value
                else:
                    delta[key] += 2 * value

            for key, value in neighbors2.items():
                if bit_string[key] == bit_string[index2]:
                    delta[key] -= 2 * value
                else:
                    delta[key] += 2 * value

    '''
        Gamma is the amount of bit flips that the tabu search performs
        Alpha is the amount of operations that is done.
    '''
    def _tabu(self, bit_string, gamma=6, alpha=3):
        iter = 0
        aspir = random.randint(3,int(0.15*len(bit_string)))
        # while iter < gamma:
        NonImpN1 = 0
        # NonImpN2 = 0
        TL = np.zeros(len(bit_string))
        delta = self.create_delta(bit_string)
        # #One flip
        while NonImpN1 < alpha:
            mut_string = bit_string.copy()
            index = self.get_best_index(delta, TL)
            one_flip_move(mut_string, index)
            TL[TL.nonzero()] -= 1
            TL[index] += aspir
            self.update_delta(bit_string, delta, index)
            if self.compare(mut_string, bit_string):
                bit_string = mut_string
                NonImpN1 = 0
            else:
                NonImpN1 += 1
            iter+=1
        #Exchange
        # while NonImpN2 < alpha and iter < gamma :
        #     mut_string = bit_string.copy()
        #     index1, index2 = self.get_best_index(delta, TL, exchange=True, bit_string=bit_string)
        #     self.exchange_move(mut_string, index1, index2)
        #     self.update_delta(bit_string, delta, index1, index2)
        #     TL[TL.nonzero()] -= 2
        #     TL[index1] += aspir
        #     TL[index2] += aspir
        #     if self.compare(mut_string, bit_string):
        #         bit_string = mut_string
        #         NonImpN2 = 0
        #     else:
        #         NonImpN2 += 1
        #     iter+=2
        return bit_string

    """
        Public 'mutate' method ; can perform the mutation inplace or return a (new)
        mutated bit string.
    """
    def tabu(self, bit_string, inplace=False):
        if not inplace:
            bit_string = bit_string.copy()
        return self._tabu(bit_string, alpha=self.alpha)
