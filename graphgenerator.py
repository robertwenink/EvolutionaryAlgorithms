from random import Random

class GraphGenerator:

    @staticmethod
    def generateRandomGraph(nodes, maxWeight, edgeProb, problemId=''):

        if nodes < 1:
            print('please provide nodes >= 1')
            return

        if edgeProb < 0 or edgeProb > 1:
            print('Edge probability must by between 0 and 1')
            return

        if maxWeight < 0 or problemId < 0:
            print('Please provide positive numbers as arguments')
            return

        rand = Random()

        filename =  f'maxcut_random_{nodes}_{edgeProb}_{maxWeight}' if problemId == '' else f'maxcut_random_{nodes}_{edgeProb}_{maxWeight}_instance_{problemId}'
        
        with open(filename, "w") as f:
            for n1 in range(nodes):
                for n2 in range(nodes):
                    if n1 == n2:
                        continue
                    if rand.random() < edgeProb:
                        weigth = int(rand.random() * maxWeight + 1)
                        f.write(f'{n1} {n2} {weigth}')
        f.close()

    @staticmethod
    def generateHyperplane(length, width, maxWeight, edgeProb=1, problemId='', options=''):
        '''
        Generates hyperplane in shape of grid
        set options to 'tunnel' or 'donut' for curved hyperplane

        '''

        if length < 1 or width < 1:
            print('please provide length >= 1 and width >= 1')
            return
        
        if edgeProb < 0 or edgeProb > 1:
            print('Edge probability must by between 0 and 1')
            return

        if maxWeight < 0:
            print('Please provide positive numbers as arguments')
            return
        
        rand = Random()

        filename = f'maxcut_grid_{length}x{width}_{edgeProb}_{maxWeight}'
        if problemId != '':
            filename += f'_instance_{problemId}'
        if options != '':
            filename += f'_{options}'

        with open(filename, "w") as f:
            for x in range(width):
                for y in range(length):

                    node = x * length + y

                    if x > 0 and rand.random() < edgeProb:
                        weigth = int(rand.random() * maxWeight + 1)
                        left_node = (x - 1) * length + y
                        f.write(f'{left_node} {node} {weigth}\n')

                    if y > 0 and rand.random() < edgeProb:
                        weigth = int(rand.random() * maxWeight + 1)
                        bottom_node = x * length + y - 1
                        f.write(f'{bottom_node} {node} {weigth}\n')
                    
                    if options == 'tunnel' or options == 'donut':
                        if x == 0 and rand.random() < edgeProb:
                            weigth = int(rand.random() * maxWeight + 1)
                            left_node = (width - 1) * length + y
                            f.write(f'{left_node} {node} {weigth}\n')

                        if options == 'donut':
                            if y == 0 and rand.random() < edgeProb:
                                weigth = int(rand.random() * maxWeight + 1)
                                bottom_node = x * length + (length - 1)
                                f.write(f'{bottom_node} {node} {weigth}\n')
        f.close()

GraphGenerator.generateHyperplane(4, 4, 1, 1, '', 'donut')