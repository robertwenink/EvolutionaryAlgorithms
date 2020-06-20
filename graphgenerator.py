from random import Random
#
# class GraphGenerator:

#@staticmethod
def generateRandomGraph(directory, nodes, maxWeight, edgeProb, problemId=''):

    if nodes < 1:
        print('please provide nodes >= 1')
        return

    if edgeProb < 0 or edgeProb > 1:
        print('Edge probability must by between 0 and 1')
        return

    if maxWeight < 0:
        print('Please provide positive numbers as arguments')
        return

    rand = Random()

    filename =  f'{directory}/maxcut_random_{nodes}_{edgeProb}_{maxWeight}' if problemId == '' else f'{directory}/maxcut_random_{nodes}_{edgeProb}_{maxWeight}_instance_{problemId}'
    filename += '.txt'

    with open(filename, "w") as f:
        f.write(f'{nodes}\n')
        for n1 in range(nodes):
            for n2 in range(nodes):
                if n1 == n2:
                    continue
                if rand.random() < edgeProb:
                    weigth = int(rand.random() * maxWeight + 1)
                    f.write(f'{n1} {n2} {weigth}\n')
    f.close()

@staticmethod
def generateHyperplane(directory, length, width, maxWeight, edgeProb=1, problemId='', hyperplane='grid'):
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

    if hyperplane not in ['grid', 'tunnel', 'donut']:
        print('Options must be "grid", "tunnel" or "donut"')
        return

    rand = Random()

    filename = f'{directory}/maxcut_{length}x{width}_{edgeProb}_{maxWeight}'
    if problemId != '':
        filename += f'_instance_{problemId}'
    filename += f'_{hyperplane}'
    filename += '.txt'

    with open(filename, "w") as f:
        f.write(f'{width*length}\n')
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

                if hyperplane == 'tunnel' or hyperplane == 'donut':
                    if x == 0 and rand.random() < edgeProb:
                        weigth = int(rand.random() * maxWeight + 1)
                        left_node = (width - 1) * length + y
                        f.write(f'{left_node} {node} {weigth}\n')

                    if hyperplane == 'donut':
                        if y == 0 and rand.random() < edgeProb:
                            weigth = int(rand.random() * maxWeight + 1)
                            bottom_node = x * length + (length - 1)
                            f.write(f'{bottom_node} {node} {weigth}\n')
    f.close()

# GraphGenerator.generateHyperplane('instances', 20, 20, 40, 0.8, '', 'grid')
# GraphGenerator.generateRandomGraph('instances', 300, 40, 1, '')
