import os
from maxcut import MaxCut

instances_directory = 'instances/'
sub_directory = ''

def main(instances_directory, sub_directory):
    files = os.listdir(instances_directory + sub_directory)
    instances = []
    for f in files:
        instances.append(MaxCut(f))

    # TODO
    # Call solvers
    

if __name__ == '__main__':
    main(instances_directory, sub_directory)