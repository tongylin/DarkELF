import sys, os, glob

def list_all(path=os.path.dirname(__file__)+"/../data/"):
    """
    List available target materials in data directory
    """

    for file in glob.glob(path+"*/"):
        print('\t',file.split("/")[-2])

def files(target,path=os.path.dirname(__file__)+"/../data/"):
    """
    List available config files and epsilon grids in data directory

    target: string
    """
    
    print('Available configuration files: ')
    for file in glob.glob(path +str(target)+"/*yaml"):
            print('\t',file.split("/")[-1])
    print(" ")
    print('Available data for epsilon: ')
    for file in glob.glob(path +str(target)+"/*dat"):
            print('\t',file.split("/")[-1])