import sys, os, glob

def list_all(path=os.path.dirname(__file__)+"/../data/"):
    """
    List available target materials in data directory
    """

    for file in glob.glob(path+"*"):
        print('\t',file.split("/")[-1])

def files(target,path=os.path.dirname(__file__)+"/../data/"):
    """
    List available config files, epsilon grids, and density of states files in data directory

    target: string
    """

    print('Available configuration files: ')
    for file in glob.glob(path +str(target)+"/*yaml"):
            print('\t',file.split("/")[-1])
    print(" ")
    print('Available data for epsilon: ')
    for file in glob.glob(path +str(target)+"/*dat"):
        if ('DoS' not in file) and ('Fn' not in file):
            print('\t',file.split("/")[-1])
    print(" ")
    print('Available data for phonon density of states: ')
    for file in glob.glob(path +str(target)+"/*DoS.dat"):
            print('\t',file.split("/")[-1])
    print(" ")
    print('Available data for Fn(omega) functions: ')
    for file in glob.glob(path +str(target)+"/*Fn.dat"):
            print('\t',file.split("/")[-1])
