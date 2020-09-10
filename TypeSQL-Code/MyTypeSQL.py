import sys
import torch
from typesql.training import *

# Glove embeddings 
GLOVE="glove.6B.300d.txt" 

def main():
    # print command line's arguments
    parameters=sys.argv
    
    print (parameters)
    print(torch.__version__)
    
    if(len(parameters )==1 ):
        #Code description
        description()
    else:
        #train the model
        train(parameters, GLOVE)
            
        
if __name__ == "__main__":
    main()