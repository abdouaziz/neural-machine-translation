import numpy as np
import pandas as pd 
 
def load_data(filename):
    corpus = pd.read_table(filename, names=[ 'eng' , 'french' ] )
    corpus.reset_index(level=0 , inplace=True )
    corpus.rename(columns={'index' : 'eng' , 'eng' : 'fra' , 'french' : 'num' } , inplace=True)
    corpus = corpus.drop('num' , axis=1)
    corpus = corpus.iloc[ 15500: 20000 ] 
    
    return corpus



def split_data(corpus):

    french = []
    eng = []
    for line in corpus.fra:
        french.append( '<START> ' + line + ' <END>' )  

    for line in corpus.eng:
        eng.append( line ) 

    return eng,french 

