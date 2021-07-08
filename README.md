# Natural Machine Translation

<p align="center">
<img src="./images/image.jpeg"  width="512" height="397">
</p>

> Machine translation based on Seq2seq architecture   
> The model translate an english sentence to french sentences 
> For better understood how seq2seq work your can look [Andrew Ng course ](https://www.youtube.com/playlist?list=PL1F3ABbhcqa3BBWo170U4Ev2wfsF7FN8l) on youtube 

>Keras Tutorial Machine Translation [lstm seq2seq](https://keras.io/examples/nlp/lstm_seq2seq/)


## Installation

Clone this repo:

```sh
git clone https://github.com/abdouaziz/nmt.git

cd nmt
```

Download and preprocess the dataset :

```sh
cd app/data 
python downaload.py --path=dataset
```

## Test the machine translation



```sh
python app.py
```

 