# generative_ai
Testing and testing different techniques for generative AI.

----
## Markov chains
Trains on the given data and tries to generate similar sentences.  
  
To run:  
```shell
# Default values: k=3, chain_length=100,  Dataset= ./data/Sherlock_Holmes
python markov_chains.py

# python markov_chains.py <k value>  <chain_length>
python markov_chains.py 4 40

# python markov_chains.py <k value>  <chain_length> <data_path>
python markov_chains.py 4 40 ./data/John_Milton_Works
```

Reference: https://www.kdnuggets.com/2019/11/markov-chains-train-text-generation.html


----
