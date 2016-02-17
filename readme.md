Simple Greedy Layerwise RNN for Language Modelling
===

I noticed the other day that you can transduce the information in a shallow (for now) MLP quite a long way by copying out the hidden units' nodes and treating them as data, exactly like an Elman net, which is a simple recurrent net which is usually treated as a special architecture but is formally completely equivalent to just doing what I said. You could "unroll" in that fashion for 50 layers and it will perform OK on the last layer, even if you train only in a layerwise greedy fashion.

But the thing I made was not a full Elman net, since it only fed the hidden layer back to itself, not adding any more sequential input. Therefore, it wasn't a recurrent net, just a neat trick.

Given that, could you try to "unroll" an Elman net in that greedy layerwise fashion that seemed to be so successful at transducing information through many, many, many layers? It turns out that the answer is yes. And here's some results for language modelling based upon that intuition.
