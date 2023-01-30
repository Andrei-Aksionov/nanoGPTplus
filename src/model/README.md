# Notes about model architecture

Couple of words regarding Bigram Language Model.

In the file 'bigram_lm.py' one can find implemented architecture of bigram language model. This model is pretty straight forward. In order to easily understand what is going on it might be helpful to think how to implement the same logic, but without embedding layer.

One of the ways to achieve it is to iterate over all characters by block of two, calculate for each first character how many times each second character appears in the corpus. For instance let's say for char 'b' char 'a' appeared 4 times, char 'i' - 2 and so on. What does it give to us? Now we know relative frequency of each character and can use it for more true to live text generation.

| First char | Second char | Frequency |
-------------|-------------|------------
| b | a | 4
| b | i | 2
| a | m | 6
| a | s | 3

Let's say we are generating new text after we calculated relative frequencies. We start with char 'b', then in order to find what char is more likely to be the next we simply need to sample from the distribution (in our case its 'a': 4, 'i': 2), that means that most likely the next character after character 'b' will be 'a'.

And now we have to simply repeat the step above. The last generated char was 'a', find what is the most likely characters after character 'a' and sample from the distribution (in our case most likely 'm').

With the model in the 'bigram_lm.py' file we are trying to achieve the same logic, but with help of Embedding layer. We create this layer with square size, because we want to know the distribution of each of unique characters for each unique characters. And of course we want this distribution (values of correct features) to reflect the distribution in our corpus. In other words if for char 'a' the most popular char standing after it is 'b', we want in the embedding layer value for the row for char 'a' of column for the char 'b' to be the highest. And it can be easily achieved with cross-entropy loss and backpropagation.

| Character | a | b | ... | z |
|----| --- | --- | --- | --- |
| a  | 1   | ... | ... | ... |
| b  | ... | 1   | ... | ... |
| ...| ... | ... | 1   | ... |
| z  | ... | ... | ... | 1   |
