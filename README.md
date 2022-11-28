## DNA edit-distance

Repository for the class CS6219: DNA-Based Data Storage and Computation


### Project:
The clustering algorithm covered in class [1] creates binary embeddings of DNA reads
based on q-grams and uses the Hamming distance between such embeddings to
approximate the edit distance between the original DNA reads. Instead of such a static
and data-agnostic approach, one could try to learn more appropriate embeddings
through deep learning. The goal of this project is to design and evaluate learned
embeddings that achieve higher accuracy than q-gram-based embeddings and allow
for more efficient filtering of edit distance computation.

[1] C. Rashtchian, K. Makarychev, M. Rácz, S. Ang, D. Jevdjic, S. Yekhanin, L. Ceze and K.
Strauss, "Clustering billions of reads for dna data storage," Advances in Neural
Information Processing Systems (NIPS), 2017.


Clone this repository.
Then clone [this repository](https://github.com/microsoft/clustered-nanopore-reads-dataset) into the folder to get the raw data.

Run:

``` preprocess.py ```

To preprocess the data and to create the set of unordered sequenced reads, which is the input for the algorithm.
