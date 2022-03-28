# Disentangling Visual Embeddings for Attributes and Objects (OADis)

This repository provides dataset splits and code for Paper: Disentangling Visual Embeddings for Attributes and Objects, published at CVPR 2022.

### Disentangling Visual Embeddings for Attributes and Objects, CVPR 2022
[Nirat Saini](https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AJsN-F4kgg1kbcLx0j2dkvo5bGoQb9BU8bNEaEkiOirw72JFqU1cdNGVo3r8KTG7pq0yHTgIZ1M6jqtUUbXRAz_6YPTAeJjMwA&user=VsTvk-8AAAAJ),
[Khoi Pham](https://scholar.google.com/citations?user=o7hS8EcAAAAJ&hl=en),
[Abhinav Shrivastava](http://www.cs.umd.edu/~abhinav/)
 

# VAW-CZSL Dataset Setup

We provide compositional splits for Generalized CZSL, following prior works:
To Download the dataset, first download the VAW-dataset from the website: [VAW](https://github.com/adobe-research/vaw_dataset)
In the folder compositional-split-natural, the pairs for training, validation and test are available.

Some images are  part of [Visual Genome](https://visualgenome.org/), and can be downloaded from the official website. We use Visual Genome to get more attribute labels. 

Moreover, since VAW is multi-label dataset (each image with object can have multiple labels), we select attributes which are most uncommon in the dataset, to focus on long-tail attributes of VAW dataset. More details on the split are provided in supplementary material. The jupyter-notebook (```vaw_dataset_orig.ipynb''') explains the steps for creation of splits.
