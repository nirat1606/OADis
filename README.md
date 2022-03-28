# Disentangling Visual Embeddings for Attributes and Objects (OADis)

This repository provides dataset splits and code for Paper: Disentangling Visual Embeddings for Attributes and Objects, published at CVPR 2022.

### Disentangling Visual Embeddings for Attributes and Objects, CVPR 2022
[Nirat Saini](https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AJsN-F4kgg1kbcLx0j2dkvo5bGoQb9BU8bNEaEkiOirw72JFqU1cdNGVo3r8KTG7pq0yHTgIZ1M6jqtUUbXRAz_6YPTAeJjMwA&user=VsTvk-8AAAAJ),
[Khoi Pham](https://scholar.google.com/citations?user=o7hS8EcAAAAJ&hl=en),
[Abhinav Shrivastava](http://www.cs.umd.edu/~abhinav/)
 

# VAW-CZSL Dataset Setup

We provide compositional splits for Generalized CZSL, following prior works:
To Download the dataset, first download pre-requisites:
1. The VAW-dataset from the website: [VAW](https://github.com/adobe-research/vaw_dataset).
2. The VAW-CZSL folder: [VAW-CZSL](https://drive.google.com/drive/folders/1CalwDXkkGALxz0e-aCFg9xBmf7Pu4eXL?usp=sharing). This folder has a jupyter notebook ```vaw_dataset_orig.ipynb```, and folder named ```compositional-split-natural```. 
 - ```compositional-split-natural```: lists attribute-object pairs for each split [training, validation and testing].
 - ```vaw_dataset_orig.ipynb``` explains the steps for creation of splits, more details can also be found in Supplementary material.
3. Some images are  part of [Visual Genome](https://visualgenome.org/), and can be downloaded from the official website. We use Visual Genome to get more attribute labels.
