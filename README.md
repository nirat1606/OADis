# Disentangling Visual Embeddings for Attributes and Objects (OADis)
This repository provides dataset splits and code for Paper:
### Disentangling Visual Embeddings for Attributes and Objects, CVPR 2022
[Nirat Saini](https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AJsN-F4kgg1kbcLx0j2dkvo5bGoQb9BU8bNEaEkiOirw72JFqU1cdNGVo3r8KTG7pq0yHTgIZ1M6jqtUUbXRAz_6YPTAeJjMwA&user=VsTvk-8AAAAJ),
[Khoi Pham](https://scholar.google.com/citations?user=o7hS8EcAAAAJ&hl=en),
[Abhinav Shrivastava](http://www.cs.umd.edu/~abhinav/)
 
# VAW-CZSL Dataset 
We provide compositional splits for Generalized CZSL, following prior works:
The dataset and splits can be downloaded from: [VAW-CZSL](https://drive.google.com/drive/folders/1CalwDXkkGALxz0e-aCFg9xBmf7Pu4eXL?usp=sharing). This folder has a jupyter notebook ```vaw_dataset_orig.ipynb```, and folder named ```compositional-split-natural```.  The folder also has ```metadata``` file which splits image ids for each split.
 - ```compositional-split-natural```: lists attribute-object pairs for each split [training, validation and testing]. Images folfder has all relevant images used in VAW-CZSL dataset. 
 - ```vaw_dataset_orig.ipynb``` explains the steps for creation of splits, more details can also be found in Supplementary material. This file build the dataset splits from scratch. 
 
For building split files and metedata files from scratch, you need
1. The VAW-dataset from the website: [VAW](https://github.com/adobe-research/vaw_dataset).
2. Some images are part of [Visual Genome](https://visualgenome.org/), and can be downloaded from the official website. 



## Code Instructions:
To run OADis for MIT-States Dataset:
```
Training:
python train.py --cfg config/mit-states.yml

Testing:
python test.py --cfg config/mit-states.yml --load mit_final.pth
```
Please update location paths for dataset and logs in the mit-states.yml

Similar instructions can be used for other datasets: UT-Zappos and VAW-CZSL.

The code works well, and is tested for:```
Pytorch - 1.6.0+cu92
Python - 3.6.12
tensorboardx - v2.4
```


### For more qualitative results and details, please refer to webpage [Project Page](https://www.cs.umd.edu/~nirat/OADis/)
Please email [Nirat](nirat@umd.edu) for questions and concerns.


## Citation
Please cite our CVPR 2022 paper if you use the this repo for OADis.
```
@InProceedings{Saini_2022_CVPR,
    author    = {Saini, Nirat and Pham, Khoi and Shrivastava, Abhinav},
    title     = {Disentangling Visual Embeddings for Attributes and Objects},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {13658-13667}
}
```
