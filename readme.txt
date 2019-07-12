the current VAE works for frame wise inputs.......need to make it for sequence and batch wise
do the padding and packing in collate function --> https://github.com/r9y9/tacotron_pytorch/blob/master/train.py

vcvae_v2 --> returning index info batch wise
ToDo
Add padding in the collate function of dataloader
did it in vcvae_v3.py
vcvae_v3 has training and validation--> need to work on reducing the error


July 12

vcvae_v5 modified dataset class 00> was taking more time for data loading as all the  files were being loaded evry time we cal the dataset class
