###### Day wise tasks #####

June 8

Data loading and retreving
resolved len issue

June 9

resolved get item issue

June 10

Reading the wavefiles in the function get item
Added dataloader
index thing--> 671*60 and 761*60 are stored as 60,1 which means stores info column wise (dimension wise) =====> apparently this is why i did padding before calling DataLoader in my LSTM code
DataLoader --> does batching and shuffling --> https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

1) return index --> ur input shape is changed
2) use dictionary DataLoader needs tyhe input of same shape

Task to do tranform the matrix shape before feeding it to the DataLoader
