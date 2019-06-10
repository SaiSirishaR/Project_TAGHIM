import os



#data_folder = '/home/siri/backup_on8june/Documents/Projects/Interspeech_2019/Data/baseline_40_samples/input_full'
#files = sorted(os.listdir(data_folder))
#print(len(files))
#print(data_folder.index([ s for s in sorted(os.listdir(data_folder))]))
#print([ s for s in sorted(os.listdir(data_folder))])
#for i in range(0, len(sorted(os.listdir('.')))):
#   print("file is", sorted(os.listdir(data_folder[i])))
input_files = (filename for filename in sorted(os.listdir('/home/siri/backup_on8june/Documents/Projects/Interspeech_2019/Data/baseline_40_samples/input_full')))
print("len is", input_files)
#for i, input_file  in enumerate(input_files):

#    print("input file is", input_file)

