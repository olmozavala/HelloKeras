from keras.preprocessing import image
import os
from os.path import isfile, join, isdir
import numpy as np
import pickle




def readData(my_path, dc, save_container, display_images):

    if display_images:
        import pylab as pl
        import matplotlib.pyplot as plt
        from IPython import display
        import time

        display.display(pl.gcf())
        plt.show()

    print("Loading data....")
    folder_names = [f for f in os.listdir(my_path) if isdir(join(my_path, f))]
    idx_folder = 0  # This variable is used to create the output label

    tot_examples_per_class = 10  # THis part is hardcoded TODO
    dc.size = {'w':92,'h':112,'d':3}  # width, height, depth
    dc.totClasses = len(folder_names)
    # dc.train = np.zeros((tot_examples_per_class*dc.totClasses, dc.size.get('h'), dc.size.get('w'),dc.size.get('d')))
    # dc.train = np.zeros((tot_examples_per_class*dc.totClasses, dc.size.get('h'), dc.size.get('w')))
    dc.train = np.zeros((tot_examples_per_class*dc.totClasses, dc.size.get('h')* dc.size.get('w')))
    dc.train_labels = np.zeros((tot_examples_per_class*dc.totClasses, dc.totClasses))
    dc.classes = folder_names

    idx_example = 0
    # This loops will fill the variable train of the container with all the images
    for folder in folder_names:  # Iterate over folders
        print('Reading folder ', folder)
        curr_folder = my_path+'/'+folder
        file_names = [f for f in os.listdir(curr_folder) if isfile(join(curr_folder, f))]
        for file in file_names:  # Iterate over each file
            curr_file = curr_folder+'/'+file

            labels = np.zeros(dc.totClasses)  # Init temporal array with the classes
            labels[idx_folder] = 1
            dc.train_labels[idx_example,:] = labels

            temp_image = image.load_img(curr_file)  # Load the image
            # dc.train[idx_example,:,:,:] = image.img_to_array(temp_image)  # Save image as a numpy array
            # dc.train[idx_example,:,:] = image.img_to_array(temp_image)[:,:,1]  # Save image as a numpy array
            dc.train[idx_example,:] = image.img_to_array(temp_image)[:,:,1].flatten()  # Save image as a numpy array

            # Used to display a couple of images
            if display_images and idx_example < 2:
                plt.interactive(False)
                time.sleep(2)
                display.clear_output(wait=True)
                plt.clf()
                plt.imshow(dc.train[idx_example,:,:,:])
                plt.show()

            idx_example+=1

        idx_folder+=1

    # Test images
    file_names = [f for f in os.listdir(my_path) if isfile(join(my_path, f))]
    dc.test = np.zeros((len(file_names), dc.size.get('h'), dc.size.get('w'),dc.size.get('d')))
    dc.test_labels = np.zeros((len(file_names), dc.totClasses))
    labels = np.zeros(dc.totClasses)

    idx_example = 0
    for fileName in file_names:
        # TODO
        #dc.test_labelst[idx_example,:] = labels
        curr_file = my_path+'/'+fileName
        temp_image = image.load_img(curr_file)
        dc.test[idx_example,:,:,:] = image.img_to_array(temp_image)
        idx_example+=1

    with open(save_container, 'wb') as f:
        pickle.dump(dc,f)
    print("Done!!")
