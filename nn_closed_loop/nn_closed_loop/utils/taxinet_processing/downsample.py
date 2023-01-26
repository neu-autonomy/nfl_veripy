from PIL import Image
import numpy as np
import os

### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
stride = 16             # Size of square of pixels downsampled to one grayscale value
numPix = 16             # During downsampling, average the numPix brightest pixels in each square
width  = 256//stride    # Width of downsampled grayscale image
height = 128//stride    # Height of downsampled grayscale image

## File paths
# Change this depending on set of images being downsampled
image_dir = "smallsubset_1traj/"

dir_path = os.path.dirname(os.path.realpath(__file__))
ORIGINAL_IMAGES_PATH = dir_path + "/original_images/" + image_dir
DOWNSAMPLED_IMAGES_PATH = dir_path + "/downsampled_images/" + image_dir
#################################################

def save_downsampled_images(verbose=True):

    # import pdb; pdb.set_trace()
    if verbose:
        print("Opening: {}".format(ORIGINAL_IMAGES_PATH))
    for image_file in os.listdir(ORIGINAL_IMAGES_PATH):
        # check if the image ends with png
        if (image_file.endswith(".png")):
            img = Image.open(ORIGINAL_IMAGES_PATH+image_file)
            
            if verbose:
                print("Downsampling {}".format(image_file))
            
            img = np.array(img)

            # Remove yellow/orange lines
            mask = ((img[:,:,0].astype('float')-img[:,:,2].astype('float'))>60) & ((img[:,:,1].astype('float')-img[:,:,2].astype('float'))>30) 
            img[mask] = 0
            
            # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so 
            # values range between 0 and 1
            img = np.array(Image.fromarray(img).convert('L').resize((256, 128)))/255.0

            # Downsample image
            # Split image into stride x stride boxes, average numPix brightest pixels in that box
            # As a result, img2 has one value for every box
            img2 = np.zeros((height,width))
            for i in range(height):
                for j in range(width):
                    img2[i,j] = 255*np.mean(np.sort(img[stride*i:stride*(i+1),stride*j:stride*(j+1)].reshape(-1))[-numPix:])

            PIL_image = Image.fromarray(np.uint8(img2)).convert('RGB')
            PIL_image.save(DOWNSAMPLED_IMAGES_PATH+image_file)

if __name__ == "__main__":
    save_downsampled_images()


# trainingPaths = "/scratch/smkatz/NASA_ULI/data_GAN_focus_area/*.csv"  # Regex paths to training data csv files
# #validationPaths = "/scratch/smkatz/ApproxInput/data_val/*.csv"  # Regex paths to training data csv file
# saveFolder = '/scratch/smkatz/NASA_ULI/' # Folder to save training data
# saveName = 'SK_DownsampledDataLarger.h5'   # Name of HDF5 file to save training data

#     img = np.array(img)
    
#     # Remove yellow/orange lines
#     mask = ((img[:,:,0].astype('float')-img[:,:,2].astype('float'))>60) & ((img[:,:,1].astype('float')-img[:,:,2].astype('float'))>30) 
#     img[mask] = 0
    
#     # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so 
#     # values range between 0 and 1
#     img = np.array(Image.fromarray(img).convert('L').crop(
#         (55, 5, 360, 135)).resize((256, 128)))/255.0

#     # Downsample image
#     # Split image into stride x stride boxes, average numPix brightest pixels in that box
#     # As a result, img2 has one value for every box
#     img2 = np.zeros((height,width))
#     for i in range(height):
#         for j in range(width):
#             img2[i,j] = np.mean(np.sort(img[stride*i:stride*(i+1),stride*j:stride*(j+1)].reshape(-1))[-numPix:])

#     # Ensure that the mean of the image is 0.5 and that values range between 0 and 1
#     # The training data only contains images from sunny, 9am conditions.
#     # Biasing the image helps the network generalize to different lighting conditions (cloudy, noon, etc)
#     img2 -= img2.mean()
#     img2 += 0.5
#     img2[img2>1] = 1
#     img2[img2<0] = 0


# def getData(csv_path, verbose = True):
#     if verbose:
#         print("\nReading " + csv_path)
        
#     data = pandas.read_csv(csv_path)
#     y = np.array((data.CTE,data.HE,data.DTP)).T.astype('float32')
#     X = np.zeros([len(data),height,width]).astype('float32')
#     for i, fn in enumerate(data.filename):
#         fn_with_folder = csv_path.split(os.sep)[:-1] + [str(fn)]
#         X[i] = computeTrainingImage(Image.open('/'+os.path.join(*fn_with_folder)+'.png'))
        
#         if (verbose) and (i%1000 == 0):
#             print("\t%d of %d"%(i,len(data)))
            
#     return X, y

# def computeTrainingImage(img): 
    
#     img = np.array(img)
    
#     # Remove yellow/orange lines
#     mask = ((img[:,:,0].astype('float')-img[:,:,2].astype('float'))>60) & ((img[:,:,1].astype('float')-img[:,:,2].astype('float'))>30) 
#     img[mask] = 0
    
#     # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so 
#     # values range between 0 and 1
#     img = np.array(Image.fromarray(img).convert('L').crop(
#         (55, 5, 360, 135)).resize((256, 128)))/255.0

#     # Downsample image
#     # Split image into stride x stride boxes, average numPix brightest pixels in that box
#     # As a result, img2 has one value for every box
#     img2 = np.zeros((height,width))
#     for i in range(height):
#         for j in range(width):
#             img2[i,j] = np.mean(np.sort(img[stride*i:stride*(i+1),stride*j:stride*(j+1)].reshape(-1))[-numPix:])

#     # Ensure that the mean of the image is 0.5 and that values range between 0 and 1
#     # The training data only contains images from sunny, 9am conditions.
#     # Biasing the image helps the network generalize to different lighting conditions (cloudy, noon, etc)
#     img2 -= img2.mean()
#     img2 += 0.5
#     img2[img2>1] = 1
#     img2[img2<0] = 0
#     return img2

# # Initialize training and validation variables
# X_train = np.zeros((0,height,width))
# y_train = np.zeros((0,3))
# # X_val = np.zeros((0,height,width))
# # y_val = np.zeros((0,2))

# # Training data
# csv_paths = glob.glob(trainingPaths)
# for csv_path in csv_paths:
#     X, y = getData(csv_path)
#     y_train = np.concatenate((y_train, y))
#     X_train = np.concatenate((X_train, X))

# # Validation data
# # csv_paths = glob.glob(validationPaths)
# # for csv_path in csv_paths:
# #     X, y = getData(csv_path)
# #     y_val = np.concatenate((y_val, y))
# #     X_val = np.concatenate((X_val, X))
        
# # Save data
# if not os.path.exists(saveFolder):
#     os.makedirs(saveFolder)
# with h5py.File(os.path.join(saveFolder,saveName), 'w') as f:
#     f.create_dataset('X_train',data=X_train)
#     f.create_dataset('y_train',data=y_train)
#     # f.create_dataset('X_val',data=X_val)
#     # f.create_dataset('y_val',data=y_val)

