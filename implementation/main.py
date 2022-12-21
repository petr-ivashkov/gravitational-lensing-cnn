import sys
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb 

from loadData import loadData
from loadData import createTrainingSet
from model import init
from model import train
from model import evaluate
from model import predict_rot
from model import RotationalConv2D

dir_path = "/scratch/ge54jas/LensingProject/tum_project"


def  main():
    
    #x_train, y_train = createTrainingSet(number_of_files=3000, dir_path=dir_path, first_index=0)
    #print("Training x-data shape: {}".format(x_train.shape))
    #print("Training y-data shape: {}".format(y_train.shape))

    model = keras.models.load_model("5.0.h5", custom_objects={'RotationalConv2D': RotationalConv2D})
    #model = keras.models.load_model("3.0.h5")
    #model = init(input_shape=x_train[0].data.shape)
    model.summary()

    #train(model, x_train, y_train, batch_size=128, epochs=15, initial_epoch=0)

    x_test, y_test = createTrainingSet(number_of_files=3000, dir_path=dir_path, first_index=7000)
    #print("Testing x-data shape: {}".format(x_test.shape))
    #print("Testing y-data shape: {}".format(y_test.shape))
    
    evaluate(model, x_test, y_test)

    #model.save("5.0.h5")

    # feedforward and show a single image 
    #x_sample = loadData(dir_path=dir_path + "/lens_1", number_of_files=1000, first_index=0)[0]

    #y_predicted ,y_stddev = predict_rot(model, x_sample)
    #print(y_stddev.shape)
    #print(np.average(y_stddev))
    #print(np.std(y_stddev))

    #print(y_predicted.shape)
    #print(np.average(y_predicted))
    
    #x_image = np.transpose(x_sample[0], (2,0,1))
    #image = make_lupton_rgb(x_image[0], x_image[1], x_image[2], stretch=0.5, Q=5)
    #plt.imshow(image, origin='lower')
    #plt.show()

    return

if __name__ == "__main__":
    main()
    sys.exit()