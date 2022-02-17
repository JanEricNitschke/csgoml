#!/usr/bin/env python

import os
import logging
import pandas as pd
import numpy as np
import argparse
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def TransformTicksToSeconds(tick,firsttick):
    result=int((int(tick)-int(firsttick))/128)
    return result

def transformToDataFrame(JsonFormatDict):
    returnDF=pd.DataFrame(JsonFormatDict)
    FirstTick=int(returnDF.iloc[0]["Tick"])
    returnDF["Tick"]=returnDF["Tick"].apply(TransformTicksToSeconds,args=(FirstTick,))
    return returnDF

def PadToLength(position_df,time):
    if time>len(position_df):
        # Pad with last entry
        #idx = np.minimum(np.arange(time), len(position_df) - 1)
        #position_df=position_df.iloc[idx]
        position_df=position_df.reindex(range(time), fill_value=0.0, method="pad")
    else:
        # Cut if the required size is smaller.
        position_df=position_df.head(time)
    return position_df

def GenerateArrayForTokens(position_df,token,time):
    # Split the token strings into a columns so that there is one column for each integer in the token string
    # Only work with the token entry of the df
    position_df=position_df[[token]]
    # Initialize the numpy array
    return_array=np.zeros(tuple([time,len(position_df.iloc[0][token])]))
    # Transform the string into a list of the individual letters(ints) in the string
    position_df=position_df[token].apply(list)
    # Convert the dataframe to a list and back into a df to have one column for each entry of the string
    position_df=pd.DataFrame(position_df.tolist())
    # Convert the individual string into the respective integers
    # Divide numbers by 5 so they are all from 0 to 1
    position_df.applymap(lambda x: float(x)/5)
    # Pad the df to the specified length
    position_df=PadToLength(position_df,time)
    # Convert to numpy array
    #return_array=position_df.to_numpy()
    for ind, column in enumerate(position_df.columns):
        return_array[:,ind]=position_df[column].to_numpy()
    return return_array


def modifyDataFrameShape(position_df,team,time,coordinates):
    # Throw away all unneeded columns from the dataframe and then convert it into a numpy array
    # If coordinates is false than it is a 1-D array of the tokens for each timestep
    # If coordinates is true than it is a 4-D array with the first index representing the timestep, the second the team, the third the playernumber in that team and the fourth the feature.
    position_df.reset_index(drop=True, inplace=True)
    # Set length of dataframe to make sure all have the same size
    # Pad each column if set size is larger than dataframe size
    # If the full coordinates should be used then for each player their alive status as well as their x,y,z coordinates are kept, everything else (player names and tokens) is discarded.
    if coordinates:
        featurelist=["x","y","z"]
        #featurelist=["Alive","x","y","z"]
        dimensions=[time,5,len(featurelist)] # time for timesteps, 5 for players, len(feautrelist) for features
        position_df=PadToLength(position_df,time)
        if team=="BOTH":
            sides=["CT","T"]
            # Dimension with size 2 for teams
            dimensions.append(2)
        else:
            sides=[team]
            # Dimension with size 1 for team
            dimensions.append(1)

        # Only keep the information for the side(s) that should be used.
        return_array=np.zeros(tuple(dimensions))
        for side in sides:
            for number in range(1,6):
                for feature in featurelist:
                    return_array[:,number-1,featurelist.index(feature),sides.index(side)]=position_df[side+"Player"+str(number)+feature].to_numpy()
    else:
        # Remove all columns from the dataframe except for the column of interest
        # Split the token strings into a columns so that there is one column for each integer in the token string
        if team=="BOTH":
            return_array=GenerateArrayForTokens(position_df,"token",time)
        else:
            return_array=GenerateArrayForTokens(position_df,team+"token",time)
    return return_array

def getMaximumLength(position_dfs):
    lengths=position_dfs["position_df"].apply(len)
    return lengths.max()

def TransFormDataframeToArrays(dataframe):
    label_array=dataframe["Winner"].to_numpy()
    feature_array=np.zeros(tuple([len(dataframe["position_df"])]+list(dataframe.iloc[0]["position_df"].shape)))
    for index, array in dataframe["position_df"].iteritems():
        feature_array[index]=array
    return label_array, feature_array

def GetTokenModel(NodesPerLayer,input_shape):
    model = tf.keras.Sequential([
        layers.LSTM(NodesPerLayer,input_shape=input_shape),
        layers.Dense(NodesPerLayer, activation='relu'), #,kernel_regularizer=tf.keras.regularizers.l2(0.001)
        #layers.Dropout(0.4),
        layers.Dense(NodesPerLayer, activation='relu'), #,kernel_regularizer=tf.keras.regularizers.l2(0.001)
        #layers.Dropout(0.4),
        layers.Dense(1)
        ])
    return model

def GetCoordinateModel(NodesPerLayer,input_shape):
    pooling_size=(2,1)
    model = tf.keras.Sequential([
        layers.TimeDistributed(layers.Conv2D(NodesPerLayer/2,pooling_size,activation='elu',padding="same"),input_shape=input_shape),
        layers.TimeDistributed(layers.AveragePooling2D(pool_size=pooling_size,strides=(1,1))),
        layers.TimeDistributed(layers.Conv2D(NodesPerLayer,pooling_size,activation='elu',padding="same"),input_shape=input_shape),
        layers.TimeDistributed(layers.AveragePooling2D(pool_size=pooling_size,strides=(1,1))),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(NodesPerLayer),
        layers.Dense(NodesPerLayer, activation='relu'), #,kernel_regularizer=tf.keras.regularizers.l2(0.001)
        layers.Dropout(0.4),
        layers.Dense(NodesPerLayer, activation='relu'), #,kernel_regularizer=tf.keras.regularizers.l2(0.001)
        layers.Dropout(0.4),
        layers.Dense(1)
        ])
    return model


def main(args):
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument("-d", "--debug",  action='store_true', default=False, help="Enable debug output.")
    parser.add_argument("-m", "--map",  default="ancient", help="Map to analyze")
    parser.add_argument("-l", "--log",  default='D:\CSGO\ML\CSGOML\ReadTensorflowInput.log', help="Path to output log.")
    parser.add_argument("-s", "--side",  default='BOTH', help="Which side to include in analysis (CT,T,BOTH) .")
    parser.add_argument("--coordinates",  action='store_true', default=False, help="Whether to use full coordinate or just tokens.")
    parser.add_argument("-t", "--time",  default='MAX', help="How many position snapshots should be used. Either an integer or MAX.")
    parser.add_argument("--exampleid", type=int, default=22, help="For which round the position_df should be printed when debugging.")
    parser.add_argument("--randomstate", type=int, default=123, help="Random state for train_test_split")
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.DEBUG,filemode='w',format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.INFO,filemode='w',format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

    ExampleIndex=options.exampleid
    RandomState=options.randomstate

    if options.side not in ["CT","T","BOTH"]:
        logging.error("Side "+options.side+" unknown. Has to be one of 'CT','T','BOTH'!")
        sys.exit

    # Time should either an integer or MAX.
    # If it is an integer n that means that the first n timesteps will be considered for each round and the rest discarded.
    # If it is 'MAX' then we look what the longest round in the dataset is and pad all other rounds to that length.
    # Should probably do the Training,Test split before and only look at the max length of the training set. TODO
    try:
        int(options.time)
    except ValueError:
        if  not options.time =="MAX":
            logging.error("Time "+options.time+" unknown. Has to be either an integer or 'MAX'!")
            sys.exit

    # Read in the prepared json file.
    # File="D:\CSGO\Demos\Maps\\"+options.map+"\Analysis\Prepared_Input_Tensorflow_"+options.map+".json"
    File="E:\PhD\MachineLearning\CSGOData\ParsedDemos\\"+options.map+"\Analysis\Prepared_Input_Tensorflow_"+options.map+".json"
    if os.path.isfile(File):
        with open(File, encoding='utf-8') as PreAnalyzed:
            dataframe=pd.read_json(PreAnalyzed)
    else:
        logging.error("File "+File+" does not exist! Probably because map "+options.map+" does not exist!")

    logging.debug("Initial dataframe:")
    logging.debug(dataframe)


    # Only keep the label (Winner) and training feature (position_df) and discard the MatchID, Map_name and round number.
    dataframe=dataframe[["Winner","position_df"]]
    dataframe["position_df"]=dataframe["position_df"].apply(transformToDataFrame)

    logging.debug("Example for position_df dataframe entry before cleaning.")
    logging.debug(dataframe.iloc[ExampleIndex]["position_df"])

    # Get the time length to be used.
    if options.time =="MAX":
        time=getMaximumLength(dataframe)
        if time > 160:
            time=160
    else:
        time=int(options.time)
    logging.debug("Time has been set to "+str(time)+"!")

    # Transform the feature to the proper length and discard all columns that should not be used.
    dataframe["position_df"]=dataframe["position_df"].apply(modifyDataFrameShape,args=(options.side,time,options.coordinates))

    logging.debug("Dataframe after cleanup")
    logging.debug(dataframe)
    logging.debug("Example for position_df dataframe entry after cleaning.")
    logging.debug(dataframe.iloc[ExampleIndex]["position_df"])

    # Shuffle and split dataframe into training, test and validation set
    # set aside 20% of train and test data for evaluation
    train_df, test_df = train_test_split(dataframe, test_size=0.2, shuffle = True, random_state = RandomState)
    # Use the same function above for the validation set  0.25 x 0.8 = 0.2
    train_df, val_df = train_test_split(train_df, test_size=0.25, shuffle = True, random_state = RandomState)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    test_labels, test_features = TransFormDataframeToArrays(test_df)
    val_labels, val_features = TransFormDataframeToArrays(val_df)
    train_labels, train_features = TransFormDataframeToArrays(train_df)

    logging.info("Input preparation done!")
    logging.info(train_labels.shape)
    logging.info(train_features.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels))

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 40

    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    NodesPerLayer=32


    if options.coordinates:
        model=GetCoordinateModel(NodesPerLayer,train_features[0].shape)
    else:
        model=GetTokenModel(NodesPerLayer,train_features[0].shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00007),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            "accuracy",
            tf.keras.losses.BinaryCrossentropy(
                from_logits=True, name='binary_crossentropy'),
                ])

    model.summary()

    history = model.fit(
        train_dataset,
        epochs=50,
        batch_size=BATCH_SIZE,
        validation_data=val_dataset,
        callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=5),
        )

    loss, accuracy, entropy = model.evaluate(test_dataset)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = history.history

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

    # #input_tensor=tf.convert_to_tensor(dataframe)
    # #logging.info(input_tensor)


if __name__ == '__main__':
    main(sys.argv[1:])