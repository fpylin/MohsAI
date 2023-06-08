#!/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import numpy as np
import pandas as pd
import pathlib

import tensorflow as tf
import tensorflow_hub as hub
import multiprocessing as mp

from sklearn.utils import class_weight

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator

tf.executing_eagerly()


from optparse import OptionParser

parser = OptionParser()

parser.usage = "%prog [options] train filename.model img_data_dir\n\
       %prog [options] predict filename.model image_file image_file ...\n\
"
parser.add_option("-l", "--hidden-layers", dest="hidden_layers", default="",
                  help="hiddern layers structure 3,4,6,... before the final softmax layer [default: %default]", metavar="LAYERS")
parser.add_option("-a", "--arch", dest="arch", default="google/tf2-preview/mobilenet_v2/feature_vector/4",
                  help="Model architecture [default: %default]", metavar="")
parser.add_option("-c", "--class-labels-file", dest="fn_class_label", default="",
                  help="Export the labels corresponding to trained model to FILE [default: model_name.labels]", metavar="FILE")
parser.add_option("-s", "--validation-split", dest="validation_split", default=0.2,
                  help="Keeping FRAC of training data for validation during training [default: %default]", metavar="FRAC")
parser.add_option("-r", "--learning-rate", dest="learning_rate", default=0.01,
                  help="Learning rate of Adam optimizer [default: %default]", metavar="FRAC")
parser.add_option("-b", "--batch-size", dest="batch_size", default=24,
                  help="Batch size of N used durign training [default: %default]", metavar="N")
parser.add_option("-e", "--epochs", dest="epochs", default=10,
                  help="Training the network with N epochs [default: %default]", metavar="N")
parser.add_option("-p", "--patience", dest="patience", default=3,
                  help="Early stopping after N epochs [default: %default]", metavar="N")
parser.add_option("-t", "--trainable", dest="trainable", default=False,
                  help="The network is trainable", action="store_true")
parser.add_option("-A", "--augmentation", dest="augmentation", default="FZRC",
                  help="Augmentation options: Flip, Zoom, Translate, Rotate, Contrast [default: %default] " )
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

(options, args) = parser.parse_args()

if len(args) < 3:
	print(options);
	print(args);
	parser.error("Wrong number of arguments")
	sys.exit(1);
	 
action    = args[0]
fn_model  = args[1]

feature_extractor_model = "https://tfhub.dev/" + options.arch

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


img_height = 224
img_width = 224

if options.arch == "inception_v3":
	img_height = 299
	img_width = 299

if action == 'train':
	data_root = args[2]
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		str(data_root),
		validation_split=options.validation_split,
		subset="training",
		seed=0,
		image_size=(img_height, img_width),
		batch_size=options.batch_size)

	class_names = np.array(train_ds.class_names)
	num_classes = len(class_names)
	print(class_names)

	normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./256)
	train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

	AUTOTUNE = tf.data.AUTOTUNE
	train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
	
	feature_extractor_layer = hub.KerasLayer(
		feature_extractor_model, input_shape=(img_width, img_height, 3), trainable=options.trainable) # False

	model = tf.keras.Sequential()
	model.add( tf.keras.layers.experimental.preprocessing.RandomZoom( height_factor=(-0.1, 0.1), fill_mode="constant", input_shape=(img_height, img_width, 3)) )  
	model.add( tf.keras.layers.experimental.preprocessing.RandomRotation(1.0) )
	model.add( tf.keras.layers.experimental.preprocessing.RandomContrast(0.1) )
	model.add( tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical") )
	
	model.add(feature_extractor_layer);
	
	layers_def = options.hidden_layers
	layers_struct = [ int(l) for l in layers_def.split(',') if len(l) > 0 ]

	for l in layers_struct:
		model.add( tf.keras.layers.Dense(l, activation="relu") )
	
	model.add( tf.keras.layers.Dense(num_classes, activation="softmax", name="output") )
		
	model.summary()

 
	model.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
		metrics=['acc'])

	fit_callbacks = [EarlyStopping(monitor='loss', patience=int(options.patience), restore_best_weights=True)]

	y_train = np.concatenate([y for x, y in train_ds], axis=0)
	y_classes = np.unique(y_train)
	class_weights = class_weight.compute_class_weight('balanced', classes=y_classes, y=y_train)
	class_weights = {i : class_weights [i] for i in range( len(class_weights) )}

	history = model.fit(train_ds, epochs=int(options.epochs), class_weight=class_weights, callbacks=[fit_callbacks])
	
	model.save(fn_model, save_format="h5")
	
	fn_class_label = options.fn_class_label
	
	if not fn_class_label:
		fn_class_label = fn_model + '.labels'
		
	with open( fn_class_label, 'w' ) as f:
		f.write( "\n".join(class_names) )
		f.close()
	
elif action == 'predict':
	tf.config.set_visible_devices([], 'GPU')
        
	if ( args[2].find('index:') != -1 ) :
		index_filename = args[2][6:];
		f = open(index_filename, "r")
		images_to_test = [x.strip() for x in f.readlines()]
		f.close()
	else :
		images_to_test = args[2:] 
	
	loaded_model = tf.keras.models.load_model(fn_model, custom_objects={'KerasLayer':hub.KerasLayer})

	testdf=pd.DataFrame( { "fn": images_to_test }, dtype=str)
	test_datagen=ImageDataGenerator(rescale=1./255.)
	test_generator=test_datagen.flow_from_dataframe(dataframe=testdf, directory=".", x_col="fn", y_col=None, batch_size=8, shuffle=False, class_mode=None, target_size=(img_height,img_width) )

	test_generator.reset()

	predictions = loaded_model.predict ( test_generator )

	nc = predictions.shape[1]

	i=0
	for pred_dict in predictions:
		p = list( map(lambda c: pred_dict[c], range(nc)) )
		print(*p, sep="\t")
		i += 1
	
