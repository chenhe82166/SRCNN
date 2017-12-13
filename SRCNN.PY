from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def model():
	SRCNN=Sequential()
	SRCNN.add(Conv2D(filters=64,row=9,col=9,activation='relu',init='glorot_uniform',
						padding='valid',use_bias=True,input_size=(32,32,3))
	SRCNN.add(Conv2D(filters=32,row=1,col=1,activation='relu',init='glorot_uniform',
						padding='same',use_bias=True)
	SRCNN.add(Conv2D(filters=1,row=5,col=5,activation='linear',init='glorot_uniform',
						padding='valid',use_bias=True)	

	adam=Adam(lr=0.0003)
	SRCNN.compile(optimizer=adam,loss='mse',metrics=['accuracy'])
	return SRCNN
	
def predict_model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=64,row=9,col=9,activation='relu'，init='glorot_uniform',
                     padding='valid', use_bias=True, input_shape=(None, None, 3)))
    SRCNN.add(Conv2D(filters=32,row=1,col=1,activation='relu',init='glorot_uniform',
						padding='same',use_bias=True))
    SRCNN.add(Conv2D(filters=1,row=5,col=5,activation='linear',init='glorot_uniform',
						padding='valid',use_bias=True))	
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    return SRCNN
	
def train():
	srcnn_model=model()
	data,label=
	val_data,val_label=
	checkpoint=ModelCheckpoint(filepath,moniter='val_loss',verbose=1,save_best_only=True,
								save_weights_only=False,mode='auto')
								
	callbacks_list=[checkpoint]
	srcnn_model.fit(data,label,batch_size=128,validation_data=(val_data,val_label),
					callbacks=callbacks_list,epochs=200)
	