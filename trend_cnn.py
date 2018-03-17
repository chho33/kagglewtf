import json
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet201,DenseNet121
from keras.layers import Input,Conv2D,Reshape
from trend_etl_dl import *
from random import shuffle
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, History
from keras.optimizers import Adam
from keras.models import Model

train_tags = get_train_answers()
group_n = 6
split_n = 36
#split_n=6*24
split_n=2*6*24
batch_size = 2
epochs = 8
norm = False 
class_weight = {0:1,1:8}
class_weight = None 
lr = 0.001
min_lr = 0.00001
optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def cnn_train_generator(ids,split_n=split_n,group_n=group_n,norm=False):
    while True:
        for ix in ids:
            x = get_matrix(ix,'train',split_n=split_n,group_n=group_n,norm=norm)[np.newaxis,:]
            y = np.array(train_tags[ix]).reshape(-1,1)
            yield (x,y)

def cnn_train_generator(ids,split_n=split_n,group_n=group_n,norm=norm):
    n_ids = len(ids)
    n_ids = int(np.floor(n_ids/batch_size))
    while True:
        for i in range(0,n_ids)[::batch_size]:
            ixs = ids[i*batch_size:(i+1)*batch_size]
            x = [get_matrix(ix,'train',split_n=split_n,group_n=group_n,norm=norm)[np.newaxis,:] for ix in ixs]
            x = np.concatenate(x)
            y = [np.array(train_tags[ix]).reshape(-1,1) for ix in ixs]
            y = np.concatenate(y)
            print('x,y shape: ',x.shape,y.shape)
            yield (x,y)

y = []
ids = []
for k,v in train_tags.items():
    y.append(v)
    ids.append(k)

history = History()
reduce_lr = ReduceLROnPlateau(factor=0.2,
                              patience=10, min_lr=min_lr, mode='auto')
model_checkpoint = ModelCheckpoint('export/models/trend_cnn.%s.{epoch:02d}.hdf5'%int(time.time()), 
     verbose=0, mode='auto', period=1,save_best_only=False)
#model_checkpoint = ModelCheckpoint(filepath='best_weights_%s.hdf5'%version, verbose=0, save_best_only=True)
callbacks = [reduce_lr]
callbacks = [model_checkpoint,reduce_lr,history]

ids_train, ids_val, y_train, y_val = train_test_split(ids,y,train_size=0.9,shuffle=True,random_state=11)
print('ytran: ',len(y_train))
input_shape = (split_n, int(518400/(split_n*group_n)), 3)
input_tensor = Input(shape=input_shape)  # this assumes K.image_data_format() == 'channels_last'
#model = InceptionV3(input_tensor=input_tensor, weights=None, include_top=True, classes=2)
#model = InceptionResNetV2(input_tensor=input_tensor, weights=None, include_top=True, classes=2)
#model = ResNet50(input_tensor=input_tensor, weights=None, include_top=True, classes=2)
#model = DenseNet201(include_top=True, weights=None, input_tensor=input_tensor,  classes=1)
#model = DenseNet121(include_top=True, weights=None, input_tensor=input_tensor,  classes=2)
#model = DenseNet121(include_top=False, weights=None, input_shape=input_shape,  classes=2)
#model = DenseNet201(include_top=False, weights=None, input_shape=input_shape,  classes=1)
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy',tf_auc_roc])
#model = DenseNet121(include_top=True, weights=None, input_tensor=input_tensor, classes=2)


'''
model = DenseNet121(include_top=False, pooling='avg', weights=None, input_shape=input_shape, input_tensor=input_tensor)
out = model.get_layer('avg_pool').output
print(out.shape)
#out = Dense(1, activation='sigmoid',name='fc1000')(out)
model = Model(model.input, out)


input1 = Input(shape=input_shape,name='pre')
pre1 = Conv2D(filters=3, kernel_size=(1, 1), padding='SAME', 
input_shape=input_shape, name='first_dense')(input1)

model_base = DenseNet121(include_top=False, pooling='avg', weights=None, input_shape=input_shape, input_tensor=input_tensor)
x = model_base.get_layer('avg_pool').output
x = Dense(3, activation='softmax')(x)
model_base = Model(model_base.input, x)
base_out = model_base(pre1)     
model = Model(input1, base_out)
'''

print('input shape: ',input_shape)
#model = DenseNet121(include_top=False, pooling='avg', weights=None, input_shape=input_shape, input_tensor=input_tensor)
model = DenseNet121(include_top=False, pooling='avg', weights=None, input_shape=input_shape)
out = Reshape((1024,))(model.output)
out = Dense(1, activation='sigmoid',name='fc1000')(out)
print('out shape: ',out.shape)
model = Model(model.input,out)
print(model.summary())

model.compile(loss=binary_crossentropy_with_ranking, optimizer=optimizer, metrics=['accuracy',tf_auc_roc])
a = time.time()
model.fit_generator(cnn_train_generator(ids=ids_train),
                    validation_data=cnn_train_generator(ids=ids_val),
                    validation_steps=int(np.floor(len(y_val)/batch_size)),
                    steps_per_epoch=int(np.floor(len(y_train)/batch_size)),
                    callbacks=callbacks,
                    class_weight=class_weight,
                    epochs=epochs)
print ('cost time: ', time.time() - a)

records = model.history.history
with open('export/trend_cnn_history_%s.json'%(int(time.time())),'w') as f:
    json.dump(records,f,cls=MyEncoder)

