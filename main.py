import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras import metrics
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os

print("Input Data:")
print(os.listdir("input"))

train = pd.read_csv('input/train.csv')
test  = pd.read_csv('input/test.csv')
print(train.head(5))
print("===============================")

print(len(train), 'train data rows')
print(len(test), 'test data rows')
print("===============================")

#Check for Null values
print("Null values in train data: ", train.isnull().any().any())
print("Null values in test data: ", test.isnull().any().any())
print("===============================")

#classes
classes = np.unique(train.iloc[:, 562].values)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

#normalize data
train_x = train.iloc[:, :561].values
test_x = test.iloc[:, :561].values

train_norm=(train_x-train_x.mean())/train_x.std()
test_norm=(test_x-test_x.mean())/test_x.std()

train_normalized = pd.DataFrame(train_norm)
test_normalized = pd.DataFrame(test_norm)

train_norm = train_normalized.join(train.iloc[:,561:563])
test_norm = test_normalized.join(test.iloc[:,561:563])

###########################################
#Feature matrix
train_features = train_norm.iloc[:,:561].values
test_features = test_norm.iloc[:,:561].values

train_results = train_norm.iloc[:,562:].values
test_results = test_norm.iloc[:,562:].values
train_resultss=np.zeros((len(train_results),6))
test_resultss=np.zeros((len(test_results),6))
test_labels = np.zeros((len(test_results),1))
###############################################
#plot counts
train_norm.Activity.value_counts().plot(kind='bar')
plt.title('Frequency of Activities')
plt.show()
subjects = train_norm.subject.unique()
plt.hist([train_norm.loc[train_norm.subject == x, 'Activity'] for x in subjects], label=subjects)
plt.legend(train_norm.subject.unique())
plt.title('Activity per Subject')
plt.show()

for k in range (0,len(train_results)):
    if train_results[k] =='STANDING':
        train_resultss[k][0]=1
    elif train_results[k] =='WALKING':
        train_resultss[k][1]=1
    elif train_results[k] =='WALKING_UPSTAIRS':
        train_resultss[k][2]=1
    elif train_results[k] =='WALKING_DOWNSTAIRS':
        train_resultss[k][3]=1
    elif train_results[k] =='SITTING':
        train_resultss[k][4]=1
    else:
        train_resultss[k][5]=1

for k in range (0,len(test_results)):
    if test_results[k] =='STANDING':
        test_resultss[k][0]=1
        test_labels[k] = 0
    elif test_results[k] =='WALKING':
        test_resultss[k][1]=1
        test_labels[k] = 1
    elif test_results[k] =='WALKING_UPSTAIRS':
        test_resultss[k][2]=1
        test_labels[k] = 2
    elif test_results[k] =='WALKING_DOWNSTAIRS':
        test_resultss[k][3]=1
        test_labels[k] = 3
    elif test_results[k] =='SITTING':
        test_resultss[k][4]=1
        test_labels[k] = 4
    else:
        test_resultss[k][5]=1
        test_labels[k] = 5

# define an empty sequential structure
model = Sequential()
# add a dense layer (MLP)
model.add(Dense(64, activation='relu', input_dim=561))
# use a dropout layer with  50% of inputs dropped
model.add(Dropout(0.5))
#add 2nd dense layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# output layer, use a softmax activation
model.add(Dense(6, activation='softmax'))

# print model layers' info
print("Model layer's info:")
print(model.summary())

# we compile the model
# using categorical crossentropy as a loss function
# and adam optimizer
#learning rate and decay values are the ones recommended by TensorFlow documentation
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, decay=1e-6)
model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=[
        metrics.mae,
        metrics.categorical_accuracy
    ],
)

# We train (fit our data to) our model
history = model.fit(
    train_features,                      # features
    train_resultss,                      # labels
    epochs=30,                           # numbers of epoch
    batch_size=256,                      # define batch size
    verbose=1,                           # the most extended verbose
    validation_split=0.2                 # 80% for train and 20% for validation
)

print(history.history)

print("***********************************************************************")

score = model.evaluate(
    test_features,                  # features
    test_resultss,                  # labels
    batch_size=256,                 # batch size
    verbose=1                       # the most extended verbose
)
print('\nTest categorical_crossentropy:', score[0])
print('\nTest mean_absolute_error:', score[1])
print('\nTest accuracy:', score[2])

# confusion matrix and accuracy
pre = model.predict_classes(test_features, batch_size=256, verbose=1)
cm=confusion_matrix(test_labels,pre)

data_classes = ["STANDING", "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "LAYING"]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap="YlGn")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_loss_acc():
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    train_accurancy = history.history['categorical_accuracy']
    test_accurancy = history.history['val_categorical_accuracy']
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.plot(epoch_count, train_accurancy, 'g--')
    plt.plot(epoch_count, test_accurancy, 'y-')
    plt.legend(['Training Loss', 'Test Loss', 'Train Accuracy', 'Test Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

plot_confusion_matrix(cm,data_classes, title="Confusion Matrix")
plot_loss_acc()
