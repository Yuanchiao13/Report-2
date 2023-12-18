#南華大學跨領域-人工智慧期中報告

主題:不平衡分類：信用卡欺詐檢測

組員: 11118111李遠樵、11115022陳廷羽、11115004林彥佑

#作業流程

```python
import csv
import numpy as np

# Get the real data from https://www.kaggle.com/mlg-ulb/creditcardfraud/
fname = "/creditcard.csv"

all_features = []
all_targets = []
with open(fname) as f:
    for i, line in enumerate(f):
        if i == 0:
            print("HEADER:", line.strip())
            continue  # Skip header
        fields = line.strip().split(",")
        all_features.append([float(v.replace('"', "")) for v in fields[:-1]])
        all_targets.append([int(fields[-1].replace('"', ""))])
        if i == 1:
            print("EXAMPLE FEATURES:", all_features[-1])

features = np.array(all_features, dtype="float32")
targets = np.array(all_targets, dtype="uint8")
print("features.shape:", features.shape)
print("targets.shape:", targets.shape)
```
HEADER: "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"
EXAMPLE FEATURES: [0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]
features.shape: (284807, 30)
targets.shape: (284807, 1)

```python
num_val_samples = int(len(features) * 0.2)
train_features = features[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_features = features[-num_val_samples:]
val_targets = targets[-num_val_samples:]

print("Number of training samples:", len(train_features))
print("Number of validation samples:", len(val_features))
```
Number of training samples: 227846
Number of validation samples: 56961
```python
counts = np.bincount(train_targets[:, 0])
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(train_targets)
    )
)

weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]
```
Number of positive samples in training data: 417 (0.18% of total)
```python
mean = np.mean(train_features, axis=0)
train_features -= mean
val_features -= mean
std = np.std(train_features, axis=0)
train_features /= std
val_features /= std
```
```python
import keras

model = keras.Sequential(
    [
        keras.Input(shape=train_features.shape[1:]),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 256)               7936      
                                                                 
 dense_1 (Dense)             (None, 256)               65792     
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_2 (Dense)             (None, 256)               65792     
                                                                 
 dropout_1 (Dropout)         (None, 256)               0         
                                                                 
 dense_3 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 139777 (546.00 KB)
Trainable params: 139777 (546.00 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```python
metrics = [
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
)

callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.keras")]
class_weight = {0: weight_for_0, 1: weight_for_1}

model.fit(
    train_features,
    train_targets,
    batch_size=2048,
    epochs=30,
    verbose=2,
    callbacks=callbacks,
    validation_data=(val_features, val_targets),
    class_weight=class_weight,
)
```
Epoch 1/30
112/112 - 8s - loss: 2.2613e-06 - fn: 45.0000 - fp: 25989.0000 - tn: 201440.0000 - tp: 372.0000 - precision: 0.0141 - recall: 0.8921 - val_loss: 0.0795 - val_fn: 9.0000 - val_fp: 1153.0000 - val_tn: 55733.0000 - val_tp: 66.0000 - val_precision: 0.0541 - val_recall: 0.8800 - 8s/epoch - 75ms/step
Epoch 2/30
112/112 - 5s - loss: 1.3434e-06 - fn: 29.0000 - fp: 7782.0000 - tn: 219647.0000 - tp: 388.0000 - precision: 0.0475 - recall: 0.9305 - val_loss: 0.0897 - val_fn: 8.0000 - val_fp: 1473.0000 - val_tn: 55413.0000 - val_tp: 67.0000 - val_precision: 0.0435 - val_recall: 0.8933 - 5s/epoch - 43ms/step
Epoch 3/30
112/112 - 6s - loss: 1.1415e-06 - fn: 25.0000 - fp: 6994.0000 - tn: 220435.0000 - tp: 392.0000 - precision: 0.0531 - recall: 0.9400 - val_loss: 0.0423 - val_fn: 10.0000 - val_fp: 318.0000 - val_tn: 56568.0000 - val_tp: 65.0000 - val_precision: 0.1697 - val_recall: 0.8667 - 6s/epoch - 55ms/step
Epoch 4/30
112/112 - 5s - loss: 1.3049e-06 - fn: 32.0000 - fp: 8370.0000 - tn: 219059.0000 - tp: 385.0000 - precision: 0.0440 - recall: 0.9233 - val_loss: 0.1764 - val_fn: 4.0000 - val_fp: 3021.0000 - val_tn: 53865.0000 - val_tp: 71.0000 - val_precision: 0.0230 - val_recall: 0.9467 - 5s/epoch - 42ms/step
Epoch 5/30
112/112 - 6s - loss: 9.8869e-07 - fn: 25.0000 - fp: 7608.0000 - tn: 219821.0000 - tp: 392.0000 - precision: 0.0490 - recall: 0.9400 - val_loss: 0.0227 - val_fn: 10.0000 - val_fp: 479.0000 - val_tn: 56407.0000 - val_tp: 65.0000 - val_precision: 0.1195 - val_recall: 0.8667 - 6s/epoch - 56ms/step
Epoch 6/30
112/112 - 5s - loss: 7.7553e-07 - fn: 17.0000 - fp: 5773.0000 - tn: 221656.0000 - tp: 400.0000 - precision: 0.0648 - recall: 0.9592 - val_loss: 0.1337 - val_fn: 3.0000 - val_fp: 3404.0000 - val_tn: 53482.0000 - val_tp: 72.0000 - val_precision: 0.0207 - val_recall: 0.9600 - 5s/epoch - 42ms/step
Epoch 7/30
112/112 - 5s - loss: 5.5290e-07 - fn: 13.0000 - fp: 4688.0000 - tn: 222741.0000 - tp: 404.0000 - precision: 0.0793 - recall: 0.9688 - val_loss: 0.0213 - val_fn: 9.0000 - val_fp: 393.0000 - val_tn: 56493.0000 - val_tp: 66.0000 - val_precision: 0.1438 - val_recall: 0.8800 - 5s/epoch - 42ms/step
Epoch 8/30
112/112 - 6s - loss: 8.0407e-07 - fn: 17.0000 - fp: 9239.0000 - tn: 218190.0000 - tp: 400.0000 - precision: 0.0415 - recall: 0.9592 - val_loss: 0.0346 - val_fn: 10.0000 - val_fp: 275.0000 - val_tn: 56611.0000 - val_tp: 65.0000 - val_precision: 0.1912 - val_recall: 0.8667 - 6s/epoch - 54ms/step
Epoch 9/30
112/112 - 5s - loss: 8.3283e-07 - fn: 17.0000 - fp: 7462.0000 - tn: 219967.0000 - tp: 400.0000 - precision: 0.0509 - recall: 0.9592 - val_loss: 0.0413 - val_fn: 8.0000 - val_fp: 734.0000 - val_tn: 56152.0000 - val_tp: 67.0000 - val_precision: 0.0836 - val_recall: 0.8933 - 5s/epoch - 42ms/step
Epoch 10/30
112/112 - 6s - loss: 6.3612e-07 - fn: 13.0000 - fp: 6121.0000 - tn: 221308.0000 - tp: 404.0000 - precision: 0.0619 - recall: 0.9688 - val_loss: 0.1115 - val_fn: 6.0000 - val_fp: 2679.0000 - val_tn: 54207.0000 - val_tp: 69.0000 - val_precision: 0.0251 - val_recall: 0.9200 - 6s/epoch - 53ms/step
Epoch 11/30
112/112 - 5s - loss: 4.4499e-07 - fn: 5.0000 - fp: 4986.0000 - tn: 222443.0000 - tp: 412.0000 - precision: 0.0763 - recall: 0.9880 - val_loss: 0.0524 - val_fn: 8.0000 - val_fp: 714.0000 - val_tn: 56172.0000 - val_tp: 67.0000 - val_precision: 0.0858 - val_recall: 0.8933 - 5s/epoch - 43ms/step
Epoch 12/30
112/112 - 5s - loss: 4.9975e-07 - fn: 10.0000 - fp: 5665.0000 - tn: 221764.0000 - tp: 407.0000 - precision: 0.0670 - recall: 0.9760 - val_loss: 0.0231 - val_fn: 8.0000 - val_fp: 411.0000 - val_tn: 56475.0000 - val_tp: 67.0000 - val_precision: 0.1402 - val_recall: 0.8933 - 5s/epoch - 45ms/step
Epoch 13/30
112/112 - 7s - loss: 6.9749e-07 - fn: 11.0000 - fp: 8648.0000 - tn: 218781.0000 - tp: 406.0000 - precision: 0.0448 - recall: 0.9736 - val_loss: 0.0433 - val_fn: 11.0000 - val_fp: 272.0000 - val_tn: 56614.0000 - val_tp: 64.0000 - val_precision: 0.1905 - val_recall: 0.8533 - 7s/epoch - 66ms/step
Epoch 14/30
112/112 - 5s - loss: 5.8866e-07 - fn: 10.0000 - fp: 6486.0000 - tn: 220943.0000 - tp: 407.0000 - precision: 0.0590 - recall: 0.9760 - val_loss: 0.0478 - val_fn: 9.0000 - val_fp: 1133.0000 - val_tn: 55753.0000 - val_tp: 66.0000 - val_precision: 0.0550 - val_recall: 0.8800 - 5s/epoch - 48ms/step
Epoch 15/30
112/112 - 7s - loss: 4.9786e-07 - fn: 9.0000 - fp: 5924.0000 - tn: 221505.0000 - tp: 408.0000 - precision: 0.0644 - recall: 0.9784 - val_loss: 0.0151 - val_fn: 9.0000 - val_fp: 324.0000 - val_tn: 56562.0000 - val_tp: 66.0000 - val_precision: 0.1692 - val_recall: 0.8800 - 7s/epoch - 61ms/step
Epoch 16/30
112/112 - 5s - loss: 3.3685e-07 - fn: 6.0000 - fp: 4256.0000 - tn: 223173.0000 - tp: 411.0000 - precision: 0.0881 - recall: 0.9856 - val_loss: 0.0251 - val_fn: 10.0000 - val_fp: 584.0000 - val_tn: 56302.0000 - val_tp: 65.0000 - val_precision: 0.1002 - val_recall: 0.8667 - 5s/epoch - 45ms/step
Epoch 17/30
112/112 - 6s - loss: 3.8044e-07 - fn: 4.0000 - fp: 5624.0000 - tn: 221805.0000 - tp: 413.0000 - precision: 0.0684 - recall: 0.9904 - val_loss: 0.0147 - val_fn: 11.0000 - val_fp: 352.0000 - val_tn: 56534.0000 - val_tp: 64.0000 - val_precision: 0.1538 - val_recall: 0.8533 - 6s/epoch - 55ms/step
Epoch 18/30
112/112 - 5s - loss: 3.4783e-07 - fn: 3.0000 - fp: 3898.0000 - tn: 223531.0000 - tp: 414.0000 - precision: 0.0960 - recall: 0.9928 - val_loss: 0.0159 - val_fn: 9.0000 - val_fp: 372.0000 - val_tn: 56514.0000 - val_tp: 66.0000 - val_precision: 0.1507 - val_recall: 0.8800 - 5s/epoch - 42ms/step
Epoch 19/30
112/112 - 5s - loss: 4.2034e-07 - fn: 3.0000 - fp: 5424.0000 - tn: 222005.0000 - tp: 414.0000 - precision: 0.0709 - recall: 0.9928 - val_loss: 0.1473 - val_fn: 6.0000 - val_fp: 2462.0000 - val_tn: 54424.0000 - val_tp: 69.0000 - val_precision: 0.0273 - val_recall: 0.9200 - 5s/epoch - 42ms/step
Epoch 20/30
112/112 - 6s - loss: 5.0629e-07 - fn: 9.0000 - fp: 5965.0000 - tn: 221464.0000 - tp: 408.0000 - precision: 0.0640 - recall: 0.9784 - val_loss: 0.0385 - val_fn: 9.0000 - val_fp: 909.0000 - val_tn: 55977.0000 - val_tp: 66.0000 - val_precision: 0.0677 - val_recall: 0.8800 - 6s/epoch - 54ms/step
Epoch 21/30
112/112 - 5s - loss: 7.7273e-07 - fn: 7.0000 - fp: 6984.0000 - tn: 220445.0000 - tp: 410.0000 - precision: 0.0555 - recall: 0.9832 - val_loss: 0.0449 - val_fn: 10.0000 - val_fp: 445.0000 - val_tn: 56441.0000 - val_tp: 65.0000 - val_precision: 0.1275 - val_recall: 0.8667 - 5s/epoch - 42ms/step
Epoch 22/30
112/112 - 6s - loss: 5.8235e-07 - fn: 10.0000 - fp: 4855.0000 - tn: 222574.0000 - tp: 407.0000 - precision: 0.0773 - recall: 0.9760 - val_loss: 0.0329 - val_fn: 8.0000 - val_fp: 681.0000 - val_tn: 56205.0000 - val_tp: 67.0000 - val_precision: 0.0896 - val_recall: 0.8933 - 6s/epoch - 53ms/step
Epoch 23/30
112/112 - 5s - loss: 3.4212e-07 - fn: 3.0000 - fp: 3965.0000 - tn: 223464.0000 - tp: 414.0000 - precision: 0.0945 - recall: 0.9928 - val_loss: 0.0247 - val_fn: 8.0000 - val_fp: 696.0000 - val_tn: 56190.0000 - val_tp: 67.0000 - val_precision: 0.0878 - val_recall: 0.8933 - 5s/epoch - 43ms/step
Epoch 24/30
112/112 - 5s - loss: 3.0463e-07 - fn: 1.0000 - fp: 3783.0000 - tn: 223646.0000 - tp: 416.0000 - precision: 0.0991 - recall: 0.9976 - val_loss: 0.0121 - val_fn: 10.0000 - val_fp: 296.0000 - val_tn: 56590.0000 - val_tp: 65.0000 - val_precision: 0.1801 - val_recall: 0.8667 - 5s/epoch - 42ms/step
Epoch 25/30
112/112 - 6s - loss: 3.2233e-07 - fn: 1.0000 - fp: 3917.0000 - tn: 223512.0000 - tp: 416.0000 - precision: 0.0960 - recall: 0.9976 - val_loss: 0.0242 - val_fn: 9.0000 - val_fp: 666.0000 - val_tn: 56220.0000 - val_tp: 66.0000 - val_precision: 0.0902 - val_recall: 0.8800 - 6s/epoch - 54ms/step
Epoch 26/30
112/112 - 5s - loss: 2.7953e-07 - fn: 2.0000 - fp: 3291.0000 - tn: 224138.0000 - tp: 415.0000 - precision: 0.1120 - recall: 0.9952 - val_loss: 0.0634 - val_fn: 7.0000 - val_fp: 2197.0000 - val_tn: 54689.0000 - val_tp: 68.0000 - val_precision: 0.0300 - val_recall: 0.9067 - 5s/epoch - 42ms/step
Epoch 27/30
112/112 - 5s - loss: 1.7501e-07 - fn: 2.0000 - fp: 2039.0000 - tn: 225390.0000 - tp: 415.0000 - precision: 0.1691 - recall: 0.9952 - val_loss: 0.0132 - val_fn: 12.0000 - val_fp: 240.0000 - val_tn: 56646.0000 - val_tp: 63.0000 - val_precision: 0.2079 - val_recall: 0.8400 - 5s/epoch - 49ms/step
Epoch 28/30
112/112 - 5s - loss: 4.4529e-07 - fn: 4.0000 - fp: 4957.0000 - tn: 222472.0000 - tp: 413.0000 - precision: 0.0769 - recall: 0.9904 - val_loss: 0.0218 - val_fn: 9.0000 - val_fp: 408.0000 - val_tn: 56478.0000 - val_tp: 66.0000 - val_precision: 0.1392 - val_recall: 0.8800 - 5s/epoch - 46ms/step
Epoch 29/30
112/112 - 5s - loss: 3.1511e-07 - fn: 5.0000 - fp: 3697.0000 - tn: 223732.0000 - tp: 412.0000 - precision: 0.1003 - recall: 0.9880 - val_loss: 0.0318 - val_fn: 9.0000 - val_fp: 760.0000 - val_tn: 56126.0000 - val_tp: 66.0000 - val_precision: 0.0799 - val_recall: 0.8800 - 5s/epoch - 41ms/step
Epoch 30/30
112/112 - 6s - loss: 2.2887e-07 - fn: 3.0000 - fp: 2951.0000 - tn: 224478.0000 - tp: 414.0000 - precision: 0.1230 - recall: 0.9928 - val_loss: 0.0336 - val_fn: 8.0000 - val_fp: 657.0000 - val_tn: 56229.0000 - val_tp: 67.0000 - val_precision: 0.0925 - val_recall: 0.8933 - 6s/epoch - 54ms/step
<keras.src.callbacks.History at 0x78cb169e3370>
