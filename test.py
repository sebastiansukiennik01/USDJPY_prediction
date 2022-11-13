from Preprocessing import Preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics

from JPYModel import JPYModelLinaer, JPYModelClasssifier
from Losses import MSLogScaledErrorLoss
from Callbacks import DetectOverfittingCallback, TensorBoardCallback, EarlyStoppingCallback


def main():
    train_x, train_y, val_x, val_y, test_x, test_y = Preprocessing().\
        loadDataFromCache(fileName='USDJPY_transformed_1',
                          labels=['target_1']).\
        divideTrainTest().\
        standarize().\
        createDataset(batchSize=128,
                      prefetch=True)

    model = JPYModelLinaer()
    model.compile(
        loss=MSLogScaledErrorLoss(alpha=1)
    )
    model.fit(train_x,
              train_y,
              epochs=40,
              validation_data=(val_x, val_y),
              callbacks=[TensorBoardCallback(),
                         tf.keras.callbacks.EarlyStopping(
                             patience=10,
                             min_delta=0.01,
                             mode='min',
                             monitor='val_loss',
                             restore_best_weights=True,
                             verbose=1
                         )])

    test_x = test_x.numpy()[:30]
    test_y = test_y.numpy()[:30]
    pred = model.predict(test_x)
    # print(model.evaluate(test_x, test_y))

    model.save_model()
    model.plot_results(test_y, pred)

    return model


def load_test():
    train_x, train_y, val_x, val_y, test_x, test_y = Preprocessing(). \
        loadDataFromCache(fileName='USDJPY_transformed_1',
                          labels=['target_1']). \
        divideTrainTest(). \
        standarize(). \
        createDataset(batchSize=128,
                      prefetch=True)

    model = JPYModelLinaer.load_model("JPYLinear_20221112_2328")
    pred = model.predict(train_x)
    JPYModelLinaer.plot_results(train_y, pred)


if __name__ == '__main__':
    main()