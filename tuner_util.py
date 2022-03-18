from keras import callbacks
import keras
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=32, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    return model


def tune_model(X, y):
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = tuner.search(
        x=X,
        y=y,
        epochs=2,
        validation_split=0.2,
        callbacks=[stop_early]
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    return tuner, history

def checkTuner():
    # deve essere lo stesso
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')
    return tuner

def checkTuner_RS():
    # deve essere lo stesso
    tuner = kt.RandomSearch(model_builder,
                            objective='val_accuracy',
                            max_epochs=10,
                            factor=3,
                            directory='my_dir',
                            project_name='intro_to_kt')
    return tuner

if __name__ == '__main__':
    (img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values between 0 and 1
    img_train = img_train.astype('float32') / 255.0
    img_test = img_test.astype('float32') / 255.

    tune_load = 'tune'

    if tune_load == 'tune':
        tuner, history_from_f = tune_model(img_train, label_train)
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    elif tune_load == 'load':
        tuner = checkTuner()
    elif tune_load == 'both':
        tuner_search, history_from_f = tune_model(img_train, label_train)
        tuner_load = checkTuner()

        tuner_search.results_summary()
        tuner_load.results_summary()
        print('reload tuner')
        best_model_search = tuner_search.get_best_models(num_models=1)[0]
        best_model_load = tuner_load.get_best_models(num_models=1)[0]
        a = best_model_search.predict(img_test)
        b = best_model_load.predict(img_test)
        print(a == b)

    if tune_load == 'tune' or tune_load == 'load':
        best_model = tuner.get_best_models(num_models=1)[0]
        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

