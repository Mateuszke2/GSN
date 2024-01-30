print("Poszł")
import os
import datetime
os.add_dll_directory("D:/CUDA/bin")
os.add_dll_directory("D:/CUDnn/cudnn-windows-x86_64-8.6.0.163_cuda11-archive/bin")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('pogoda.csv')

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

# print(train_mean)
# print(train_std)


train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

print(train_df.head())



################### PARAMETRYYYYYYYYY ###################

ACTIVATION = 'tanh'
BATCH_SIZE = 128
OUT_STEPS = 24
IN_STEPS = 168
MAX_EPOCHS = 10
LEARNING_RATE_ADAM = 0.001
UNITS = 32
UNITS_2 = 32
SEQ_STRIDE = 12

checkpoint_path = "pogoda/model_A{}_E{}_S{}-{}_BS{}_SS{}_U{}/model".format(ACTIVATION, MAX_EPOCHS, IN_STEPS, OUT_STEPS, BATCH_SIZE, SEQ_STRIDE, UNITS)
plot_pred_path = "pogoda/model_A{}_E{}_S{}-{}_BS{}_SS{}_U{}/predict".format(ACTIVATION, MAX_EPOCHS, IN_STEPS, OUT_STEPS, BATCH_SIZE, SEQ_STRIDE, UNITS)
plot_loss_path = "pogoda/model_A{}_E{}_S{}-{}_BS{}_SS{}_U{}/history".format(ACTIVATION, MAX_EPOCHS, IN_STEPS, OUT_STEPS, BATCH_SIZE, SEQ_STRIDE, UNITS)
history_path = "pogoda/model_A{}_E{}_S{}-{}_BS{}_SS{}_U{}/train_history.pkl".format(ACTIVATION, MAX_EPOCHS, IN_STEPS, OUT_STEPS, BATCH_SIZE, SEQ_STRIDE, UNITS)

numer = 1430
labels  = test_df['T (degC)'].values[numer:numer+OUT_STEPS]
inputs = test_df.values[numer-IN_STEPS:numer]
# print(inputs)
# print(labels)
inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
# print(inputs_tensor)
# print(labels_tensor)
inputs_tensor = tf.expand_dims(inputs_tensor, axis=0)
labels_tensor = labels_tensor[tf.newaxis, :, tf.newaxis]
# print(inputs_tensor)
# print(labels_tensor)

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  # Wykonywanie predykcji i wykreślanie wyników

  def plot(self, model=None, plot_col='Acceleration1', max_subplots=3, normed = True):
    if normed:
      norm_str = '[normed]'
    else:
      norm_str = ''
      
    inputs, labels = inputs_tensor, labels_tensor

    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    
    if inputs.ndim < 3:
      inputs = tf.expand_dims(inputs, axis=0)
      labels = tf.expand_dims(labels, axis=0)
      max_n = 1
    else:
      max_n = min(max_subplots, len(inputs))


    if model is not None:
      predictions = model(inputs)
      # print(predictions)
      
    for n in range(max_n):
      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      if normed:
        inputs_2_plot =  inputs[n, :, plot_col_index]
        labels_2_plot = labels[n, :, label_col_index]
        if model is not None:
          predictions_2_plot = predictions[n, :, label_col_index]
      else:
        inputs_2_plot =  inputs[n, :, plot_col_index] * train_std[plot_col_index] + train_mean[plot_col_index]
        labels_2_plot = labels[n, :, label_col_index] * train_std[plot_col_index] + train_mean[plot_col_index]
        if model is not None:
          predictions_2_plot = predictions[n, :, label_col_index] * train_std[plot_col_index] + train_mean[plot_col_index]

      plt.subplot(max_n, 1, n+1,)
      plt.title('A:{}_E:{}_S:{}-{}_BS:{}_SS:{}_U:{}'.format(ACTIVATION, MAX_EPOCHS, IN_STEPS, OUT_STEPS, BATCH_SIZE, SEQ_STRIDE, UNITS))
      plt.ylabel(f'{plot_col} {norm_str}')
      # print(inputs[n, :, plot_col_index])
      # print(tf.constant(train_std,dtype=tf.float32))
      # print(tf.constant(train_mean,dtype=tf.float32))
      plt.plot(self.input_indices, inputs_2_plot,
              label='Inputs', marker='.', zorder=-10,)
      
      

      plt.scatter(self.label_indices, labels_2_plot,
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        # odleglosc = np.linalg.norm(labels[n, :, label_col_index]-predictions[n, :, label_col_index])
        # print("odl: ", odleglosc)
        # print("odl. śr: ", odleglosc/OUT_STEPS)
        # predictions = model(inputs)

        
        # print(inputs_2_plot)
        # print(labels_2_plot)
        # print(predictions_2_plot)
        
        # METRYKI JAKOSCI
        mae = mean_absolute_error(labels_2_plot, predictions_2_plot)
        mse = mean_squared_error(labels_2_plot, predictions_2_plot)
        r2 = r2_score(labels_2_plot, predictions_2_plot)

        plt.scatter(self.label_indices, predictions_2_plot,
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)
        plt.text(0.05, 0.95, "MAE: {:.3f}, MSE: {:.3f}, R2:{:.3f}".format(mae,mse,r2), transform=plt.gca().transAxes,
                  verticalalignment='top', horizontalalignment='left')
      if n == 0:
        plt.legend(loc = 'lower left')
      plt.grid()
    plt.xlabel('')
    if model is not None:
        plt.savefig(plot_pred_path)
        plt.show()

  # Podział na sekwencje danych
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    # print(data)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride = SEQ_STRIDE,
        shuffle=False,
        batch_size=BATCH_SIZE)

    ds = ds.map(self.split_window)

    return ds

  # Własności pozwalające na odnoszenie się do konkretnych sekwencji zawierających się w batchach

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      result = next(iter(self.train))
      self._example = result
    return result

  @property
  def last_train(self):
    for batch in self.train:
        last_batch = batch
    inputs, labels = last_batch
    inputs_last, labels_last = inputs[-1], labels[-1]
    return inputs_last, labels_last

  @property
  def example_test(self):
    result = next(iter(self.test))
    return result
  
  @property
  def example_val(self):
    result = next(iter(self.val))
    return result

  # @property
  # def example_test_no(self):
  #   for _ in range(no):
  #     result = next(iter(self.test))
  #   return result

# Definicja Okienka

multi_window = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS, label_columns = ['T (degC)'])

# Definicja modelu sieci neuronowej

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    tf.keras.layers.LSTM(units = UNITS, return_sequences=False, activation = ACTIVATION,
                         kernel_regularizer = "l2"),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
    
])


# Funkcja kompilująca i trenująca zdefiniowany model

def compile_and_fit(model, window, patience=MAX_EPOCHS):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_ADAM),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
  # print(window.train)
  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=early_stopping)
  print(model.summary())
  return history




# Trening (do zakomentowania gdy wczytujemy model)
history = compile_and_fit(multi_lstm_model, multi_window)

# WCZYTANIE MODELU
# multi_lstm_model = tf.keras.models.load_model(r'D:\Users\User\Desktop\Studia\MGR\SEM2\GSN\pogoda\model_sin()_Atanh_E100_S100-24_BS128_SS5_U128\model')
# plot_pred_path = r'D:\Users\User\Desktop\Studia\MGR\SEM2\GSN\pogoda\model_sin()_Atanh_E100_S100-24_BS128_SS5_U128\pred2'

#Zapisanie modelu
multi_lstm_model.save(checkpoint_path)

# Zapisanie pliku historii
with open(history_path, 'wb') as file:
  pickle.dump(history, file)

# Ewentualne wczytanie historii
# with open(history_path, 'rb') as file:
#     history = pickle.load(file)

# Predykuj i wyświetl
multi_window.plot(model = multi_lstm_model, plot_col='T (degC)', max_subplots=1)

# Wykres historii uczenia
plt.figure(120)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid()
plt.savefig(plot_loss_path)
plt.show()

# Koniec programu
print('Done')