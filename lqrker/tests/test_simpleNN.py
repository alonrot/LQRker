import tensorflow as tf
import pdb


class MyDenseReLuLayer(tf.keras.layers.Layer):
  # Adding **kwargs to support base Keras layer arguments
  def __init__(self, in_dim, out_dim, name="DenseReLu_layer", **kwargs):
    super().__init__(**kwargs)

    self.w = self.add_weight(shape=(in_dim, out_dim), initializer="random_normal", trainable=True)
    self.b = self.add_weight(shape=(out_dim,), initializer="zeros", trainable=True)

  def call(self, inputs):
    y = tf.matmul(inputs, self.w) + self.b
    return tf.nn.relu(y)


# Simpler, with the add_weight() method:
class BayesianLinearRegressionLayer(tf.keras.layers.Layer):
    def __init__(self, in_dim, n_features, name="BLR_layer",**kwargs):
        super(BayesianLinearRegressionLayer, self).__init__(**kwargs)

        # Parameters:
        self.n_features = n_features
        
        # Adding weights:
        self.L0 = self.add_weight(shape=(n_features, n_features), initializer="random_normal", trainable=True)
        self.beta0 = self.add_weight(shape=(n_features, 1), initializer="random_normal", trainable=True)


        # units: number of features Nf
        self.w = self.add_weight(shape=(in_dim, out_dim), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(out_dim,), initializer="zeros", trainable=False)

    def call(self, inputs):

        # Lambda0:
        Lambda0 = tf.matmul(self.L0,tf.transpose(self.L0)) + 1e-6*tf.eye(self.n_features)



        return tf.matmul(inputs, self.w) + self.b


class BayesianLinearRegressionModel(tf.keras.Model):
  def __init__(self, num_features_out, name="BLRModel", **kwargs):
    super().__init__(**kwargs)

    self.dense = MyDenseReLuLayer(in_dim=4, out_dim=10)
    self.blr = BayesianLinearRegressionLayer(in_dim=10, out_dim=num_features_out)

  def call(self, x):
    x = self.dense(x)
    return self.blr(x)


if __name__ == "__main__":

	Ndesign_weights = 4
	Nels = 1

	model = BayesianLinearRegressionModel(num_features_out=8)

	# Call model on a test input
	x = tf.ones((1,Ndesign_weights))
	y = model(x)

	# # print(model.variables)
	# print(model.weights)
	# print(model.layers)
	print(model.summary())
	# print(model.variables)
	# print("\nTrainable\n=========")
	# print(model.trainable_variables)


	# pdb.set_trace()


	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		loss=tf.keras.losses.MeanSquaredError(),
		metrics=[tf.keras.metrics.Accuracy()],
	)



	print("Fit model on training data")
	history = model.fit(
		x_train,
		y_train,
		batch_size=64,
		epochs=2,
		# We pass some validation for
		# monitoring validation loss and metrics
		# at the end of each epoch
		validation_data=(x_val, y_val),
	)



