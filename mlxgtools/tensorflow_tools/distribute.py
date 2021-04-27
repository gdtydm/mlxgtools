import tensorflow as tf 
  
def get_distribute_strategy():
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return strategy


def get_tpu_train_loop_template():
    return r"""
    def make_tpu_train_loop(strategy, model, optimizer = None, learning_rate = None):
        with strategy.scope():
        if optimizer is None:
            if learning_rate is None:
                learning_rate = 1e-3
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, 
                                                beta_1=0.9, beta_2=0.98,
                                                epsilon=1e-9)
        training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        training_accuracy = tf.keras.metrics.Accuracy('training_accuracy', dtype=tf.float32)
        valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
        valid_accuracy = tf.keras.metrics.Accuracy('valid_accuracy', dtype=tf.float32)
        valid_auc = tf.keras.metrics.AUC(num_thresholds = 8192, 
                                                name = 'valid_auc', 
                                                dtype=tf.float32)

        loss_function = make_loss_function()

        @tf.function   
        def step_fn(inputs):
            'The computation to run on each TPU device'
            answered_correctly = inputs['answered_correctly']
            non_padding_mask = inputs['non_padding_mask']
            label_mask = tf.cast(tf.greater_equal(answered_correctly, 0), tf.float32)
            label_mask *= non_padding_mask
            y_true = tf.clip_by_value(answered_correctly, clip_value_min=0, clip_value_max=1)
            y_true = y_true[..., tf.newaxis]
            with tf.GradientTape() as tape:
                logits = model(inputs, training=True)
                loss = loss_function(y_true, logits)
                # 每个 sample loss list
                loss = tf.reduce_mean(loss*label_mask, axis = -1) 
                # 每个 sample loss / num_replicas_in_sync
                loss = tf.nn.compute_average_loss(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(list(zip(grads, model.trainable_variables))) 
            # 加权 （根据label数）
            training_loss.update_state(loss*strategy.num_replicas_in_sync,
                                    sample_weight = tf.reduce_sum(label_mask))
            y_true = tf.reshape(y_true, (-1,))
            y_pred = tf.reshape(tf.cast(logits > 0, y_true.dtype), (-1,))
            label_mask = tf.reshape(label_mask, (-1,))
            training_accuracy.update_state(y_true, 
                                        y_pred,
                                        sample_weight=label_mask)

        @tf.function
        def train_multiple_steps(iterator, steps):
            has_data = True
            for _ in tf.range(steps):
                optional_data = iterator.get_next_as_optional()
                if not optional_data.has_value():
                    has_data = False
                    break
                strategy.run(step_fn, args=(optional_data.get_value(),))
            return has_data
        
        
        @tf.function
        def valid_step_fn(inputs):
            answered_correctly = inputs['answered_correctly']
            in_valid_set = inputs['in_valid_set']
            non_padding_mask = inputs['non_padding_mask']
            label_mask = tf.cast(tf.greater_equal(answered_correctly, 0), tf.float32)
            label_mask *= tf.cast(in_valid_set, tf.float32)
            label_mask *= non_padding_mask
            y_true = tf.clip_by_value(answered_correctly, clip_value_min=0, clip_value_max=1)
            y_true = y_true[..., tf.newaxis]
            inputs = {k:v for k,v in inputs.items() if k != 'in_valid_set'}
            logits = model(inputs, training=False)
            loss = loss_function(y_true, logits)
            loss = tf.reduce_mean(loss*label_mask, axis = -1)
            loss = tf.nn.compute_average_loss(loss)
            valid_loss.update_state(loss*strategy.num_replicas_in_sync,
                                    sample_weight = tf.reduce_sum(label_mask))
            y_true = tf.reshape(y_true, (-1,))
            y_pred = tf.reshape(tf.cast(logits > 0, y_true.dtype), (-1,))
            label_mask = tf.reshape(label_mask, (-1,))        
            valid_accuracy.update_state(y_true, 
                                y_pred, 
                                sample_weight = label_mask)
            sigmoids = tf.reshape(tf.sigmoid(logits), (-1,))

            valid_auc.update_state(y_true, 
                                sigmoids, 
                                sample_weight = label_mask)
    
        @tf.function
        def predict_and_valid(iterator):
            while tf.constant(True):
                optional_data = iterator.get_next_as_optional()
                if not optional_data.has_value():
                    break
                valid_data = optional_data.get_value()
                strategy.run(valid_step_fn, args=(valid_data,))

    
        def train_loop(train_ds, 
                    valid_ds = None, 
                    batch_size = 64,
                    steps_per_call = 128,
                    steps_per_epoch = 5500,
                    epochs = 4):

            batch_valid_ds = None
            if valid_ds:
                batch_valid_ds = valid_ds.batch(
                        batch_size,
                        drop_remainder = True
                        ).prefetch(tf.data.experimental.AUTOTUNE).cache()
            if train_ds is not None:
                train_iterator = iter(strategy.experimental_distribute_dataset(
                        train_ds.batch(
                            batch_size,
                            drop_remainder = True
                            ).prefetch(tf.data.experimental.AUTOTUNE)))             
            for epoch in range(epochs):
                if train_ds is not None:
                    steps_in_epoch = 0
                    training_loss.reset_states()
                    training_accuracy.reset_states()

                    while (steps_in_epoch < steps_per_epoch):
                        steps_in_epoch += steps_per_call
                        train_multiple_steps(train_iterator,
                                            tf.convert_to_tensor(steps_per_call))
                        print('Current step: {}, training loss: {:.4f}, accuracy: {:.2f}%'.format(
                            optimizer.iterations.numpy(),
                            float(training_loss.result()),
                            float(training_accuracy.result()) * 100))
                if batch_valid_ds is not None:
                    valid_loss.reset_states()
                    valid_accuracy.reset_states()
                    valid_auc.reset_states()
                    valid_iterator = iter(strategy.experimental_distribute_dataset(
                        batch_valid_ds
                    ))
                    predict_and_valid(valid_iterator)
                    print('Current epoch: {}, valid loss: {:.4f}, accuracy: {:.2f}%, auc: {:.4f}'.format(
                        epoch,
                        float(valid_loss.result()),
                        float(valid_accuracy.result()) * 100,
                        float(valid_auc.result())))
        return train_loop
    """