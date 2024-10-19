from utils.imports import *
from utils.variables import *
from lib.preprocessing import *
from lib.visualization import *
from lib.callbacks import *



class ManualLearningRateScheduler:
    def __init__(self, model, patience=5, factor=10, min_lr=1e-7):
        self.model = model
        self.patience = patience  # How many epochs to wait before reducing the LR
        self.factor = factor      # Factor by which to reduce the LR
        self.min_lr = min_lr      # Minimum allowed learning rate
        self.wait = 0             # Counter for epochs without improvement
        self.best_loss = float('inf')

    def adjust_learning_rate(self, current_loss):
        """Adjust learning rate if validation loss doesn't improve."""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                old_lr = float(self.model.optimizer.learning_rate)
                new_lr = old_lr / self.factor
                # new_lr = format(new_lr, '.4f')
                if new_lr < self.min_lr:
                    st.session_state.stop_train = 1

                # Apply the new learning rate
                self.model.optimizer.learning_rate.assign(new_lr)
                print(f"Reducing learning rate to {new_lr}.")
                
                self.wait = 0  # Reset wait counter



class NNRegressionModel(tf.keras.Model):
    """
    Base model class for building and training neural network models.

    This class is the base of all dense models that will be made. The architecture and the hyper parameters will be changeable
    in the child classes.

    Attributes:
        input_shape (tuple): Shape of the input data.
        config (dict): Configuration dictionary with model parameters.
        model (tf.keras.Model): Keras model built using the provided configuration.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initializes the BaseModel.

        Args:
            config (dict): Configuration dictionary with model parameters.
        """
        super().__init__(**kwargs)
        self.model_path = BEST_MODEL_PATH_REG  # Replace with actual path
        self.config = config
        self.model = None  # Initialize the model attribute

    def add_dense_block(self, units):
        """Adds a dense block to the model."""
        self.model.add(layers.Dense(units))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ELU(alpha=self.config["alpha_elu"]))
        self.model.add(layers.Dropout(self.config["dropout"]))
        self.model.add(visualkeras.SpacingDummyLayer(spacing=40))

    def build_model(self, input_shape):
        """Builds the neural network model."""
        self.model = models.Sequential()  # Initialize the Sequential model

        self.model.add(layers.Input(shape=input_shape))
        
        
        hidden_layers = [value for key, value in self.config.items() if key.startswith('dense')]
        for units in hidden_layers:
            self.add_dense_block(units)
        
        self.model.add(layers.Dense(1))  # Output layer for regression
        # From here self.model is built.
        self.upload_model_archi()  # Upload the model architecture
        optimizer = optimizers.deserialize(self.config["optimizer"])
        print("Just before the compilation")
     
        self.model.compile(optimizer=optimizer, loss=self.config["loss"], metrics=self.config["metrics"])
      

        return self.model
    
    # def compile(self, optimizer, loss, metrics):
    #     self.model.compile( optimizer, loss, metrics)
    def summary(self):
        return self.model.summary()
    
    def save(self, model_path):
        return self.model.save(model_path)
        
    def upload_model_archi(self):
        """Uploads the model architecture visualization to Weights & Biases (Wandb)."""
        vis_net = visualkeras.layered_view(self.model, min_xy=20, max_xy=4000, min_z=20, max_z=4000, padding=100, scale_xy=5, scale_z=20, spacing=20, one_dim_orientation='x', legend=True, draw_funnel=True)
        st.image(vis_net, caption="Network Architecture", use_column_width=True)

    def get_callbacks(self):
        """Gets the list of callbacks for model training."""
        return [
            callbacks.ReduceLROnPlateau(
                monitor=self.config["monitor"],
                factor=self.config["reduce_lr_factor"],
                patience=self.config["reduce_lr_patience"],
                min_lr=self.config["min_lr"]
            ),
            # callbacks.EarlyStopping(
            #     monitor=self.config["monitor"],
            #     mode='min',
            #     patience=self.config["early_stopping_patience"],
            #     restore_best_weights=True  # Change to True to restore best weights
            # ),
            # callbacks.ModelCheckpoint(
            #     filepath=self.model_path,
            #     monitor=self.config["metrics"][0],
            #     save_best_only=True
            # ),
        ]

    def train(self, X_train, y_train, X_val, y_val):
        """Trains the model and returns it with the history logs."""
        if "metrics_df" not in st.session_state:
            st.session_state.metrics_df = pd.DataFrame(columns=['Loss', 'MAE', 'Val Loss', 'Val MAE', 'Learning Rate'])

        progress_text = "Training will start soon"
        bar = st.progress(0, text=progress_text)
        history = []
        metrics_df = pd.DataFrame(columns=['Loss', 'MAE', 'Val Loss', 'Val MAE', 'Learning Rate'])
        st.session_state.metrics_df = metrics_df

        if not st.session_state.stop_train:
            with st.empty():
                scheduler = ManualLearningRateScheduler(self.model, patience=3)
                for epoch in range(self.config["epochs"]):
                    if st.session_state.stop_train:  # Check if training should stop
                        print(f"Training stopped at epoch {epoch + 1}")
                        st.info("The training has been stopped because some condition have been reached")
                        st.session_state.stop_train = 0
                        break
                    
                    # Train for one epoch
                    history_epoch = self.model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=1,  # Train for one epoch
                        batch_size=self.config["batch_size"],
                        verbose=1,
                        callbacks=self.get_callbacks()
                    )

                    # Append the history from this epoch
                    history_epoch = history_epoch.history
                    current_val_loss = history_epoch['val_loss'][-1]

                    # Adjust learning rate if needed
                    scheduler.adjust_learning_rate(current_val_loss)
                    metrics_row = pd.DataFrame({
                        'Loss': round(history_epoch["loss"][0], 4),
                        'MAE': round(history_epoch["mae"][0], 4),
                        'Val Loss': round(history_epoch["val_loss"][0], 4),
                        'Val MAE': round(history_epoch["val_mae"][0], 4),
                        'Learning Rate': f"{history_epoch["learning_rate"][0]:.0e}"
                    }, index=[epoch])

                    history.append(history_epoch)
                    st.session_state.metrics_df = pd.concat([st.session_state.metrics_df, metrics_row])
                    st.dataframe(st.session_state.metrics_df, height=200, width=800)

                    percent = int(100 * epoch / self.config["epochs"] + 1)
                    bar.progress(percent, text=f"Training: {percent}%")
                    st.session_state.history = history

        return self.model, history

    def evaluate(self, X_test, y_test):
        """Evaluates the model on test data."""
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        """Makes predictions using the model."""
        return self.model.predict(X)
    

def train_nn(X,y,epochs, test_size, val_size):

    """
    Trains a neural network model for temperature prediction, either for classification or regression tasks, 
    depending on the step parameter. This function sets up the model, prepares the data, configures the training 
    process, and manages the training using parameters provided in the input.

    Parameters:
        params_list (tuple): A tuple containing two elements:
            - params_to_take (list): The parameters to keep from the dataset.
            - params_to_drop (list): The parameters to remove from the dataset.
        step (int, optional): The step number indicating the type of training. 
            If None or 0, the task is a regression, otherwise it performs classification. Default is None.
        
    Returns:
        None. This function trains the model and logs the results to WandB. 
        The model and its history are saved, and visualizations of the results are generated.

    """

    # Layers
    dense_1 = 150
    dense_2 = 256
    dense_3 = 150
    dense_4 = 64
    dense_5 = 32
    
    # Hyper parameters
    frac_urb_train = 0.8
    dropout = 0.10
    alpha_elu = 1.0
    initial_lr =  1e-2
    weight_decay = 4e-4
    min_lr  =  1e-9

    batch_size = 512
    optimizer = optimizers.Nadam(learning_rate=initial_lr, weight_decay=weight_decay)
    optimizer_dict = optimizers.serialize(optimizer)
    monitor = "val_loss"

    # Metrics
    loss = "mse"
    metrics = ["mae"]

    # Callback parameters
    early_stopping_patience = 25
    reduce_lr_factor = 0.2
    reduce_lr_patience =  7
    
    # Other parameters
    # model = None

    config={
        "dense_1": dense_1,
        "dense_2": dense_2,
        "dense_3": dense_3,
        "dense_4": dense_4,
        "dense_5": dense_5,

        "epochs":epochs,
        "batch_size": batch_size,
        "loss": loss, 
        "optimizer" : optimizer_dict,
        "metrics": metrics,
        "monitor": monitor, 

        "frac_urb_train" : frac_urb_train,
        "initial_lr": initial_lr,
        "min_lr" : min_lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "alpha_elu": alpha_elu,
        
        "early_stopping_patience": early_stopping_patience,
        "reduce_lr_factor": reduce_lr_factor,
        "reduce_lr_patience": reduce_lr_patience,
    }
    save_config(config)
    print("Creating input...") 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size , random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size , random_state=42)
    
    # Scale the data    
    scaler = StandardScaler()
    X_train, X_test, X_val, scaler = scale_data(X_train, X_test, X_val, scaler)   
    print(X_train.shape)
    input_shape = (X_train.shape[1],)
    
    # Model init
    if not st.session_state.stop_train : 
        st.session_state.model = NNRegressionModel(config=config)
        print("Not in the class",st.session_state.model)
        st.session_state.model.build_model(input_shape=input_shape)
        print("After the build",st.session_state.model.summary())
        print(st.session_state.model)
        
        print("Training the model...")
        st.button("Stop training the neural network right now", on_click=callback_stop_train)
        st.session_state.model.train(X_train, y_train, X_val, y_val)  
        

    st.session_state.stop_train = 0
    history = compute_history(st.session_state.history)
    print(history)
    # print(st.session_state.model.summary())
    save_history(history,MODEL_HISTORY_REG)
    loss_and_metrics_vis(history)
    basic_visualization(X_test=X_test, y_test=y_test, model=st.session_state.model)
    return st.session_state.model, scaler
