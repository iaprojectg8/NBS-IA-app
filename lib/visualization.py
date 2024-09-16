from utils.imports import *

def stat_on_prediction(pred, y_test, threshold, wandb_push:bool, title: str):
    """
    Generates statistical plots to compare predictions with ground truth values and logs them to Weights and Biases (wandb).
    
    Args:
    pred (np.ndarray): Predicted values.
    y_test (np.ndarray): Ground truth values.
    threshold (float): Threshold to consider a prediction as well-predicted.
    title (str): Title for the plot to be logged in wandb.
    
    """
    # Compute the min and max values for predictions and ground truth
    min_pred = pred.min()
    max_pred = pred.max()
    min_y_test = y_test.min()
    max_y_test = y_test.max()
    
    # Create ranges and bins for the histograms
    pred_range = range(int(min_pred), int(max_pred) + 1, 1)
    y_test_range = range(int(min_y_test), int(max_y_test) + 1, 1)
    bins = sorted(set(pred_range).union(set(y_test_range)))

    # Reshape predictions to match the shape of y_test
    pred = np.reshape(pred, y_test.shape)
    
    # Determine which predictions are within the threshold of the ground truth
    well_predicted = (abs(pred - y_test) <= threshold)
    well_predicted_array = pred[well_predicted]
    others = pred[~well_predicted]

    # Calculate the percentage of well-predicted values
    well_predicted_counts, _ = np.histogram(well_predicted_array, bins=bins)
    total_counts, _ = np.histogram(np.concatenate([well_predicted_array, others]), bins=bins)
    percentages = well_predicted_counts / total_counts
    percentages = [0 if (percentage < 1e-5 or np.isnan(percentage)) else round(percentage, 2) for percentage in percentages]

    # Create the first figure for distribution comparison
    fig1 = go.Figure()

    fig1.add_trace(go.Histogram(x=pred.flatten(), nbinsx=len(bins), name='Prediction', marker_color='blue', opacity=0.5, histnorm='probability'))
    fig1.add_trace(go.Histogram(x=y_test.flatten(), nbinsx=len(bins), name='Ground Truth', marker_color='green', opacity=0.5, histnorm='probability'))

    fig1.update_layout(
        title=f"{title} - Distribution Comparison",
        xaxis_title='Intervals',
        yaxis_title='Probability',
        barmode='overlay'
    )

    # Create the second figure for well-predicted and mispredicted values
    fig2 = go.Figure()

    # Calculate the overall percentage of well-predicted values
    well_predicted_all = sum(well_predicted_counts) / sum(total_counts)
    
    fig2.add_trace(go.Histogram(x=well_predicted_array.flatten(), nbinsx=len(bins), name=f'{well_predicted_all:.2f} of well predicted\nwith {threshold} precision', marker_color='green', opacity=0.7))
    fig2.add_trace(go.Histogram(x=others.flatten(), nbinsx=len(bins), name=f'{1 - well_predicted_all:.2f} Mispredicted', marker_color='red', opacity=0.7))

    # Annotate histogram with percentages
    bin_centers = [bin + 0.5 for bin in bins]  # Center of each bin
    for i, percentage in enumerate(percentages):
        if percentage != 0:
            fig2.add_annotation(
                x=bin_centers[i],
                y=max(well_predicted_counts[i], total_counts[i]),  # Position on top of the bar
                text=str(percentage),
                showarrow=False,
                font=dict(color='black', size=10),
                align='center'
            )
    
    fig2.update_layout(
        title=f"{title} - Well Predicted vs Mispredicted",
        xaxis_title='Intervals',
        yaxis_title='Count',
        barmode='stack'
    )


    # Display the figures in Streamlit
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

    if wandb_push:
        import wandb
        wandb.log({
            f"{title} - Distribution Comparison": wandb.Image(fig1),
            f"{title} - Well Predicted vs Mispredicted": wandb.Image(fig2)
        })

def visualization(classif,classes_to_temps,test_inputs,wandb_title, model): 
    """
    Visualizes the model's performance by predicting on the test set, calculating statistics, and generating plots.

    Parameters:
    model_type (str): The type of model (e.g., "classifier").
    model_path (str): Path to the saved model.
    classes_to_temps (dict): Dictionary mapping class labels to temperatures.
    inputs (tuple): A tuple containing datasets (train, validation, test splits).
    wandb_title (str): Title for the plots to be logged in wandb.
    model (optional): Loaded model object. If None, the model is loaded from the model_path.

    """
    # Unpack the data
    X_test,y_test = test_inputs

    # Make the prediction and the evaluation
    y_pred = model.predict(X_test.copy())
    # comparison_to_csv(X_test, y_test, y_pred)
    
    # Take the labels back
    if classif:
        y_pred = np.argmax(y_pred, axis = 1)
        y_pred = np.array([classes_to_temps[label] for label in y_pred])
        y_test = np.array([classes_to_temps[label] for label in y_test])
        

    stat_on_prediction(y_pred,y_test,threshold=0.5, wandb_push=1,title="Prediction vs Ground Truth")

    # Evaluation of the model
    n,k = X_test.shape
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / ( n - k- 1)
    print("\nFor Y pred:")
    print("R2:",r2,"MSE:",mse, "MAE:",mae)
    print("Adjusted R2:",adjusted_r2)
    
    # Reshaping the y_pred because it is (n,1), instead of (n,) for y_test, which is a problem to broadcast in the substraction operation later
    y_pred = np.reshape(y_pred,y_test.shape)
    # Residuals calculation
    residuals = abs(y_test - y_pred)

    # Intervals definition
    
    intervals = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0,2.0), (2.0,5.0),(5.0,10.0)]  # Liste des intervals des intervalles
    # intervals = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),(0.5, 0.6),(0.6, 0.7),(0.7, 0.8),(0.8, 0.9),(0.9, 1.0), (1.0,1.2), (1.2,1.5), (1.5,2.0), (2.0,5.0),(5.0,10.0)]  # nouvelle liste

    # Residuals interval calculation
    residuals_amount_per_interval = []
    total_residual_amount = len(residuals)

    # Amount of residuals per interval
    for i,interval in enumerate(intervals):
        if i==0:
            nb_residus = sum((residuals >= interval[0]) & (residuals <= interval[1]))
        else: 
            nb_residus = sum((residuals > interval[0]) & (residuals <= interval[1]))
        residuals_amount_per_interval.append(nb_residus)

    
    percentages = [nb / total_residual_amount * 100 for nb in residuals_amount_per_interval]
    
    # Create the figure to plot
    fig = plt.figure(figsize=(13, 6))
    
    # Plot 1
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    colors = cm.nipy_spectral(np.linspace(0.45, 0.95, len(intervals)))
    bars = ax1.bar(range(len(intervals)), residuals_amount_per_interval, color=colors, alpha=0.7)
    

    # Put the percentages on the bars
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{percentages[i]:.1f}%', ha='center', va='bottom')

    # Put the right label for each interval 
    labels = [f'[{interval[0]}, {interval[1]}]' for interval in intervals]
    ax1.set_xticks(range(len(intervals)), labels)
    ax1.set_xlabel('Residual interval (°C)')
    ax1.set_ylabel('Residual amount')
    ax1.set_title('Residual amounts in each interval')
    for label in ax1.get_xticklabels():
        label.set_rotation(45)

    # Plot 2 
    
    ax2.scatter(y_test, y_pred, label=f'R2 = {r2:.2f}       R2*= {adjusted_r2:.2f}\nMSE = {mse:.2f}    MAE = {mae:.2f}')
    # Make all the line to define the +/- intervals to have a better idea on the distribution
    for i in range(colors.shape[0]-1):
        ax2.plot( y_test,y_test+intervals[i][1], color=colors[i], label=f'+/- {intervals[i][1]}')  # Plotting the line of ground truth temp
        ax2.plot( y_test,y_test-intervals[i][1], color=colors[i])  # Plotting the line of ground truth temp
    ax2.set_xlabel('Ground Truth Temperatures')
    ax2.set_ylabel('Predicted Temperatures')
    ax2.set_title('Predicted vs Ground Truth temperature')
    ax2.legend(loc="upper left")

    plt.tight_layout()
    wandb.log({wandb_title : wandb.Image(plt)})
    # plt.show()

def basic_visualization(X_test,y_test,model): 
    """
    Visualizes the model's performance by predicting on the test set, calculating statistics, and generating plots.

    Parameters:
    model_type (str): The type of model (e.g., "classifier").
    model_path (str): Path to the saved model.
    classes_to_temps (dict): Dictionary mapping class labels to temperatures.
    inputs (tuple): A tuple containing datasets (train, validation, test splits).
    wandb_title (str): Title for the plots to be logged in wandb.
    model (optional): Loaded model object. If None, the model is loaded from the model_path.

    """

    # Make the prediction and the evaluation
    y_pred = model.predict(X_test.copy())
    # comparison_to_csv(X_test, y_test, y_pred)
        

    stat_on_prediction(y_pred,y_test,threshold=0.5,wandb_push=0,title="Prediction vs Ground Truth")

    # Evaluation of the model
    n,k = X_test.shape
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / ( n - k- 1)
    print("\nFor Y pred:")
    print("R2:",r2,"MSE:",mse, "MAE:",mae)
    print("Adjusted R2:",adjusted_r2)
    
    # Reshaping the y_pred because it is (n,1), instead of (n,) for y_test, which is a problem to broadcast in the substraction operation later
    y_pred = np.reshape(y_pred,y_test.shape)
    # Residuals calculation
    residuals = abs(y_test - y_pred)

    # Intervals definition
    
    intervals = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0,2.0), (2.0,5.0),(5.0,10.0)]  # Liste des intervals des intervalles
    # intervals = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),(0.5, 0.6),(0.6, 0.7),(0.7, 0.8),(0.8, 0.9),(0.9, 1.0), (1.0,1.2), (1.2,1.5), (1.5,2.0), (2.0,5.0),(5.0,10.0)]  # nouvelle liste

    # Residuals interval calculation
    residuals_amount_per_interval = []
    total_residual_amount = len(residuals)

    # Amount of residuals per interval
    for i,interval in enumerate(intervals):
        if i==0:
            nb_residus = sum((residuals >= interval[0]) & (residuals <= interval[1]))
        else: 
            nb_residus = sum((residuals > interval[0]) & (residuals <= interval[1]))
        residuals_amount_per_interval.append(nb_residus)

    
    percentages = [nb / total_residual_amount * 100 for nb in residuals_amount_per_interval]
    
    # Create the figure to plot
    fig = plt.figure(figsize=(13, 6))
    
    # Plot 1
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    colors = cm.nipy_spectral(np.linspace(0.45, 0.95, len(intervals)))
    bars = ax1.bar(range(len(intervals)), residuals_amount_per_interval, color=colors, alpha=0.7)
    

    # Put the percentages on the bars
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{percentages[i]:.1f}%', ha='center', va='bottom')

    # Put the right label for each interval 
    labels = [f'[{interval[0]}, {interval[1]}]' for interval in intervals]
    ax1.set_xticks(range(len(intervals)), labels)
    ax1.set_xlabel('Residual interval (°C)')
    ax1.set_ylabel('Residual amount')
    ax1.set_title('Residual amounts in each interval')
    for label in ax1.get_xticklabels():
        label.set_rotation(45)

    # Plot 2 
    
    ax2.scatter(y_test, y_pred, label=f'R2 = {r2:.2f}       R2*= {adjusted_r2:.2f}\nMSE = {mse:.2f}    MAE = {mae:.2f}')
    # Make all the line to define the +/- intervals to have a better idea on the distribution
    for i in range(colors.shape[0]-1):
        ax2.plot( y_test,y_test+intervals[i][1], color=colors[i], label=f'+/- {intervals[i][1]}')  # Plotting the line of ground truth temp
        ax2.plot( y_test,y_test-intervals[i][1], color=colors[i])  # Plotting the line of ground truth temp
    ax2.set_xlabel('Ground Truth Temperatures')
    ax2.set_ylabel('Predicted Temperatures')
    ax2.set_title('Predicted vs Ground Truth temperature')
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()