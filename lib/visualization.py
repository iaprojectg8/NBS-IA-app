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
    pred_range = range(floor(min_pred), ceil(max_pred) + 1, 1)
    bins=pred_range

    
    # Reshape predictions to match the shape of y_test
    pred = np.reshape(pred, y_test.shape)
    

    
    # Determine which predictions are within the threshold of the ground truth
    well_predicted = (abs(pred - y_test) <= threshold)
    well_predicted_array = pred[well_predicted]
    others = pred[~well_predicted]

    # Calculate the percentage of well-predicted values
    well_predicted_counts, bin_edges = np.histogram(well_predicted_array, bins=bins)
    total_counts, _ = np.histogram(np.concatenate([well_predicted_array, others]), bins=bins)
    percentages = well_predicted_counts / total_counts
    percentages = [0 if (percentage < 1e-5 or np.isnan(percentage)) else round(percentage, 2) for percentage in percentages]

    # Trim leading and finishing 0 to fit the histogram
    percentages = np.trim_zeros(percentages)
   
    # Create the first figure for distribution comparison
    fig1 = go.Figure()
    print("len_bins",len(bins))

    fig1.add_trace(go.Histogram(
        x=pred.flatten(), 
        nbinsx=len(bins),
        xbins=dict(
            start=floor(min(pred)),  # Starting value for bins
            end=ceil(max(pred)),    # Ending value for bins
            size=1              # Bin size (step) set to 1
        ), 
        name='Prediction', 
        marker_color='blue', 
        opacity=0.8))
    
    fig1.add_trace(go.Histogram(
        x=y_test.flatten(), 
        nbinsx=len(bins), 
        xbins=dict(
            start=floor(min(pred)),  # Starting value for bins
            end=ceil(max(pred)),    # Ending value for bins
            size=1              # Bin size (step) set to 1
        ),
        name='Ground Truth', 
        marker_color='green', 
        opacity=0.8))

    fig1.update_layout(
        title=f"{title} - Distribution Comparison",
        xaxis_title='Intervals',
        yaxis_title='Count',
        barmode='group',
        bargroupgap=0.1,
        legend=dict(
            font=dict(size=10),  # Adjust the legend font size
            orientation="v",  # Horizontal legend
            y=0.8,  # Push the legend below the plot
            xanchor="right",  # Center the legend
            
        )
    )

    # Create the second figure for well-predicted and mispredicted values
    fig2 = go.Figure()

    # Calculate the overall percentage of well-predicted values
    well_predicted_all = sum(well_predicted_counts) / sum(total_counts)
    font_color = "white"
    
    fig2.add_trace(go.Histogram(
        x=others.flatten(), 
        nbinsx=len(bins),
        name=f'{1 - well_predicted_all:.2f} Mispredicted',
        marker_color='red', 
        opacity=1))
    
    fig2.add_trace(go.Histogram(
        x=well_predicted_array.flatten(), 
        nbinsx=len(bins),
        name=f'{well_predicted_all:.2f} of well predicted \nwith {threshold} precision',
        text=[percentage for percentage in percentages],
        textposition='auto',
        textangle=0,
        textfont=dict(color=f"{font_color}",weight=90,shadow="auto"),
        marker_color='green', 
        opacity=1))
    
    fig2.update_layout(
        title=f"{title} - Well Predicted vs Mispredicted",
        xaxis_title='Intervals',
        yaxis_title='Count',
        barmode='stack',
        bargap=0.1,
        bargroupgap = 0,
        autosize=True,
        legend=dict(
            font=dict(size=10),  # Adjust the legend font size
            orientation="v",  # Horizontal legend
            yanchor="auto",  # Push the legend below the plot
            xanchor="auto",  # Center the legend
            
        )
    )


    # Display the figures in Streamlit
    st.session_state.stat_on_pred_fig1 = fig1
    st.session_state.stat_on_pred_fig2 = fig2
    st.plotly_chart( st.session_state.stat_on_pred_fig1)
    st.plotly_chart( st.session_state.stat_on_pred_fig2)

    if wandb_push:
        import wandb
        wandb.log({
            f"{title} - Distribution Comparison": wandb.Image(fig1),
            f"{title} - Well Predicted vs Mispredicted": wandb.Image(fig2)
        })



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
    
    # Create bar chart for residuals
    labels = [f'[{interval[0]}, {interval[1]}]' for interval in intervals]

    # Colors needed to create the second graphs
    colors = [
        'rgba(0, 153, 0, 0.6)',          # Equivalent to [0, 0.60261373, 0, 1]
        'rgba(0, 209, 0, 0.6)',          # Equivalent to [0, 0.82223333, 0, 1]
        'rgba(59, 255, 0, 0.6)',         # Equivalent to [0.2300549, 1, 0, 1]
        'rgba(239, 237, 0, 0.6)',        # Equivalent to [0.93591569, 0.92807255, 0, 1]
        'rgba(255, 169, 0, 0.6)',        # Equivalent to [1, 0.6627451, 0, 1]
        'rgba(243, 0, 0, 0.6)',          # Equivalent to [0.95556667, 0, 0, 1]
        'rgba(204, 12, 12, 0.6)'         # Equivalent to [0.8, 0.04705882, 0.04705882, 1]
    ]
    font_color = "white"
    fig1 = go.Figure()
    
    # Bar chart for residual amounts
    fig1.add_trace(go.Bar(
        x=labels,
        y=residuals_amount_per_interval,
        marker=dict(color=colors),
        text=[f'{percentage:.1f}%' for percentage in percentages],
        textposition='auto',
        textfont=dict(color=f"{font_color}",weight=90,shadow="auto"),
        name='Residual Amounts'
    ))  

    

    fig1.update_layout(
        title="Residual amounts in each interval",
        xaxis_title="Residual interval (°C)",
        yaxis_title="Residual amount",
        barmode='group',
        autosize=True,
        
    )

    # Scatter plot for predicted vs ground truth temperatures
    print(y_test.shape[0])
    if y_test.shape[0] > 20000:
        indices = np.random.choice(len(y_test), 20000, replace=False) 
        y_test = y_test[indices]
        y_pred = y_pred[indices]


    scatter_trace = go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name=f'R2 = {r2:.2f}       R2*= {adjusted_r2:.2f}\nMSE = {mse:.2f}    MAE = {mae:.2f}',
        marker=dict(color=' rgb(0,191,255, 1)', size=8)
    )

    # Add interval lines to the scatter plot
    lines = []
    for i, interval in enumerate(intervals[:-1]):
        line_color = colors[i]
        lines.append(go.Scatter(
            x=y_test, y=y_test + interval[1],
            mode='lines',
            name=f'+/- {interval[1]}',
            line=dict(color=line_color)
        ))
        lines.append(go.Scatter(
            x=y_test, y=y_test - interval[1],
            mode='lines',
            line=dict(color=line_color),
            showlegend=False
        ))

    fig2 = go.Figure(data= [scatter_trace] + lines  )
    fig2.update_layout(
        title="Predicted vs Ground Truth temperature",
        xaxis_title="Ground Truth Temperatures",
        yaxis_title="Predicted Temperatures",
        autosize=True,
        legend=dict(
            font=dict(size=10),  # Adjust the legend font size
            orientation="v",  # Horizontal legend
            yanchor="auto",  # Push the legend below the plot
            xanchor="auto",  # Center the legend
            x=0.05,
            y=1,
            bgcolor='rgba(0, 0, 0, 0)',
            
        )
    )
    print("All graphs are ok")
    st.session_state.results_fig1 = fig1
    st.session_state.results_fig2 = fig2
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)