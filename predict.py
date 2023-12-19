import pandas as pd
import prophet
import mlflow

class ProphetModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def predict(self, context, model_input):
        future = self.model.make_future_dataframe(periods=model_input['periods'][0])
        return self.model.predict(future)

df = pd.read_csv(r'C:\Users\nico_\OneDrive\Portfolio\Stock Level Predictions\data\sales.csv', encoding='latin1')

# Assuming 'Order Date' is your date column and 'Order Quantity' is what you want to predict
df = df[['date', 'qty', 'prod_code']]
df.columns = ['ds', 'y', 'prod_code']

# Get list of unique products
products = df['prod_code'].unique()

for product in products:
    # Filter data for one product
    df_product = df[df['prod_code'] == product]
    
    # Initialize and fit the model
    m = prophet()
    m.fit(df_product)
    
    # Make future dataframe for prediction
    future = m.make_future_dataframe(periods=365)  # Predict for next 365 days
    
    # Predict
    forecast = m.predict(future)
    
    # Log model
    with mlflow.start_run():
        mlflow.log_param("ProductCode", product)
        mlflow.pyfunc.log_model("model", python_model=ProphetModel(m))