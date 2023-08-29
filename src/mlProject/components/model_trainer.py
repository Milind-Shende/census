import pandas as pd
import os
from mlProject import logger
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler,MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain
import dill




class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        numeric_features= ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
        categorical_features=['workclass', 'education', 'marital_status', 'occupation', 'sex', 'country']

        # Numerical and Categorical Pipeline Transformation
        logger.info("Numerical and Categorical Pipeline Transformation")
        numeric_transformer = Pipeline(steps=[('Scaler', RobustScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))])

        # Numerical and Categorical Column Transformation
        logger.info("Numerical and Categorical Column Transformation")
        transformer = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                      ('cat', categorical_transformer, categorical_features)])
        logger.info("Transforming the Train and Test")
        train_x = transformer.fit_transform(train_x)
        test_x = transformer.transform(test_x)

        logger.info("Scaling train_y and test_y")
        target_scaler=LabelEncoder()
        train_y=target_scaler.fit_transform(train_y)
        test_y=target_scaler.fit_transform(test_y)

        
        

       

        logger.info("define Xgboost Classification")
        xgb_model = XGBClassifier(n_estimators=self.config.n_estimators,
                                scale_pos_weight=self.config.scale_pos_weight)

        logger.info("fit train_x and train_y")
        xgb_model.fit(train_x, train_y)

        # Save the model, transformer, and target_scaler
        model_filename = os.path.join(self.config.root_dir, self.config.model_name)
        transformer_filename = os.path.join(self.config.root_dir, self.config.transformer_name)
        target_scaler_filename = os.path.join(self.config.root_dir, self.config.target_name)

        with open(model_filename, 'wb') as model_file:
            dill.dump(xgb_model, model_file)

        with open(transformer_filename, 'wb') as transformer_file:
            dill.dump(transformer, transformer_file)

        with open(target_scaler_filename, 'wb') as target_scaler_file:
            dill.dump(target_scaler, target_scaler_file)