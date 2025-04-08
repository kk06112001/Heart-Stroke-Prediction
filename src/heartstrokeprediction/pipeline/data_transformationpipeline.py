from src.heartstrokeprediction.config.configuration import ConfigurationManager
from src.heartstrokeprediction.components.data_transformation import DataTransformation
from src.heartstrokeprediction import logger

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass        
    def initiate_data_transformation(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.train_test_spliting()
    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e