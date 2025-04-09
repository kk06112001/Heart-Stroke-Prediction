from src.heartstrokeprediction.config.configuration import ConfigurationManager
from src.heartstrokeprediction.components.model_eval import ModelEvaluation
from src.heartstrokeprediction import logger

STAGE_NAME = "Model Training stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def evalmodel(self):
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        model_eval = ModelEvaluation(config=model_eval_config)
        model_eval.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluation()
        obj.evalmodel()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e