# save the code in this cell to "service.py"
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, Text

tag = bentoml.models.get("emotion_detection_lr:latest")
class_labels_d = tag.custom_objects.get('class_labels_d')

# create a bentoml runner from a stored bentoml model
# emotion_detection_lr_clf_runner = bentoml.models.get("emotion_detection_lr:latest").to_runner()
emotion_detection_lr_clf_runner = tag.to_runner()

# Create a BentoML Service object
# The service name will become the name of the Bento.
svc = bentoml.Service("emotion_detection_lr_classifier", runners=[emotion_detection_lr_clf_runner])

@svc.api(input=Text(), output=Text())
def classifyemotion(input_text: str) -> str:
    # this function name, 'classify' will be transformed into the HTTP endpoint /classify
    # but it can also be customised, see: https://docs.bentoml.org/en/latest/concepts/service.html
    # e.g. in the @svc.api decorator, adddir
    # route="/v2/models/my_model/versions/v0/infer",
    # print('\n\n')
    # print(type(input_text), input_text)
    # print('\n\n')
    
 
    class_num = emotion_detection_lr_clf_runner.predict.run([input_text])[0]
    class_lbl = class_labels_d.get(class_num)
    
    # print('\n\n')
    # print(type(result), ' -- ', result)
    # print('\n\n')
    # return "world says hello"    
    return class_lbl