#!/bin/bash
python vs2013/PythonProj/roc_curve.py build/bin/rocs/predictROC.txt build/bin/rocs/roc.png "ROC Curve" "FPR" "TPR"
python vs2013/PythonProj/roc_curve.py build/bin/rocs/recognitionMissing.txt build/bin/rocs/recognitionMissing.png "Recognition Missings" "parameter" "error"
python vs2013/PythonProj/roc_curve.py build/bin/rocs/recognitionFalseDetections.txt build/bin/rocs/recognitionFalseDetections.png "Recognition False Detections" "parameter" "error"