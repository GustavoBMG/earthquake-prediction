from enum import Enum, IntEnum

problems = IntEnum('Problems', 'REGRESSION BINARY BINARY_AUTENC BINARY_CONV') #threshold, multiclass, multilabel zero-inflated

cv_strategies = IntEnum('CV_Strategies', 'KFOLD TIME')