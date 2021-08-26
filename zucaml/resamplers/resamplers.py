# from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# from zucaml.transformers.log import LogNormalizer
# from zucaml.transformers.iforest import iForestScore
# from zucaml.transformers.kmeans import KMeansDistances

# def get_transformer(transformer):
    
#     if isinstance(transformer, str):

#         if transformer.lower() == 'passthrough':
#             return 'passthrough'
#         elif transformer.lower() == 'standard_scaler':
#             return StandardScaler()
#         elif transformer.lower() == 'minmax_scaler':
#             return MinMaxScaler()
#         elif transformer.lower() == 'log_normalizer':
#             return LogNormalizer()
#         elif transformer.lower() == 'iforest_score':
#             return iForestScore()
#         elif transformer.lower() == 'kmeans_distances':
#             return KMeansDistances()
#         else:
#             print(f'\n Unknown transformer: {str(transformer)} \n')
#             return

#     elif isinstance(transformer, list):
        
#         return Pipeline([('step_' + str(i), v) for i, v in enumerate(transformer)])
        
#     else:
#         return transformer