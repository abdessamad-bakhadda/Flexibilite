import EJP_Predictor_Deployment as epd
import pickle

# Creating a pickle file for the classifier
name_poste_source = 'sum_conso_postes_sources_2015_2021.xlsx'
my_pipeline_sum = epd.pipeline_trained(name_poste_source,epd.function_to_apply)
filename = 'EJP-model.pkl'
pickle.dump(my_pipeline_sum , open(filename, 'wb'))