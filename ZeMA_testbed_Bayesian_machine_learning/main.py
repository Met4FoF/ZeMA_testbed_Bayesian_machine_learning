"""
Zema Testbed Bayesian Machine Learning
created by: loïc Coquelin (LNE)
"""

# %%
from ZeMA_testbed_Bayesian_machine_learning.utils import Data, Model

# os.environ['MKL_THREADING_LAYER'] = 'GNU'


def test_pipeline(feature_extraction_param,
                  feature_selection_param,
                  data_processing_param,
                  dimensionality_reduction_param,
                  classification_param,
                  bayesian_fusion_param,
                  deployment_param):
    print('START PROGRAM')
    # %%
    '''
    ## DATA I/O
    Downloads / Imports and converts ADC mesurements to real SI units
    '''
    print('DOWNLOAD/IMPORT DATA')
    print('Downloads / Imports and converts ADC mesurements to real SI units')
    # %%
    options = {'vizualisation': False}
    # data url
    url = 'https://zenodo.org/record/1326278/files/Sensor_data_2kHz.h5'
    zema_data = Data()
    # download data
    zema_data.download_file(url)
    # import data
    zema_data.import_data()
    # get only part of the data 1 over 10 cycle is kept
    zema_data.get_data_subset(sub_sampling_factor=data_processing_param['sub_sampling_factor'])
    # Tranforming x axis into time domain.
    zema_data.transform_xaxis_to_time_domain()
    # Convert measured ADC values to SI units
    zema_data.convert_ADC_to_SI_units()
    # Ploting data examples from all sensors in time domain
    if options['vizualisation']:
        # Choose cycle number to plot
        cycle_number = int(input("Enter a number of example cycle you want to plot ( 1-6291 ): "))
        cycle_number = cycle_number - 1
        zema_data.plot_ADC_traces(cycle_number)
        zema_data.plot_traces(cycle_number)

    # %%
    '''
    ## Launch full process : Feature computation, selection and Classification
    '''

    print('Launch full process : Feature computation, selection and Classification')
    # %%

    zema_model = Model()
    zema_model.full_process(zema_data,
                            feature_extraction_param=feature_extraction_param,
                            feature_selection_param=feature_selection_param,
                            data_processing_param=data_processing_param,
                            dimensionality_reduction_param=dimensionality_reduction_param,
                            classification_param=classification_param,
                            bayesian_fusion_param=bayesian_fusion_param,
                            deployment_param=deployment_param)


if __name__ == '__main__':
    feature_extraction_param_ = {
        'method': 'fft',
        'percentage_of_features': 10
    }

    feature_selection_param_ = {
        'method': 'pearson_correlation',
        'nb_features_to_keep': 500
    }

    data_processing_param_ = {
        'percentage_training': 80,
        'data_normalization': True,
        'sub_sampling_factor': 1
    }

    dimensionality_reduction_param_ = {
        'active': False,
        'method': 'LDA',
        'n_components': 20,
        'plot': True
    }

    classification_param_ = {
        'method': 'LDA',
        'classifier_comparison': False,
        'plot': True
    }

    bayesian_fusion_param_ = {
        'method': 'RGS',
        'nb_samples_posterior': 10000,
        'target_accept': 0.8,
        'tune': 2000,
        'cores': 1,
        'chains': 4,
        'burn': 500,
        'thin': 1,
        'nb_samples_prediction': 10000,
        'sampler_init_method': 'jitter+adapt_diag'
    }

    deployment_param_ = {
        'active': False,
        'model_to_load': ''
    }

    test_pipeline(feature_extraction_param_,
                  feature_selection_param_,
                  data_processing_param_,
                  dimensionality_reduction_param_,
                  classification_param_,
                  bayesian_fusion_param_,
                  deployment_param_)
