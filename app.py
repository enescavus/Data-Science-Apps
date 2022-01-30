##################################################################################
# @author     : Enes Cavus
# @subject    : HyperParameter With Optuna Framework + Random Forest + Streamlit
# @date       : Jan 29 2022
# source code : https://github.com/enescavus
##################################################################################

#imports

# custom libraries
import data_operations # my custom data loading-preprocessing library

# optuna related imports
import optuna

# sklearn - algorithms and scoring 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# streamlit app, stats, visualizations, dataframe operations
import streamlit as st
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
import plotly.offline as pyo

# full width streamlit page 
st.set_page_config(layout="wide")

# run - full module
def app_main():
    run_optuna(n_estimators, max_depth, min_samples_leaf, min_samples_split, n_trials)
        
# data operations
X,y= data_operations.data_load()

# -------------------------------------------------------------------- OPTUNA --------------------------------------------------------------------

# optuna main operations in one complete function 
def run_optuna(n_estimators, max_depth, min_samples_leaf, min_samples_split,n_trials):

    selected_max_features = check_max_features_options()
    if not selected_max_features:
        st.error("Please Select at least one max_features type")
    else:
        hyper_params = {"n_estimators" : n_estimators , "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "min_samples_split":min_samples_split, 'selected_max_features':selected_max_features}
        st.info("Your hyperparameters selected as : {}".format(
            {str(hp):value for hp,value in hyper_params.items()}
        ) )
        def objective(trial):
            params = {
                "n_estimators" : trial.suggest_int("n_estimators", n_estimators[0],n_estimators[1]),
                "max_depth" : trial.suggest_int("max_depth", max_depth[0],max_depth[1]),
                "min_samples_leaf" : trial.suggest_int("min_samples_leaf", min_samples_leaf[0],min_samples_leaf[1]),
                'min_samples_split': trial.suggest_int('min_samples_split', min_samples_split[0], min_samples_split[1]),
                "max_features" : trial.suggest_categorical("max_features", selected_max_features),
            }
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            rf_model = RandomForestClassifier(**params,
                                            n_jobs=-1,
                                            random_state=0)

            acc = mean(cross_val_score(rf_model, X,y, cv=skf, scoring='accuracy', n_jobs=-1))
            return acc

        # ###################### Spinner & Info ################################
        with st.spinner('Wait for it... Tuning the model for {} trials'.format(n_trials)):
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
        st.success('Done!, Model trainings and tuning is completed. Here are your best fitting hyper-parameters and best accuracy!')
        param1, param2, param3, param4, param5,score = st.columns(6)
        cols = [param1, param2, param3, param4, param5]
        for (param,value), col in zip(study.best_params.items(), cols):
            with col:
                st.info("{} : {}".format(param,value))
        with score:
            st.success("Best Acc : {}".format(round(study.best_value, 3)))

        # ###################### Visual Analysis ###############################
        # hyperParameter Impotance and Optuna History Plots analysis
        fig = plt.figure(figsize = (10, 5))
        fig_par_importances = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig('figures/fig_par_importances.jpeg', bbox_inches='tight')
        fig_history = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig('figures/fig_history.jpeg', bbox_inches='tight')

        st.markdown("## Visual Analysis")
        st.info("**Hyperparameter Importances** shows us how important each parameter is for our model. __**History Plot**__ shows us the history records of the Optuna trials. Objective value is the return value of objective function which is accuracy in our case and we are trying to maximize it. If you tune the model with a lot of trials you can see the changes Optuna made.")
        fig_par_importances_col, fig_history_col = st.columns(2)
        with fig_par_importances_col:
            st.image("figures/fig_par_importances.jpeg")
        with fig_history_col:
            st.image("figures/fig_history.jpeg")

        # slice plot and parallel coordinates plot analysis
        slice_plot = optuna.visualization.plot_slice(study)
        st.info("When we look at the insights of __**Slice Plot**__ , we can see that there are some intensities in specific ranges for each hyperparameter. If a hyperparameter gets its high scores in high limit we might want to increase the limit or vice versa.")
        st.plotly_chart(slice_plot,use_container_width=True)
        st.info("__**parallel_coordinates**__  plot is similar to 'slice plot'. We can see each connections between the hyperparameters. When we analyze the graph we can say that model gives us better scores with lower max features, lower max depth and lower estimator numbers. Note: This data has only 150 instances so that analyses may vary! Main focus of this app is understanding Optuna and Creating an app with Streamlit!")
        parallel_coordinates = optuna.visualization.plot_parallel_coordinate(study)
        st.plotly_chart(parallel_coordinates,use_container_width=True)

# -------------------------------------------------------------------- sidebar design --------------------------------------------------------------------

# app header - sidebar
st.sidebar.markdown("# HyperParameter Tuning With Optuna")

# related links and contact
sidebar_info_html_string = """<p><b></br>You can check out the blog post of this application here on <a href="https://medium.com/@enesblog/optuna-ve-streamlit-ile-hiper-parametre-optimizasyonu-random-forest-f831db7ec405" target="_blank">Medium</a>, 
                            contact with developer on <a href="https://www.linkedin.com/in/enes-cavus" target="_blank">LinkedIn</a>, 
                            see the code on <a href="https://github.com/enescavus" target="_blank">Github</a></br></br>By Enes Cavus</p></b>"""
st.sidebar.markdown(sidebar_info_html_string, unsafe_allow_html=True)

# sliders - hyperparamaters
# each hyperparameter's ranges specifically selected for the data , iris dataset is an easy to use dataset
n_estimators = st.sidebar.slider('n_estimators -- Select a range of values',2, 100, (3, 20))
max_depth = st.sidebar.slider('max_depth -- Select a range of values',2, 50, (3, 10))
min_samples_leaf = st.sidebar.slider('min_samples_leaf -- Select a range of values',2, 10, (3, 4))
min_samples_split = st.sidebar.slider('min_samples_split -- Select a range of values',2, 30, (3, 15))
st.sidebar.write('Select max_features types:')
option_sqrt = st.sidebar.checkbox('sqrt')
option_auto = st.sidebar.checkbox('auto')
option_log2 = st.sidebar.checkbox('log2')
n_trials = st.sidebar.slider('How many iterations(trial) you want?',10, 100, 15)

# list operations for max_features values
def check_max_features_options():
    selected_max_features = []
    # check boolean value
    if option_sqrt:
        selected_max_features.append("sqrt")
    if option_auto:
        selected_max_features.append("auto")
    if option_log2:
        selected_max_features.append("log2")
            
    return selected_max_features


# -------------------------------------------------------------------- main page design --------------------------------------------------------------------

# app header - main page
st.markdown("# HyperParameter Tuning With Optuna")

# columns for info and data preview
info, data_preview = st.columns(2)

# first column of first row, information about the app and how to use
with info:
    main_page_info_html_string = """<p><i>Info & How to Use</i></br>Hi, this is an app for HyperParameter Tuning. You can specify the hyperparameter values for Random Forest Machine Learning algorithm.Use the left sidebar and make sure you did not skip any hyperparameter. Data selected automaticaly.
     Also you can see the dataset as form of pandas dataframe right side. 
    Iris dataset selected for making training and tuning steps faster. If you want more detail about dataset or you might want to read more explanations about how this app created, see this <a href="https://medium.com/@enesblog/optuna-ve-streamlit-ile-hiper-parametre-optimizasyonu-random-forest-f831db7ec405" target="_blank">blog post</a>.
    You will get best hpyperparameters and visual analysis after tuning. 
    </br></br>Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. 
    Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters. <a href="https://optuna.readthedocs.io/en/stable/" target="_blank">Official Optuna Documentation</a>.
    </p>

    """
    st.markdown(main_page_info_html_string, unsafe_allow_html=True)

# check the data and preview
with data_preview:
    data = pd.read_csv("data/iris_data.csv")
    try:
        data = data.drop("Unnamed: 0", axis=1)
    except:
        print("data doesn't have unnamed: 0 column")
    st.dataframe(data)


# -------------------------------------------------------------------- main function --------------------------------------------------------------------

# run the main function
if __name__ == '__main__':
    if st.button("...Start Tuning The Model With Selected HyperParameters..."):
        app_main()



# END - By Enes Cavus