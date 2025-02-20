```mermaid
flowchart TD
    A["main.py: Load & Save Config"]
    B["Data Cleaning<br>(data_cleaning.py)"]
    C["Data Preprocessing<br>(data_preparation.py)"]
    D["Model Building & Compilation<br>(model_architecture.py)"]
    E["Initial Training & Cross-Validation<br>(train_model.py)"]
    F["Result Filtering<br>(result_filtering.py)"]
    G["Config Update for Retraining"]
    H["Retraining / K-Fold CV<br>(retrain_kfold.py)"]
    I["Ensemble Search<br>(ensemble_search.py)"]
    J["Ensemble Evaluation<br>(ensemble_evaluation.py)"]
    K["Utilities & Metrics Logging<br>(utils.py)"]
    L["Visualization<br>(graphs.py)"]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L

    %% Parallel/Shared Data Sources
    C --- E
    C --- H
    C --- I
    C --- J

    %% Model building is used by both training and retraining stages
    D --- E
    D --- H


