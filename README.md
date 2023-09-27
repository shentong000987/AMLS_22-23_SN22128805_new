This is the final version of project, the previous temporary version can be access from: https://github.com/shentong000987/AMLS_22-23_SN22128805

PART1 - brief description of the organization of project

This Project is designed to complete the ELEC0134 AMLS Coursework (22-23)
Total of 4 tasks are given to be solved and they are named as A1, A2, B1, B2 respectively

A - Binary classification
    A1 - Binary Gender detection: male or female.
    A2 - Binary Emotion detection: smiling or not smiling.
B - Multiclassification
    B1 - Face shape recognition: 5 types of face shapes.
    B2 - Eye color recognition: 5 types of eye colors.

Four folders named A1, A2, B1, B2 contains the separate executive files to train the models required to complete the task
main.py are used to call the executive files saved in the previous four folders to rerun the whole program to complete the four tasks from task A1 to task B2
Please run main.py to start the program.
If you could like to see how each task works, please run A1_new.py, A2_new.py, B1.py, B2.py respectively

PART2 - the role of each file

In A1 folder
    A1_new.py is the executive file to train the CNN model for task A1
    A1.py is the old executive file to train the SVM model for task A1
    ReadCSV.py and Readimg.py are used to save label information saved in csv and save img to dateframe
In A2 folder
    A2_new.py is the executive file to train the CNN model for task A2
    A2.py is the old executive file to train the SVM model for task A2
    ReadCSV.py and Readimg.py are used to save label information saved in csv and save img to dateframe
In B1 folder
    B1.py is the executive file to train the CNN model for task B1
In B2 folder
    B2.py is the executive file to train the CNN model for task B2

4 Trained models are saved in the main directory as model_A1.keras, model_A2.keras, model_B1.keras, model_B2.keras (model_B1.keras and model_b2.keras are computed in remote server and it is too large to be upload here in Github)
Datasets folder includes the datasets (img + label.csv files) needed for training and validation

PART3 - the packages required to run code

appdirs==1.4.4
bcrypt==4.0.1
beautifulsoup4==4.12.2
certifi==2023.7.22
cffi==1.15.1
charset-normalizer==3.2.0
click==8.1.3
colorama==0.4.6
contourpy==1.1.0
cryptography==40.0.2
cycler==0.11.0
Deprecated==1.2.13
dnspython==2.4.2
duckdb==0.7.1
Flask==2.2.3
Flask-Cors==3.0.10
Flask-Limiter==3.3.0
Flask-SSLify==0.1.5
flask-talisman==1.0.0
fonttools==4.42.1
frozendict==2.3.8
html5lib==1.1
idna==3.4
imageio==2.31.3
importlib-resources==5.12.0
itsdangerous==2.1.2
Jinja2==3.1.2
joblib==1.3.2
keras==2.14.0
kiwisolver==1.4.5
lazy_loader==0.3
limits==3.4.0
lxml==4.9.3
markdown-it-py==2.2.0
MarkupSafe==2.1.2
matplotlib==3.7.2
mdurl==0.1.2
multitasking==0.0.11
networkx==3.1
numpy==1.24.2
ordered-set==4.1.0
packaging==23.1
pandas==2.0.0
pandas-datareader==0.10.0
Pillow==10.0.0
pyarrow==11.0.0
pycparser==2.21
Pygments==2.15.1
pylance==0.4.3
pymongo==4.5.0
PyMySQL==1.0.3
pyOpenSSL==23.1.1
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2023.3
PyWavelets==1.4.1
requests==2.31.0
rich==13.3.4
scikit-image==0.21.0
scikit-learn==1.3.0
scipy==1.11.2
six==1.16.0
sklearn==0.0.post9
soupsieve==2.4.1
threadpoolctl==3.2.0
tifffile==2023.8.30
typing_extensions==4.5.0
tzdata==2023.3
urllib3==2.0.4
webencodings==0.5.1
Werkzeug==2.2.3
wrapt==1.15.0
yfinance==0.2.28


