Task 1:

    python3 Assignment2_Task_1.py

I have converted the Jupyter Notebook to a python script using 
$ jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb

When running Task 2 is import the script from Task 1 and run the training trough the code from Task 1 script. This is done by checking whetether the script is imported or not. If imported (running Task 2) the appropriate environment is loaded, and the same goes when running Task 1.

I decided to exclude the continous printing of progress due to it slowing down the script. This can be undone by removing the comments for calls to plot_durations()

Task 2:

    python3 Assignement_2_task_2.py

Task 3:

    python3 try_gym_code_sarsa.py (for code with SARSA implementation)
    python3 try_gym_code.py (for code with QL implementation)

Environment is appended with the scripts, and listed beneath. All scripts use the same environment.

Environment list:
Package                   Version
------------------------- ---------------
anyio                     4.3.0
appnope                   0.1.4
argon2-cffi               23.1.0
argon2-cffi-bindings      21.2.0
arrow                     1.3.0
asttokens                 2.4.1
async-lru                 2.0.4
attrs                     23.2.0
Babel                     2.14.0
beautifulsoup4            4.12.3
bleach                    6.1.0
certifi                   2024.2.2
cffi                      1.16.0
charset-normalizer        3.3.2
cloudpickle               3.0.0
comm                      0.2.1
contourpy                 1.2.0
cycler                    0.12.1
debugpy                   1.8.1
decorator                 5.1.1
defusedxml                0.7.1
exceptiongroup            1.2.0
executing                 2.0.1
Farama-Notifications      0.0.4
fastjsonschema            2.19.1
fonttools                 4.49.0
fqdn                      1.5.1
gym                       0.26.2
gym-notices               0.0.8
gymnasium                 0.29.1
h11                       0.14.0
httpcore                  1.0.4
httpx                     0.27.0
idna                      3.6
importlib-metadata        7.0.1
importlib_resources       6.1.2
ipdb                      0.13.13
ipykernel                 6.29.3
ipython                   8.18.1
isoduration               20.11.0
jedi                      0.19.1
Jinja2                    3.1.3
json5                     0.9.17
jsonpointer               2.4
jsonschema                4.21.1
jsonschema-specifications 2023.12.1
jupyter_client            8.6.0
jupyter_core              5.7.1
jupyter-events            0.9.0
jupyter-lsp               2.2.3
jupyter_server            2.12.5
jupyter_server_terminals  0.5.2
jupyterlab                4.1.2
jupyterlab_pygments       0.3.0
jupyterlab_server         2.25.3
kiwisolver                1.4.5
MarkupSafe                2.1.5
matplotlib                3.8.3
matplotlib-inline         0.1.6
mistune                   3.0.2
nbclient                  0.9.0
nbconvert                 7.16.1
nbformat                  5.9.2
nest-asyncio              1.6.0
notebook                  7.1.1
notebook_shim             0.2.4
numpy                     1.26.4
overrides                 7.7.0
packaging                 23.2
pandocfilters             1.5.1
parso                     0.8.3
pexpect                   4.9.0
pillow                    10.2.0
pip                       24.0
platformdirs              4.2.0
prometheus_client         0.20.0
prompt-toolkit            3.0.43
psutil                    5.9.8
ptyprocess                0.7.0
pure-eval                 0.2.2
pycparser                 2.21
pygame                    2.5.2
Pygments                  2.17.2
pyparsing                 3.1.1
python-dateutil           2.8.2
python-json-logger        2.0.7
PyYAML                    6.0.1
pyzmq                     25.1.2
referencing               0.33.0
requests                  2.31.0
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rpds-py                   0.18.0
Send2Trash                1.8.2
setuptools                58.0.4
six                       1.16.0
sniffio                   1.3.1
soupsieve                 2.5
stack-data                0.6.3
terminado                 0.18.0
tinycss2                  1.2.1
tomli                     2.0.1
torch                     1.8.1
tornado                   6.4
traitlets                 5.14.1
types-python-dateutil     2.8.19.20240106
typing_extensions         4.10.0
uri-template              1.3.0
urllib3                   2.2.1
wcwidth                   0.2.13
webcolors                 1.13
webencodings              0.5.1
websocket-client          1.7.0
zipp                      3.17.0