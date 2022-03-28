<h1 align="center"><a>CS6700 Project</a></h1>

The following environments have been simulated.

- [Acrobot-v1](docs/acrobot.md)
- [Taxi-v3](docs/taxi.md)
- [KBC](docs/kbc.md)

The link to the AIcrowd challenge page: https://www.aicrowd.com/challenges/rl-project-2021

> Recommended software versions - `python 3.8`, `pip > 21.1.1`

Clone the starter kit repository and install the dependencies.

```bash
git clone https://github.com/narendv/CS6700_project/
cd CS6700_project
pip install -U -r requirements.txt

```


## Run and Evaluate your agents

Run the file [`run.py`](run.py) to test the agents.

To run the evaluation locally, run the following commands.

```bash
ENV_NAME="acrobot" python run.py
ENV_NAME="taxi" python run.py
ENV_NAME="kbca" python run.py
ENV_NAME="kbcb" python run.py
ENV_NAME="kbcc" python run.py

```
