This project started as random shower thought so I decided to give it a try.

This app is aimed to be (sort of) an AI agent to aid students schedule their work ahead and look up resources online. More to come.

## Installation

Install the requirements.

```shell
pip install -r requirements.txt
```

**Warning**: Further requirements may change heavily in the future.

`webrtcdav` requires MSVC and C++ Build Tools. Installing them will require about 3 GB of system space. Get it with [Microsoft C++ Build Tools.](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Spacey models must be installed using a Python command, not `pip`:

```shell
python -m spacy download en_core_web_sm
```

Small versions of speech-to-word and TTS models have been used. These might still be a lot for your computer.


