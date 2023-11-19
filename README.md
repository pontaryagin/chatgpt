# langchain-sample

## Requirements

- OpenAI API key
- python 3.8 and later
- langchain (0.0.229)

## Install

```
pip3 install langchain
```

## Set API key

```
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

## Create index

```
python3 langchain_index.py
```

## Query index

```
python3 langchain_query.py
```

## usage

```
Q:
how parametrized is markov functional model? explain with formula.
A: The passage does not provide specific formulas for the parametrization of the models mentioned. It states that the parametrization of these models can be non-trivial, especially when market prices are not given by Black's formula or when interest rates are very low. It also mentions that the dynamics of forward rates in these models are path-dependent, making efficient numerical implementation difficult. 

Therefore, without further information, it is not possible to explain the parametrization of these models with formulas.
```
