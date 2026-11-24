# Typst Util

To run this utility:
+ [The data folder](./data/) is for the json data, it can hold multiple json files, but the json files need to follow the naming convention, `experiment_name.metric.json`. `experiment_name` and `metric` is used to find the correct files.
+ The script resolves `./data/` relative to the current working directory and imports [parser.py](./parser.py) from its own directory, so run it from [this folder](.) — that is, `src/marl/typst`, not the [root](../../../).
+ Then execute:
    ```console
    uv run master.py OUTPUT_FILE_NAME --experiments EXPERIMENT_NAME.. --metric METRIC
    ```
    Example with files `data/exp1.accuracy.json` and `data/exp2.accuracy.json`:
    ```console
    uv run master.py output --experiments exp1 exp2 --metric accuracy
    Created typst plot at: 'data/output.txt'
    ```

Use `--help` to see all the arguments:
```console
uv run master.py --help
```

That is all.
