For logging we use a [Python's standard library `logging`](https://docs.python.org/3.9/library/logging.html) 
[through Hydra](https://hydra.cc/docs/tutorials/basic/running_your_app/logging/).
Hydra manages our configs and commandline args and takes care of logging with sensible defaults.

If you need to run a script with debug messages printed, add `hydra.verbose=true` to the commandline.

```bash
python scriptname.py hydra.verbose=true
```

If you have different named loggers for different parts of your code, 
you can also pass the names of the loggers that you would like to have printed at debug level.
For example, suppose you have a logger named `trainer` and one named `MyModel` somewhere in your code, 
then you could do the following:

```bash
python scriptname.py hydra.verbose=[__main__,trainer,MyModel]
```