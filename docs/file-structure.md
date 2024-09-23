# How to format our files

## Python

### Headers

#### Docstrings

A file-level docstring should start each Python file, surrounded by triple quotes.

#### Imports

Immediately following this, we should have the imports, in three sections: standard library imports, 
third-party imports, and imports from within this project. Within each of the three sections, we want 
`from module import X` statements to precede the `import module (as y)` statements.

#### File config

##### TYPE_CHECKING

If using `typing.TYPE_CHECKING`, include it first after the import statements.

##### Setting the Logger Name

Immediately after the import statements, define the logger for this module or file with the line:

```python
logger = logging.getLogger(__name__)
```

For scripts (rather than modules), use `enunlg-scripts.subdirectory.filename` as the format for the logger name.
