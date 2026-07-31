import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def execute_notebook_file():
    nb_path = os.path.join('notebooks', 'week5_nlp.ipynb')
    print(f"Executing notebook {nb_path} to generate outputs...")

    # Change working directory to notebooks directory so relative paths in notebook work
    cwd = os.path.abspath('notebooks')

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': cwd}})

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"Successfully executed notebook and updated {nb_path} with live outputs!")

if __name__ == '__main__':
    execute_notebook_file()
