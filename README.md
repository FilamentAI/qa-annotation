# qa-annotation
The Streamlit tool for the Filament Synthetic QA Pairs project, used to annotate generated data. 

## Setup
You should install the project requirements via Pip using `pip install -r requirements.txt` and add some data to the file "data/generated\_data/generated\_data.json". An example generated\_data.json is provided.

You should also create a `password` file containing a SHA 512-encoded password for better security (and you'll no longer need to pass "--insecure"). 

Subset-specific data should be placed in `subset_1_generated_data.json` and so on.

## Running the tool 
Once setup is done, the tool can be run with the command `streamlit run qa_annotation_tool.py -- --insecure` (we pass the "--insecure" flag since no password file is included in the tool.)

Note that any arguments to the tool must follow the `--` above, else they'll be given to Streamlit instead.

You can pass three arguments to it, and get descriptions of them by passing "--help" to the script. 

* "--insecure" - Allows for running without a password file.

* "--preliminary" - Indicates that the tool is running in "preliminary mode", changing the data it gets and where it outputs to. Mutually exclusive with "--subset". 

* "--subset" - Specifies a subset of the annotators that we wish to run for. E.g. If you have 30 annotators are split into 10 subsets of 3 each, you would run the tool once for each subset with "--subset 1", "--subset 2", etc. and each would be presented with different data (corresponding to the data marked as being for that subset). 

## Output
The tool creates various JSON files per user, in `data/generated_questions`. An example user output is provided.
